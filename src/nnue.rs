// The network: reading it, keeping it up to date, and asking it for a score.
//
// Shape: (768 x 32 -> 512) x 2 -> 1, plus a PSQT term that skips the hidden
// layer, both read through four output buckets chosen by piece count. One
// hidden layer, clipped ReLU, everything in i16.
//
// The file is a raw dump of the weights in declaration order with no header, so
// the struct below IS the format. Its size is checked on load against the byte
// count, which is the cheapest possible guard against reading a file that is
// almost but not quite this network.

use crate::bitboard::Bitboard;
use crate::types::*;

/// King buckets. Note what this number means: the bucket is the king's own
/// square after folding, not a coarse region, so EVERY king move changes it.
/// That is what makes the refresh cache below load-bearing rather than an
/// optimisation -- see the comment there.
pub const INPUT_BUCKETS: usize = 32;
pub const HIDDEN: usize = 512;
pub const OUT_BUCKETS: usize = 4;
pub const NUM_FEATURES: usize = INPUT_BUCKETS * 2 * 6 * 64; // 24576

/// Weights are stored multiplied by this, and the hidden layer is clipped to
/// it. Both facts come from the training quantisation and neither is free to
/// change here.
pub const SCALE: i32 = 1024;

#[repr(C)]
pub struct Network {
    pub l0w: [[i16; HIDDEN]; NUM_FEATURES],
    pub psqt: [[i16; OUT_BUCKETS]; NUM_FEATURES],
    pub l1w: [[i16; 2 * HIDDEN]; OUT_BUCKETS],
    pub l1b: [i16; OUT_BUCKETS],
}

pub const NET_BYTES: usize = std::mem::size_of::<Network>();

/// The win-rate model the network was trained against.
///
/// These constants are not a choice and not a default: they are the output of a
/// search over hundreds of validated training runs, which is why they carry
/// sixteen digits. Changing one here without changing it in training makes the
/// program announce probabilities the network was never taught to produce.
///
/// What makes them usable directly on our score, with no conversion at all, is
/// that the quantisation stores each weight already multiplied by the training
/// scale. Measured on ten thousand positions with known results, the fit gives
/// k = 0.9939 for this network -- so the raw score IS this model's centipawn.
/// For a network trained at a different scale it would not be, and this would
/// quietly lie.
pub const WDL_OFFSET: f64 = 285.2706341467852;
pub const WDL_SCALING: f64 = 295.6539508488627;

/// Win, draw and loss in thousandths, from the side to move's point of view.
pub fn wdl(score: i32) -> (i32, i32, i32) {
    let sig = |x: f64| 1.0 / (1.0 + (-x).exp());
    let cp = score as f64;
    let w = sig((cp - WDL_OFFSET) / WDL_SCALING);
    let l = sig((-cp - WDL_OFFSET) / WDL_SCALING);
    let wi = (1000.0 * w).round() as i32;
    let li = (1000.0 * l).round() as i32;
    (wi, 1000 - wi - li, li)
}

/// The one network, set once at startup.
///
/// Global rather than carried around because every path that moves a piece
/// needs it, and threading it through the board would put a lifetime on the
/// position itself. Set once and never replaced, so no locking is involved
/// after startup.
static NET: std::sync::OnceLock<Box<Network>> = std::sync::OnceLock::new();

pub fn net() -> Option<&'static Network> {
    NET.get().map(|b| &**b)
}

/// Returns false if a network was already installed.
pub fn install(n: Box<Network>) -> bool {
    NET.set(n).is_ok()
}

/// Read the weights from a file.
///
/// The allocation is made and then overwritten wholesale rather than parsed
/// field by field: the file is exactly this struct's bytes, and any parsing
/// step would be a second place for the layout to be described, free to drift
/// from the first.
pub fn load(path: &str) -> std::io::Result<Box<Network>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() != NET_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("network is {} bytes, expected {}", bytes.len(), NET_BYTES),
        ));
    }
    // SAFETY: `Network` is `repr(C)` and made entirely of `i16`, so every bit
    // pattern is a valid value and there is no padding to leave uninitialised.
    // The length was just checked. Little-endian is assumed, as the file was
    // written on the same class of machine that reads it.
    unsafe {
        let layout = std::alloc::Layout::new::<Network>();
        let raw = std::alloc::alloc_zeroed(layout) as *mut Network;
        if raw.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), raw as *mut u8, NET_BYTES);
        Ok(Box::from_raw(raw))
    }
}

#[inline(always)]
fn flip_rank(s: Square) -> Square {
    s ^ 56
}

#[inline(always)]
fn flip_file(s: Square) -> Square {
    s ^ 7
}

/// Which king bucket, and whether this perspective reads the board mirrored.
///
/// Both are functions of the king square and both have to travel together: a
/// king on d1 and a king on e1 fold to the same bucket while needing opposite
/// mirrors, so anything keyed on the bucket alone will hand one perspective the
/// other's values and every square comes back flipped.
#[inline(always)]
pub fn bucket_and_mirror(perspective: Color, king_sq: Square) -> (usize, bool) {
    let ks = if perspective == Color::Black {
        flip_rank(king_sq)
    } else {
        king_sq
    };
    let mirror = file_of(ks) >= 4;
    let ks = if mirror { flip_file(ks) } else { ks };
    (4 * rank_of(ks) as usize + file_of(ks) as usize, mirror)
}

/// The input number for one piece, seen from one perspective.
///
/// Black's perspective sees the board upside down and with the colours
/// exchanged, so that both sides present the network with the same problem.
/// Taking the bucket and mirror as arguments rather than recomputing them keeps
/// this honest: the caller has already decided which bucket it is working in,
/// and a feature computed under a different one would be silently wrong.
#[inline(always)]
pub fn feature(
    perspective: Color,
    bucket: usize,
    mirror: bool,
    piece_color: Color,
    pt: PieceType,
    s: Square,
) -> usize {
    let (pc, s) = if perspective == Color::Black {
        (piece_color.opp(), flip_rank(s))
    } else {
        (piece_color, s)
    };
    let s = if mirror { flip_file(s) } else { s };
    s as usize + pt.idx() * 64 + pc.idx() * 384 + bucket * 768
}

/// One side's half of the hidden layer, plus its PSQT running total.
#[derive(Clone)]
pub struct Half {
    pub values: [i16; HIDDEN],
    pub psqt: [i32; OUT_BUCKETS],
    pub bucket: usize,
    pub mirror: bool,
}

impl Half {
    fn empty() -> Self {
        Half {
            values: [0; HIDDEN],
            psqt: [0; OUT_BUCKETS],
            bucket: 0,
            mirror: false,
        }
    }
}

#[derive(Clone)]
pub struct Accumulator {
    pub half: [Half; 2], // indexed by perspective
}

/// Build one perspective from nothing.
pub fn rebuild_half(
    net: &Network,
    pieces: &[[Bitboard; 6]; 2],
    king_sq: Square,
    perspective: Color,
) -> Half {
    let (bucket, mirror) = bucket_and_mirror(perspective, king_sq);
    let mut h = Half::empty();
    h.bucket = bucket;
    h.mirror = mirror;
    for c in [Color::White, Color::Black] {
        for pt in ALL_PIECES {
            let mut bb = pieces[c.idx()][pt.idx()];
            while bb != 0 {
                let s = bb.trailing_zeros() as Square;
                bb &= bb - 1;
                let f = feature(perspective, bucket, mirror, c, pt, s);
                apply(&mut h.values, &net.l0w[f], true);
                for b in 0..OUT_BUCKETS {
                    h.psqt[b] += net.psqt[f][b] as i32;
                }
            }
        }
    }
    h
}

impl Accumulator {
    pub fn empty() -> Self {
        Accumulator {
            half: [Half::empty(), Half::empty()],
        }
    }

    /// Rebuild any perspective whose king has moved into a different bucket, or
    /// across the middle where the mirror flips.
    ///
    /// Called after the move has been applied, so the values it finds were
    /// updated under the OLD bucket and cannot be patched -- under a new bucket
    /// the same piece on the same square is a different input number, so every
    /// one of them is wrong at once. Rebuilding is the only correct answer;
    /// making it cheap is a separate problem.
    ///
    /// Returns how many perspectives were rebuilt, so the cost can be measured
    /// rather than assumed.
    pub fn refresh(
        &mut self,
        net: &Network,
        pieces: &[[Bitboard; 6]; 2],
        kings: [Square; 2],
    ) -> u32 {
        let mut done = 0;
        for p in [Color::White, Color::Black] {
            let (bucket, mirror) = bucket_and_mirror(p, kings[p.idx()]);
            let h = &mut self.half[p.idx()];
            if h.bucket == bucket && h.mirror == mirror {
                continue;
            }
            with_cache(|c| c.refresh(net, pieces, p, bucket, mirror, h));
            done += 1;
        }
        done
    }

    /// Build both perspectives from nothing. Correct always, and slow enough
    /// that it is only used to start a position and as the reference the
    /// incremental path is checked against.
    pub fn fresh(net: &Network, pieces: &[[Bitboard; 6]; 2], kings: [Square; 2]) -> Self {
        Accumulator {
            half: [
                rebuild_half(net, pieces, kings[0], Color::White),
                rebuild_half(net, pieces, kings[1], Color::Black),
            ],
        }
    }

    /// One piece appearing on or leaving a square, for both perspectives.
    #[inline]
    pub fn update(&mut self, net: &Network, c: Color, pt: PieceType, s: Square, add: bool) {
        for p in [Color::White, Color::Black] {
            let h = &mut self.half[p.idx()];
            let f = feature(p, h.bucket, h.mirror, c, pt, s);
            apply(&mut h.values, &net.l0w[f], add);
            let sign = if add { 1 } else { -1 };
            for b in 0..OUT_BUCKETS {
                h.psqt[b] += sign * net.psqt[f][b] as i32;
            }
        }
    }

    /// The score, from the side to move's point of view.
    pub fn eval(&self, net: &Network, stm: Color, piece_count: u32) -> i32 {
        let bucket = ((piece_count - 1) / 8) as usize;
        let us = &self.half[stm.idx()];
        let them = &self.half[stm.opp().idx()];
        let w = &net.l1w[bucket];

        let mut out = net.l1b[bucket] as i32 * SCALE;
        out += output(&us.values, &w[..HIDDEN]);
        out += output(&them.values, &w[HIDDEN..]);

        us.psqt[bucket] - them.psqt[bucket] + out / SCALE
    }
}

/// Add or subtract one column of weights, through the vector path if there is
/// one.
#[inline]
fn apply(dst: &mut [i16; HIDDEN], col: &[i16; HIDDEN], add: bool) {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: the feature was just checked, and both slices are HIDDEN long
        // by their types.
        unsafe {
            if add {
                simd::add(dst, col);
            } else {
                simd::sub(dst, col);
            }
            return;
        }
    }
    if add {
        for i in 0..HIDDEN {
            dst[i] += col[i];
        }
    } else {
        for i in 0..HIDDEN {
            dst[i] -= col[i];
        }
    }
}

/// `sum of clamp(acc[i], 0, SCALE) * w[i]`.
#[inline]
fn output(acc: &[i16; HIDDEN], w: &[i16]) -> i32 {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: feature checked; `acc` is HIDDEN long and so is `w`.
        unsafe {
            return simd::output(acc, w);
        }
    }
    let mut total = 0i32;
    for i in 0..HIDDEN {
        total += (acc[i] as i32).clamp(0, SCALE) * w[i] as i32;
    }
    total
}

#[cfg(target_arch = "x86_64")]
mod simd {
    use super::{HIDDEN, SCALE};
    use std::arch::x86_64::*;

    /// # Safety
    /// AVX2 must be available. Both slices are HIDDEN long.
    #[target_feature(enable = "avx2")]
    pub unsafe fn add(dst: &mut [i16; HIDDEN], src: &[i16; HIDDEN]) {
        let mut i = 0;
        while i + 16 <= HIDDEN {
            let a = _mm256_loadu_si256(dst.as_ptr().add(i) as *const __m256i);
            let b = _mm256_loadu_si256(src.as_ptr().add(i) as *const __m256i);
            _mm256_storeu_si256(
                dst.as_mut_ptr().add(i) as *mut __m256i,
                _mm256_add_epi16(a, b),
            );
            i += 16;
        }
    }

    /// # Safety
    /// As `add`.
    #[target_feature(enable = "avx2")]
    pub unsafe fn sub(dst: &mut [i16; HIDDEN], src: &[i16; HIDDEN]) {
        let mut i = 0;
        while i + 16 <= HIDDEN {
            let a = _mm256_loadu_si256(dst.as_ptr().add(i) as *const __m256i);
            let b = _mm256_loadu_si256(src.as_ptr().add(i) as *const __m256i);
            _mm256_storeu_si256(
                dst.as_mut_ptr().add(i) as *mut __m256i,
                _mm256_sub_epi16(a, b),
            );
            i += 16;
        }
    }

    /// Clipped ReLU and the dot product in one pass.
    ///
    /// `_mm256_madd_epi16` multiplies sixteen i16 pairs and adds them in pairs
    /// into eight i32 lanes, which is exactly a dot product. Keeping eight
    /// separate lanes also keeps overflow far away: each carries an eighth of
    /// the total where a scalar loop puts everything in one i32.
    ///
    /// # Safety
    /// AVX2 must be available. `acc` is HIDDEN long and `w` at least HIDDEN.
    #[target_feature(enable = "avx2")]
    pub unsafe fn output(acc: &[i16; HIDDEN], w: &[i16]) -> i32 {
        let zero = _mm256_setzero_si256();
        let top = _mm256_set1_epi16(SCALE as i16);
        let mut sum = _mm256_setzero_si256();
        let mut i = 0;
        while i + 16 <= HIDDEN {
            let x = _mm256_loadu_si256(acc.as_ptr().add(i) as *const __m256i);
            let wv = _mm256_loadu_si256(w.as_ptr().add(i) as *const __m256i);
            let c = _mm256_min_epi16(_mm256_max_epi16(x, zero), top);
            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(c, wv));
            i += 16;
        }
        let lo = _mm256_castsi256_si128(sum);
        let hi = _mm256_extracti128_si256(sum, 1);
        let mut s = _mm_add_epi32(lo, hi);
        s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
        s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b10_11_00_01));
        _mm_cvtsi128_si32(s)
    }
}

// ---------------------------------------------------------------------------
// Refresh cache
//
// The bucket here is the folded king square itself, so every king move
// invalidates one whole perspective, and king moves are around a quarter of all
// moves. Rebuilding from nothing means adding all thirty-two pieces back, and
// each of those reads a kilobyte from a twenty-five megabyte table, so the cost
// is memory rather than arithmetic and no amount of vectorising touches it.
//
// The cache turns the rebuild into a difference. Keep one accumulator per
// (perspective, bucket, mirror) alongside the piece placement that produced it;
// on a refresh, start from that and apply only the pieces that have moved
// since. A piece that stayed put contributes the same feature and needs no work
// at all, which is the whole point.
//
// The invariant that makes it safe: an entry always holds an accumulator that
// exactly matches its stored placement. Update both together or neither, and a
// stale entry becomes impossible rather than merely unlikely -- a quietly wrong
// accumulator shows up as an evaluation that is subtly off in rare positions,
// which is the hardest kind of bug to trace back to here.
//
// Keyed by the mirror as well as the bucket, and it has to be: a king on d1 and
// a king on e1 fold to the same bucket while needing opposite mirrors. Keyed by
// bucket alone, an entry built for one is handed to the other and every square
// comes back flipped.
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct CacheEntry {
    values: [i16; HIDDEN],
    psqt: [i32; OUT_BUCKETS],
    /// The placement `values` was built from.
    pieces: [[Bitboard; 6]; 2],
    used: bool,
}

pub struct RefreshCache {
    entries: Vec<CacheEntry>,
}

impl RefreshCache {
    pub fn new() -> Self {
        RefreshCache {
            entries: vec![
                CacheEntry {
                    values: [0; HIDDEN],
                    psqt: [0; OUT_BUCKETS],
                    pieces: [[0; 6]; 2],
                    used: false,
                };
                2 * 2 * INPUT_BUCKETS
            ],
        }
    }

    #[inline]
    fn index(perspective: Color, bucket: usize, mirror: bool) -> usize {
        (perspective.idx() * 2 + mirror as usize) * INPUT_BUCKETS + bucket
    }

    /// Bring one perspective up to date, starting from whatever this bucket was
    /// last seen holding. Returns how many piece updates it took, so what the
    /// cache saves can be measured rather than assumed.
    pub fn refresh(
        &mut self,
        net: &Network,
        pieces: &[[Bitboard; 6]; 2],
        perspective: Color,
        bucket: usize,
        mirror: bool,
        dst: &mut Half,
    ) -> usize {
        let e = &mut self.entries[Self::index(perspective, bucket, mirror)];
        if !e.used {
            // Nothing here yet. An empty accumulator for this network is all
            // zeros -- there is no hidden layer bias to seed it from -- so the
            // entry starts from an empty board and the first refresh pays the
            // full price. Every later one starts from this.
            e.values = [0; HIDDEN];
            e.psqt = [0; OUT_BUCKETS];
            e.pieces = [[0; 6]; 2];
            e.used = true;
        }

        let mut touched = 0usize;
        for c in [Color::White, Color::Black] {
            for pt in ALL_PIECES {
                let now = pieces[c.idx()][pt.idx()];
                let before = e.pieces[c.idx()][pt.idx()];
                let mut gone = before & !now;
                while gone != 0 {
                    let sq = gone.trailing_zeros() as Square;
                    gone &= gone - 1;
                    let f = feature(perspective, bucket, mirror, c, pt, sq);
                    apply(&mut e.values, &net.l0w[f], false);
                    for b in 0..OUT_BUCKETS {
                        e.psqt[b] -= net.psqt[f][b] as i32;
                    }
                    touched += 1;
                }
                let mut arrived = now & !before;
                while arrived != 0 {
                    let sq = arrived.trailing_zeros() as Square;
                    arrived &= arrived - 1;
                    let f = feature(perspective, bucket, mirror, c, pt, sq);
                    apply(&mut e.values, &net.l0w[f], true);
                    for b in 0..OUT_BUCKETS {
                        e.psqt[b] += net.psqt[f][b] as i32;
                    }
                    touched += 1;
                }
                // Placement and values move together, always.
                e.pieces[c.idx()][pt.idx()] = now;
            }
        }

        dst.values.copy_from_slice(&e.values);
        dst.psqt = e.psqt;
        dst.bucket = bucket;
        dst.mirror = mirror;
        touched
    }
}

thread_local! {
    /// Per thread rather than shared: two threads on one cache would each
    /// invalidate the other on every king move, which costs more than having no
    /// cache at all.
    static CACHE: std::cell::RefCell<RefreshCache> =
        std::cell::RefCell::new(RefreshCache::new());
}

pub fn with_cache<R>(f: impl FnOnce(&mut RefreshCache) -> R) -> R {
    CACHE.with(|c| f(&mut c.borrow_mut()))
}

/// Whether the vector path may be used. Decided once.
#[cfg(target_arch = "x86_64")]
pub fn has_avx2() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        // An escape hatch so the two paths can be compared on one machine
        // without rebuilding -- which is how you find out whether the vector
        // path is right, not just fast.
        if std::env::var_os("HALF2K_NO_SIMD").is_some() {
            return false;
        }
        std::is_x86_feature_detected!("avx2")
    })
}

#[cfg(not(target_arch = "x86_64"))]
pub fn has_avx2() -> bool {
    false
}
