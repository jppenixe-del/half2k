use crate::moves::{Move, MoveFlag};
use crate::types::{PieceType, Square};
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum Bound {
    Exact,
    Lower,
    Upper,
    /// No bound at all: the entry carries only a cached static evaluation.
    ///
    /// The search computes a static eval and then has fourteen ways to leave
    /// the node before it ever reaches the store at the bottom -- razoring,
    /// null move, probcut and the rest. Every one of those threw away an
    /// evaluation that now costs ~21000 instructions, and the next visit paid
    /// for it again. Worse, the loss was invisible: it shows up as a MISSING
    /// entry, not as an entry without an eval, so it counted as a first visit.
    ///
    /// Writing the eval the moment it is computed makes the work survive the
    /// early exits. Such an entry must never produce a cutoff -- it has no
    /// score to stand behind.
    NoBound,
}

/// Sentinel for "no cached static eval in this entry" -- e.g. entries
/// stored while in check, where static eval is meaningless and never
/// computed. i16::MIN is never a real eval (evals are always clamped
/// to fit comfortably inside i16 well above this), so it's safe as a
/// dedicated "absent" marker.
pub const TT_EVAL_NONE: i16 = i16::MIN;

#[derive(Copy, Clone)]
pub struct TtEntry {
    pub key: u64,
    pub depth: i32,
    pub score: i32,
    pub bound: Bound,
    pub best: Option<Move>,
    /// TTPV: was this entry written by a search at a PV node (full
    /// alpha-beta window), as opposed to a null/scout window? A later
    /// probe that only reaches this position via a scout search still
    /// knows it was "important" once -- used to reduce LESS via LMR at
    /// that position, since a genuinely quiet position wouldn't have
    /// earned full-window treatment before.
    pub pv: bool,
    /// Cached static eval (the full `evaluate(board)`, uncorrected) at
    /// the position this entry was written for. TT_EVAL_NONE if none
    /// was available (in check). Reused by the search on a later probe
    /// of the same position to skip recomputing the full eval.
    pub static_eval: i16,
}

/// Lock-free slot: `data` packs the whole entry into 64 bits (see
/// encode_data/decode_data), `key_xor_data` stores `key ^ data`. A reader
/// recomputes `key_xor_data ^ data` and compares against the key it's
/// looking for -- on a torn read (another thread wrote concurrently, so
/// the two loads came from different writes), the XOR essentially never
/// matches the real key, so the probe is safely treated as a miss instead
/// of returning garbage. This is what lets the table be shared and
/// lock-free under Lazy SMP -- no locks, so no contention between search
/// threads.
///
/// `gen` is a separate atomic byte rather than a few more bits stolen from
/// `data`. `data` is already packed to 61 of 64 bits (see the layout note
/// below) -- fitting a generation in the 3 that are left means a cycle of
/// 8, which wraps within a single long game and starts confusing recent
/// entries for stale ones. A whole byte costs a little more memory per
/// slot and buys a cycle of 256, which does not wrap inside any game this
/// engine will play.
impl TtEntry {
    /// Is there a score behind this entry?
    ///
    /// `Bound::NoBound` entries carry only a cached static eval; their `score`
    /// field is meaningless. Anything that reads `score` outside the bound
    /// match has to ask this first -- two places did not, and a score of 0
    /// read as a real bound cost 33% more nodes.
    #[inline]
    pub fn has_bound(&self) -> bool {
        !matches!(self.bound, Bound::NoBound)
    }
}

struct TtSlot {
    key_xor_data: AtomicU64,
    data: AtomicU64,
    gen: std::sync::atomic::AtomicU8,
}

/// 2026-08-03: this table used to be one slot per index, always-replace,
/// no generation at all. Two references were read specifically on this
/// point and both do the same thing, just with different constants: a
/// small bucket (3-4 entries) per index, a generation counter incremented
/// once per real move (not per node), and a replacement rule that picks
/// the WORST entry in the bucket by `depth - penalty * how_stale`, so a
/// shallow entry from three moves ago loses to a fresh one before a deep
/// entry from one move ago does.
///
/// Always-replace was not an oversight -- a bucketed, aged version was
/// built and measured against it before, and lost by 35-45% on node-count-
/// to-fixed-depth. That comparison was flagged as unsound in the same
/// commit that measured it (wrong metric for this project, and the table
/// under test was silently half the size it was asked for) and was never
/// re-run. This is that re-run, at the size actually requested, and judged
/// on games rather than nodes to a fixed depth.
const BUCKET: usize = 3;
/// Quality lost per generation of staleness. Other engines use 2 to 4
/// (out of a 32-generation cycle instead of this table's 256) -- picked 3
/// here as a genuine middle value pending a real measurement, not because
/// splitting the difference is principled on its own.
const STALE_PENALTY: i32 = 3;

const EVAL_ONLY_DEPTH: i32 = -8;

pub struct TranspositionTable {
    slots: Vec<[TtSlot; BUCKET]>,
    mask: usize,
    current_gen: std::sync::atomic::AtomicU8,
}

fn encode_move(mv: Option<Move>) -> u64 {
    let m = match mv {
        None => return 0, // from==to==0 ("a1a1") is never a real move -- sentinel for None
        Some(m) => m,
    };
    let promo: u64 = match m.promotion {
        Some(PieceType::Knight) => 1,
        Some(PieceType::Bishop) => 2,
        Some(PieceType::Rook) => 3,
        Some(PieceType::Queen) => 4,
        _ => 0,
    };
    let flag: u64 = match m.flag {
        MoveFlag::Quiet => 0,
        MoveFlag::DoublePush => 1,
        MoveFlag::Capture => 2,
        MoveFlag::EnPassant => 3,
        MoveFlag::CastleKing => 4,
        MoveFlag::CastleQueen => 5,
    };
    (m.from as u64) | ((m.to as u64) << 6) | (promo << 12) | (flag << 15)
}

fn decode_move(bits: u64) -> Option<Move> {
    let from = (bits & 0x3F) as Square;
    let to = ((bits >> 6) & 0x3F) as Square;
    if from == to {
        return None;
    }
    let promo = match (bits >> 12) & 0x7 {
        1 => Some(PieceType::Knight),
        2 => Some(PieceType::Bishop),
        3 => Some(PieceType::Rook),
        4 => Some(PieceType::Queen),
        _ => None,
    };
    let flag = match (bits >> 15) & 0x7 {
        0 => MoveFlag::Quiet,
        1 => MoveFlag::DoublePush,
        2 => MoveFlag::Capture,
        3 => MoveFlag::EnPassant,
        4 => MoveFlag::CastleKing,
        5 => MoveFlag::CastleQueen,
        _ => MoveFlag::Quiet,
    };
    Some(Move { from, to, promotion: promo, flag })
}

// Packing of `data` into 64 bits (see TtSlot doc comment above for the
// lock-free XOR-trick this feeds into):
//   bits  0..18  mv_bits       (18 bits -- encode_move)
//   bits 18..34  score16       (16 bits, signed)
//   bits 34..42  depth8        ( 8 bits, signed -- search depth never
//                                gets remotely close to the i8 range;
//                                see MAX_PLY/negamax's ply>=MAX_PLY-1
//                                guard, which hard-stops recursion at
//                                127. A pathological `go depth 200`+
//                                request just clamps harmlessly here,
//                                same as score16 already did/does.)
//   bits 42..44  bound_bits    ( 2 bits)
//   bit  44      pv_bit        ( 1 bit)
//   bits 45..61  static_eval16 (16 bits, signed -- cached full eval)
//   bits 61..64  unused        ( 3 bits free)
//
// TT structure note. This WAS a single-slot, always-replace table -- no
// generation, no bucket. A 4-way set-associative cluster with generation
// aging was tried once and measured worse, but that comparison was flagged
// the same day as unsound (wrong metric, and the table under test was
// silently half its requested size) and the promised re-run never
// happened. This file is that re-run: a 3-way bucket, a real generation
// counter (`TranspositionTable::increase_gen`, `TtSlot::gen`), judged on
// games at the sizes actually requested rather than nodes to a fixed
// depth. See the struct-level doc comments below for the design itself.
fn encode_data(depth: i32, score: i32, bound: Bound, best: Option<Move>, pv: bool, static_eval: i16) -> u64 {
    let mv_bits = encode_move(best);
    let score16 = (score.clamp(i16::MIN as i32, i16::MAX as i32) as i16 as u16) as u64;
    let depth8 = (depth.clamp(i8::MIN as i32, i8::MAX as i32) as i8 as u8) as u64;
    let bound_bits: u64 = match bound {
        Bound::Exact => 0,
        Bound::Lower => 1,
        Bound::Upper => 2,
        Bound::NoBound => 3,
    };
    let pv_bit: u64 = if pv { 1 } else { 0 };
    let eval16 = (static_eval as u16) as u64;
    mv_bits | (score16 << 18) | (depth8 << 34) | (bound_bits << 42) | (pv_bit << 44) | (eval16 << 45)
}

fn decode_data(data: u64) -> (i32, i32, Bound, Option<Move>, bool, i16) {
    let mv_bits = data & 0x3FFFF;
    let score = ((data >> 18) & 0xFFFF) as u16 as i16 as i32;
    let depth = ((data >> 34) & 0xFF) as u8 as i8 as i32;
    let bound = match (data >> 42) & 0x3 {
        0 => Bound::Exact,
        1 => Bound::Lower,
        2 => Bound::Upper,
        _ => Bound::NoBound,
    };
    let pv = (data >> 44) & 1 == 1;
    let static_eval = ((data >> 45) & 0xFFFF) as u16 as i16;
    (depth, score, bound, decode_move(mv_bits), pv, static_eval)
}

impl TranspositionTable {
    pub fn new(mb: usize) -> Self {
        let bytes = mb * 1024 * 1024;
        let bucket_size = std::mem::size_of::<[TtSlot; BUCKET]>();
        // The index is masked, so the bucket count must be a power of two,
        // and it must not exceed what was asked for. Round DOWN to the
        // previous power of two -- which, when the count already IS one,
        // keeps it.
        //
        // 2026-07-25: this used to be `next_power_of_two() / 2`, meaning that
        // whenever the requested size divided into an exact power of two --
        // which is every realistic setting, since hash is always asked for in
        // powers of two -- the table was silently halved. `Hash=32` built a
        // 16MB table. Every measurement ever taken on this engine at a given
        // Hash value was really taken at half of it.
        let requested = (bytes / bucket_size).max(1024);
        let count = 1usize << requested.ilog2();
        let mut slots = Vec::with_capacity(count);
        for _ in 0..count {
            slots.push(std::array::from_fn(|_| TtSlot {
                key_xor_data: AtomicU64::new(0),
                data: AtomicU64::new(0),
                gen: std::sync::atomic::AtomicU8::new(0),
            }));
        }
        TranspositionTable { slots, mask: count - 1, current_gen: std::sync::atomic::AtomicU8::new(0) }
    }

    /// Once per real move (see the call site in `cmd_go`), not once per
    /// node. Wraps at 256 -- staleness in `replace_index` is computed as a
    /// wrapping difference, the same way both references handle their own
    /// (smaller) cycles, so the wrap itself never confuses old for new.
    pub fn increase_gen(&self) {
        self.current_gen.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    /// Pede o balde a' memoria antes de ele ser sondado.
    ///
    /// A tabela e' maior do que qualquer cache -- e' esse o objectivo -- por
    /// isso praticamente toda a sondagem e' uma ida a' DRAM, e no perfil essas
    /// esperas apareciam com 7,8% do tempo, escondidas nas leituras atomicas
    /// do balde. Mas o indice e' sabido assim que o lance esta' feito, muito
    /// antes de o filho sondar: pedir ali deixa a latencia ser coberta pelo
    /// trabalho que fica no meio.
    ///
    /// Duas linhas de cache: o balde sao tres entradas de 24 bytes.
    #[inline]
    pub fn prefetch(&self, key: u64) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
            let idx = (key as usize) & self.mask;
            let p = self.slots.as_ptr().add(idx) as *const i8;
            _mm_prefetch(p, _MM_HINT_T0);
            _mm_prefetch(p.add(64), _MM_HINT_T0);
        }
    }

    pub fn probe(&self, key: u64) -> Option<TtEntry> {
        let idx = (key as usize) & self.mask;
        let bucket = &self.slots[idx];
        let gen = self.current_gen.load(Ordering::Relaxed);
        for slot in bucket {
            let data = slot.data.load(Ordering::Relaxed);
            let key_xor = slot.key_xor_data.load(Ordering::Relaxed);
            if key_xor ^ data != key {
                continue;
            }
            // A read is also evidence this entry still matters. Refreshing
            // its generation here protects a position the search keeps
            // transposing back into, even on a ply that only reads it and
            // never has reason to store over it -- the idea a second
            // reference's TT has that the first one's does not.
            slot.gen.store(gen, Ordering::Relaxed);
            let (depth, score, bound, best, pv, static_eval) = decode_data(data);
            return Some(TtEntry { key, depth, score, bound, best, pv, static_eval });
        }
        None
    }

    /// Which of the `BUCKET` slots to write into: an exact key match if one
    /// exists (so a re-store of the same position updates in place rather
    /// than duplicating across the bucket), otherwise the slot with the
    /// lowest `depth - STALE_PENALTY * staleness` -- the shallowest entry
    /// that has also gone the longest without being touched, ahead of a
    /// deep entry from a couple of moves ago.
    #[inline]
    fn replace_index(bucket: &[TtSlot; BUCKET], key: u64, gen: u8) -> usize {
        let mut best_idx = 0;
        let mut best_quality = i32::MAX;
        for (i, slot) in bucket.iter().enumerate() {
            let data = slot.data.load(Ordering::Relaxed);
            let key_xor = slot.key_xor_data.load(Ordering::Relaxed);
            if key_xor ^ data == key {
                return i;
            }
            let depth = ((data >> 34) & 0xFF) as u8 as i8 as i32;
            let slot_gen = slot.gen.load(Ordering::Relaxed);
            let staleness = gen.wrapping_sub(slot_gen) as i32;
            let quality = depth - STALE_PENALTY * staleness;
            if quality < best_quality {
                best_quality = quality;
                best_idx = i;
            }
        }
        best_idx
    }

    /// Takes `&self`, not `&mut self` -- entries are updated via atomics,
    /// so many search threads can call this concurrently on the SAME
    /// shared table (the whole point of Lazy SMP: independent threads,
    /// one shared TT, no locks). The bucket scan in `replace_index` races
    /// benignly under that concurrency -- two threads can pick the same
    /// slot, or one can act on a staleness reading that changes a moment
    /// later -- the same tolerance every bucketed lock-free TT accepts,
    /// and each slot's own write is still atomic and torn-read-safe.
    #[inline]
    /// Guarda uma entrada que so' traz a avaliacao estatica, sem lance nem
    /// limite -- e que NUNCA despeja a entrada de outra posicao.
    ///
    /// A `store` normal escolhe a pior ranhura do balde e escreve la'. Para
    /// uma entrada de profundidade -8 isso e' um mau negocio: troca-se uma
    /// entrada com lance e profundidade real de OUTRA posicao por uma
    /// avaliacao. E acontece a cada avaliacao nova, que sao centenas de
    /// milhares por busca.
    ///
    /// MEDIDO antes disto existir: a tabela so' trazia lance em 22,3% dos nos
    /// interiores, e o lance dela -- quando existia -- produzia 18,5% de todos
    /// os cortes. Ou seja o lance era bom e faltava; nao era mau e sobrava.
    ///
    /// Aqui so' se escreve numa ranhura que ja' e' desta chave, ou numa que
    /// nao vale a pena guardar (vazia, ou ja' so'-avaliacao, ou de uma geracao
    /// antiga). Se as tres do balde estiverem ocupadas com trabalho real,
    /// deita-se fora a avaliacao em vez do lance -- recalcula-la custa uma
    /// passagem pela rede, recalcular a ordenacao custa uma subarvore.
    pub fn store_eval_only(&self, key: u64, static_eval: i16) {
        let idx = (key as usize) & self.mask;
        let bucket = &self.slots[idx];
        let gen = self.current_gen.load(Ordering::Relaxed);
        let mut alvo: Option<usize> = None;
        for (i, slot) in bucket.iter().enumerate() {
            let data = slot.data.load(Ordering::Relaxed);
            let key_xor = slot.key_xor_data.load(Ordering::Relaxed);
            if key_xor ^ data == key {
                alvo = Some(i);
                break;
            }
            if alvo.is_some() {
                continue;
            }
            let depth = ((data >> 34) & 0xFF) as u8 as i8 as i32;
            let stale = gen.wrapping_sub(slot.gen.load(Ordering::Relaxed)) as i32;
            if data == 0 || depth <= EVAL_ONLY_DEPTH || stale >= STALE_PENALTY {
                alvo = Some(i);
            }
        }
        let Some(i) = alvo else { return };
        let slot = &bucket[i];
        // Preserva o que a ranhura ja' tivesse desta mesma chave: se ela e'
        // desta posicao, o lance dela continua a valer.
        let anterior = {
            let data = slot.data.load(Ordering::Relaxed);
            let key_xor = slot.key_xor_data.load(Ordering::Relaxed);
            if key_xor ^ data == key { Some(data) } else { None }
        };
        let best = anterior.and_then(|d| decode_move(d & 0x3FFFF));
        let data = encode_data(EVAL_ONLY_DEPTH, 0, Bound::NoBound, best, false, static_eval);
        slot.data.store(data, Ordering::Relaxed);
        slot.key_xor_data.store(key ^ data, Ordering::Relaxed);
        slot.gen.store(gen, Ordering::Relaxed);
    }

    pub fn store(&self, key: u64, depth: i32, score: i32, bound: Bound, best: Option<Move>, pv: bool, static_eval: i16) {
        let data = encode_data(depth, score, bound, best, pv, static_eval);
        let idx = (key as usize) & self.mask;
        let bucket = &self.slots[idx];
        let gen = self.current_gen.load(Ordering::Relaxed);
        let slot = &bucket[Self::replace_index(bucket, key, gen)];
        slot.data.store(data, Ordering::Relaxed);
        slot.key_xor_data.store(key ^ data, Ordering::Relaxed);
        slot.gen.store(gen, Ordering::Relaxed);
    }

    pub fn clear(&self) {
        self.current_gen.store(0, Ordering::Relaxed);
        for bucket in &self.slots {
            for slot in bucket {
                slot.data.store(0, Ordering::Relaxed);
                slot.key_xor_data.store(0, Ordering::Relaxed);
                slot.gen.store(0, Ordering::Relaxed);
            }
        }
    }
}
