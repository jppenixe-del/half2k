use crate::bitboard::*;
use crate::types::*;

pub struct Attacks {
    pub knight: [Bitboard; 64],
    pub king: [Bitboard; 64],
    pub pawn: [[Bitboard; 64]; 2], // [color][square]
    /// between[a][b] = squares strictly between a and b when they share
    /// a rank/file/diagonal, else 0. Used for pinned-piece detection in
    /// generate_legal (a fast path that skips the make/unmake legality
    /// test for the common case; see movegen.rs). Built once at startup.
    pub between: [[Bitboard; 64]; 64],
    /// ray_extension[a][b] = squares strictly beyond b, continuing the ray
    /// a->b past b to the board edge, when a and b share a rank/file/
    /// diagonal, else 0. `& occ` finds the first piece beyond b on that
    /// ray in one table lookup instead of two magic-bitboard slider calls
    /// (compute attacks from b with b's occupant present, then without,
    /// and diff) -- used by the threats accumulator's discovered-attack
    /// check (the threat delta path), which does this once per
    /// slider found among a changed square's attackers, on every piece
    /// move the search makes.
    pub ray_extension: [[Bitboard; 64]; 64],
}

fn between_from(a: Square, b: Square) -> Bitboard {
    if a == b {
        return 0;
    }
    let (af, ar) = (file_of(a) as i32, rank_of(a) as i32);
    let (bf, br) = (file_of(b) as i32, rank_of(b) as i32);
    let df = (bf - af).signum();
    let dr = (br - ar).signum();
    // Must be a straight line (rank, file, or diagonal) to have a
    // "between" set at all.
    let aligned = af == bf || ar == br || (af - bf).abs() == (ar - br).abs();
    if !aligned {
        return 0;
    }
    let mut out = 0u64;
    let (mut f, mut r) = (af + df, ar + dr);
    while (f, r) != (bf, br) {
        if !(0..8).contains(&f) || !(0..8).contains(&r) {
            return 0; // safety: shouldn't happen for aligned squares
        }
        out |= bb(sq(f as u8, r as u8));
        f += df;
        r += dr;
    }
    out
}

fn ray_extension_from(a: Square, b: Square) -> Bitboard {
    if a == b {
        return 0;
    }
    let (af, ar) = (file_of(a) as i32, rank_of(a) as i32);
    let (bf, br) = (file_of(b) as i32, rank_of(b) as i32);
    let df = (bf - af).signum();
    let dr = (br - ar).signum();
    let aligned = af == bf || ar == br || (af - bf).abs() == (ar - br).abs();
    if !aligned {
        return 0;
    }
    let mut out = 0u64;
    let (mut f, mut r) = (bf + df, br + dr);
    while (0..8).contains(&f) && (0..8).contains(&r) {
        out |= bb(sq(f as u8, r as u8));
        f += df;
        r += dr;
    }
    out
}

fn knight_attacks_from(s: Square) -> Bitboard {
    let f = file_of(s) as i32;
    let r = rank_of(s) as i32;
    let deltas: [(i32, i32); 8] = [
        (1, 2), (2, 1), (2, -1), (1, -2),
        (-1, -2), (-2, -1), (-2, 1), (-1, 2),
    ];
    let mut out = 0u64;
    for (df, dr) in deltas {
        let nf = f + df;
        let nr = r + dr;
        if (0..8).contains(&nf) && (0..8).contains(&nr) {
            out |= bb(sq(nf as u8, nr as u8));
        }
    }
    out
}

fn king_attacks_from(s: Square) -> Bitboard {
    let f = file_of(s) as i32;
    let r = rank_of(s) as i32;
    let mut out = 0u64;
    for df in -1..=1 {
        for dr in -1..=1 {
            if df == 0 && dr == 0 {
                continue;
            }
            let nf = f + df;
            let nr = r + dr;
            if (0..8).contains(&nf) && (0..8).contains(&nr) {
                out |= bb(sq(nf as u8, nr as u8));
            }
        }
    }
    out
}

fn pawn_attacks_from(s: Square, color: Color) -> Bitboard {
    let f = file_of(s) as i32;
    let r = rank_of(s) as i32;
    let dr: i32 = if color == Color::White { 1 } else { -1 };
    let mut out = 0u64;
    for df in [-1, 1] {
        let nf = f + df;
        let nr = r + dr;
        if (0..8).contains(&nf) && (0..8).contains(&nr) {
            out |= bb(sq(nf as u8, nr as u8));
        }
    }
    out
}

impl Attacks {
    pub fn new() -> Self {
        let mut knight = [0u64; 64];
        let mut king = [0u64; 64];
        let mut pawn = [[0u64; 64]; 2];
        for s in 0..64u8 {
            knight[s as usize] = knight_attacks_from(s);
            king[s as usize] = king_attacks_from(s);
            pawn[Color::White.idx()][s as usize] = pawn_attacks_from(s, Color::White);
            pawn[Color::Black.idx()][s as usize] = pawn_attacks_from(s, Color::Black);
        }
        let mut between = [[0u64; 64]; 64];
        let mut ray_extension = [[0u64; 64]; 64];
        for a in 0..64u8 {
            for b in 0..64u8 {
                between[a as usize][b as usize] = between_from(a, b);
                ray_extension[a as usize][b as usize] = ray_extension_from(a, b);
            }
        }
        Attacks { knight, king, pawn, between, ray_extension }
    }
}

/// Sliding attacks via real magic bitboards (see magic.rs) -- was a naive
/// ray-cast + blocker scan until profiling showed it at ~17% of total
/// search time. Magic numbers are found by self-verifying random search
/// at startup (see magic.rs), not hardcoded/transcribed from anywhere.
static MAGICS: std::sync::OnceLock<crate::magic::Magics> = std::sync::OnceLock::new();
fn magics() -> &'static crate::magic::Magics {
    MAGICS.get_or_init(crate::magic::Magics::new)
}

#[inline]
pub fn bishop_attacks(s: Square, occ: Bitboard) -> Bitboard {
    magics().bishop_attacks(s, occ)
}

#[inline]
pub fn rook_attacks(s: Square, occ: Bitboard) -> Bitboard {
    magics().rook_attacks(s, occ)
}

#[inline]
pub fn queen_attacks(s: Square, occ: Bitboard) -> Bitboard {
    bishop_attacks(s, occ) | rook_attacks(s, occ)
}
