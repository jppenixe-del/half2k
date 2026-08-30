use crate::board::Board;
use crate::types::*;

pub struct Zobrist {
    pub piece_sq: [[[u64; 64]; 6]; 2], // [color][piece][square]
    pub side: u64,
    pub castling: [u64; 16],
    pub ep_file: [u64; 8],
}

// A simple PRNG (splitmix64), only to generate the keys once at startup.
// Determinism across runs does not matter here; good distribution does.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

impl Zobrist {
    pub fn new() -> Self {
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut piece_sq = [[[0u64; 64]; 6]; 2];
        for c in 0..2 {
            for p in 0..6 {
                for s in 0..64 {
                    piece_sq[c][p][s] = splitmix64(&mut state);
                }
            }
        }
        let side = splitmix64(&mut state);
        let mut castling = [0u64; 16];
        for c in castling.iter_mut() {
            *c = splitmix64(&mut state);
        }
        let mut ep_file = [0u64; 8];
        for e in ep_file.iter_mut() {
            *e = splitmix64(&mut state);
        }
        Zobrist { piece_sq, side, castling, ep_file }
    }

    /// The full recompute. `Board::hash` is now maintained incrementally and
    /// is what the search reads; this remains as the definition it is checked
    /// against (`verificahash`), and for building that value in the first
    /// place.
    ///
    /// History, because it is the interesting part. This was deliberately NOT
    /// incremental until 2026-08-06: it had been tried on 2026-07-21, verified
    /// bit-exact over 124M perft nodes, and measured a NET LOSS -- perft(6)
    /// went ~4.8s -> ~5.9s. The reason given was specific to this engine:
    /// `generate_legal` settled legality with a make/unmake per candidate
    /// move, so make/unmake ran far more often than `hash()` did, and paying
    /// XORs in every make to save one recompute per node loses at a ratio the
    /// comment estimated at ~35:1. The note ended by saying it should not be
    /// re-attempted "without also changing how legality checking works".
    ///
    /// That is what changed. King moves no longer need a make/unmake (see
    /// `Board::king_move_leaves_check`), and the ratio was then MEASURED
    /// rather than estimated: 2.43 make_move per node against 0.64 hash() per
    /// node -- **3.8:1**, not 35:1. At that ratio the arithmetic reverses, and
    /// it reversed in practice too.
    pub fn hash_completo(&self, board: &Board) -> u64 {
        self.hash(board)
    }

    /// Recompute from scratch. See `hash_completo` for why this is no longer
    /// what the search calls.
    pub fn hash(&self, board: &Board) -> u64 {
        let mut h = 0u64;
        for c in [Color::White, Color::Black] {
            for pt in ALL_PIECES {
                let mut bbp = board.pieces[c.idx()][pt.idx()];
                while bbp != 0 {
                    let s = bbp.trailing_zeros() as usize;
                    bbp &= bbp - 1;
                    h ^= self.piece_sq[c.idx()][pt.idx()][s];
                }
            }
        }
        if board.side == Color::Black {
            h ^= self.side;
        }
        h ^= self.castling[(board.castling & 0xF) as usize];
        if board.ep_square != NO_SQUARE {
            h ^= self.ep_file[file_of(board.ep_square) as usize];
        }
        h
    }
}

/// The keys, built once and shared.
///
/// Every `Zobrist::new()` produces the same tables -- `splitmix64` runs from a
/// fixed seed -- so a global is not a behaviour change, it is the same numbers
/// stopped being rebuilt per caller. `Board` needs them to keep its hash up to
/// date move by move and has no `Zobrist` of its own to reach for.
pub fn tabelas() -> &'static Zobrist {
    static Z: std::sync::OnceLock<Zobrist> = std::sync::OnceLock::new();
    Z.get_or_init(Zobrist::new)
}
