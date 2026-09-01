//! Syzygy endgame tablebases.
//!
//! Below a certain number of pieces the answer is not an estimate: the position
//! is won, drawn or lost, and it is known. A search that keeps guessing there is
//! spending nodes to arrive at something already written down, and worse, it can
//! guess wrong -- the endings where a network is least reliable are exactly the
//! ones the tables cover.
//!
//! Probed at interior nodes only when the fifty-move counter is zero. A table
//! says who wins with best play and knows nothing about the fifty move rule, so
//! a position with a counter running is a position the table cannot speak for.
//!
//! The engine keeps its own board. The conversion into the shape the prober
//! wants happens here and nowhere else, so nothing about the rest of the search
//! has to know this module exists.

use std::sync::OnceLock;

use shakmaty::{Bitboard as ShBitboard, Board as ShBoard, CastlingMode, Chess,
               Color as ShColor, FromSetup, Role, Setup, Square as ShSquare};
use shakmaty_syzygy::{AmbiguousWdl, Tablebase};

use crate::types::{Color, PieceType, NO_SQUARE};

static TABELAS: OnceLock<Option<Tablebase<Chess>>> = OnceLock::new();
static TEM_DTZ: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Whether the loaded set can say how long a win takes, not just that it
/// is one. Without it a table knows the position is won and nothing more,
/// and every winning move looks identical -- which is how an engine that
/// is a rook up shuffles until the fifty move rule takes the point away.
pub fn tem_dtz() -> bool {
    TEM_DTZ.load(std::sync::atomic::Ordering::Relaxed)
}

/// What the tables say, from the point of view of the side to move.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Wdl {
    Perde,
    Empata,
    Ganha,
}

/// Load the tables once. A path that holds nothing usable is not an error worth
/// stopping for -- the engine plays without them, as it always has.
pub fn carregar(caminho: &str) -> usize {
    let mut tb = Tablebase::new();
    let mut n = 0usize;
    for parte in caminho.split(|c| c == ';' || c == ':') {
        let parte = parte.trim();
        if parte.is_empty() {
            continue;
        }
        if tb.add_directory(parte).is_ok() {
            n += 1;
        }
    }
    let max = tb.max_pieces();
    // A directory of WDL files alone is a common thing to have, and it is worth
    // knowing which we got: the two answer different questions.
    let mut dtz = false;
    for parte in caminho.split(|c| c == ';' || c == ':') {
        let parte = parte.trim();
        if parte.is_empty() {
            continue;
        }
        if let Ok(dir) = std::fs::read_dir(parte) {
            for e in dir.flatten() {
                if e.file_name().to_string_lossy().ends_with(".rtbz") {
                    dtz = true;
                    break;
                }
            }
        }
    }
    TEM_DTZ.store(dtz, std::sync::atomic::Ordering::Relaxed);
    let _ = TABELAS.set(if n > 0 { Some(tb) } else { None });
    if n > 0 {
        max
    } else {
        0
    }
}

/// How many pieces the loaded tables reach.
pub fn max_pecas() -> usize {
    match TABELAS.get() {
        Some(Some(tb)) => tb.max_pieces(),
        _ => 0,
    }
}

fn papel(pt: PieceType) -> Role {
    match pt {
        PieceType::Pawn => Role::Pawn,
        PieceType::Knight => Role::Knight,
        PieceType::Bishop => Role::Bishop,
        PieceType::Rook => Role::Rook,
        PieceType::Queen => Role::Queen,
        PieceType::King => Role::King,
    }
}

/// Our board, in the shape the prober reads.
///
/// Castling rights are dropped deliberately: a position with pieces few enough
/// to be in a table has no castling left, and carrying the flags across only
/// creates a way for the two representations to disagree.
fn converter(board: &crate::board::Board) -> Option<Chess> {
    let mut tabuleiro = ShBoard::empty();
    for cor in [Color::White, Color::Black] {
        let sh = if cor == Color::White { ShColor::White } else { ShColor::Black };
        for pt in crate::types::ALL_PIECES {
            let mut bb = board.pieces[cor.idx()][pt.idx()];
            while bb != 0 {
                let sq = bb.trailing_zeros() as u32;
                bb &= bb - 1;
                tabuleiro.set_piece_at(ShSquare::new(sq), papel(pt).of(sh));
            }
        }
    }
    let setup = Setup {
        board: tabuleiro,
        promoted: ShBitboard::EMPTY,
        pockets: None,
        turn: if board.side == Color::White { ShColor::White } else { ShColor::Black },
        castling_rights: ShBitboard::EMPTY,
        ep_square: if board.ep_square == NO_SQUARE {
            None
        } else {
            ShSquare::new(board.ep_square as u32).into()
        },
        remaining_checks: None,
        halfmoves: board.halfmove,
        fullmoves: core::num::NonZeroU32::new(board.fullmove.max(1)).unwrap(),
    };
    Chess::from_setup(setup, CastlingMode::Standard).ok()
}

/// Win, draw or loss for the side to move, if the tables cover this position.
pub fn sondar(board: &crate::board::Board) -> Option<Wdl> {
    let tb = match TABELAS.get() {
        Some(Some(t)) => t,
        _ => return None,
    };
    // A table answers about best play from here, with no fifty move counter in
    // sight. With a counter running the two questions are different ones.
    if board.halfmove != 0 {
        return None;
    }
    if board.occ_all.count_ones() as usize > tb.max_pieces() {
        return None;
    }
    let pos = converter(board)?;
    match tb.probe_wdl(&pos).ok()? {
        AmbiguousWdl::Loss => Some(Wdl::Perde),
        AmbiguousWdl::BlessedLoss | AmbiguousWdl::Draw | AmbiguousWdl::CursedWin => {
            Some(Wdl::Empata)
        }
        AmbiguousWdl::Win => Some(Wdl::Ganha),
        // A set with no distance-to-zero files cannot tell a clean win from one
        // the fifty move rule would spoil, so it says "maybe". It is still a
        // win: the doubt is about the counter, not about the result.
        //
        // These were being swallowed by a catch-all that called everything it
        // did not recognise a draw, which is the worst possible reading -- a
        // won ending seen as drawn leaves the search nothing to steer by. King
        // and rook against king went from a1a6, which drives the king to the
        // edge, to e1e2, which does nothing. The compiler found it by refusing
        // the match once the catch-all was gone.
        AmbiguousWdl::MaybeWin => Some(Wdl::Ganha),
        AmbiguousWdl::MaybeLoss => Some(Wdl::Perde),
    }
}

/// The move to play, when the position is one the tables have settled.
///
/// There is nothing to search for here. The table knows who wins and how
/// quickly, and a search can only arrive at the same answer more slowly or at a
/// worse one -- the distance-to-zero move is the one that makes progress
/// against the fifty move rule, which is exactly what a search optimising a
/// score will not do. It returns the move rather than a hint, and the caller
/// plays it.
///
/// Matched back through our own legal moves rather than built from the
/// prober's: the two representations agree on squares and promotion pieces,
/// and everything else about a move stays ours.
pub fn melhor_jogada_raiz(
    board: &mut crate::board::Board,
    atk: &crate::attacks::Attacks,
) -> Option<(crate::moves::Move, Wdl)> {
    let tb = match TABELAS.get() {
        Some(Some(t)) => t,
        _ => return None,
    };
    if board.occ_all.count_ones() as usize > tb.max_pieces() {
        return None;
    }
    // Without distance-to-zero there is no "best" move to give: every move that
    // holds the win scores the same and the one returned makes no progress.
    if !tem_dtz() {
        return None;
    }
    let pos = converter(board)?;
    let (jogada, _dtz) = tb.best_move(&pos).ok()??;
    let de = jogada.from()? as u8;
    let para = jogada.to() as u8;
    let promo = jogada.promotion().map(|r| match r {
        Role::Queen => PieceType::Queen,
        Role::Rook => PieceType::Rook,
        Role::Bishop => PieceType::Bishop,
        _ => PieceType::Knight,
    });

    let legais = crate::movegen::generate_legal(board, atk);
    let escolhida = legais
        .iter()
        .find(|m| m.from == de && m.to == para && m.promotion == promo)
        .copied()?;

    // What the position is worth, so the caller can report a score with it.
    let wdl = sondar(board).unwrap_or(Wdl::Empata);
    Some((escolhida, wdl))
}
