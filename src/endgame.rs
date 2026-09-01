//! Endgame knowledge: the positions where counting material is simply wrong.
//!
//! Why this exists. Measured before it did: two knights against a bare king
//! scored +4.20, which is not a small error but a category error -- that
//! position cannot be won at all, there is no mate to find. The same search
//! scored rook against king at +3.64, so the engine believed a dead draw was
//! worth MORE than a forced win. An evaluation that cannot tell those apart
//! will trade into the draw and decline the win, and it will do it
//! confidently. Endgame accuracy in a real lost game was 18%.
//!
//! The shape is two kinds of answer, because endgames need two kinds:
//!
//!   - `Exact`: this position's value is known, so replace the evaluation
//!     rather than adjust it. Used for theoretical draws (worth 0 whatever
//!     the material says) and for won endgames where the useful signal is not
//!     material at all but progress -- driving the defending king to the edge,
//!     to a corner, to the RIGHT corner.
//!   - `Scale`: the evaluation is right about who is better but wrong about
//!     whether it can be converted. A rook pawn with the defending king in
//!     front of it is a draw with any amount of material behind it.
//!
//! No tablebases involved. This is the knowledge that makes the difference
//! between knowing a position is won and knowing how to win it.

use crate::bitboard::{count, Bitboard, DARK_SQUARES, LIGHT_SQUARES};
use crate::board::Board;
use crate::types::{file_of, rank_of, Color, PieceType, Square};

/// Large enough to dominate any ordinary evaluation, small enough to stay
/// clear of mate scores so a known win never masquerades as a forced mate.
pub const KNOWN_WIN: i32 = 10_000;
pub const SCALE_NORMAL: i32 = 128;
pub const SCALE_DRAW: i32 = 0;

pub enum Verdict {
    /// Score from the strong side's point of view.
    Exact(i32),
    /// Multiplier over SCALE_NORMAL applied to the normal evaluation.
    Scale(i32),
}

fn n(board: &Board, c: Color, pt: PieceType) -> u32 {
    count(board.pieces[c.idx()][pt.idx()])
}

fn chebyshev(a: Square, b: Square) -> i32 {
    let dr = (rank_of(a) as i32 - rank_of(b) as i32).abs();
    let df = (file_of(a) as i32 - file_of(b) as i32).abs();
    dr.max(df)
}

/// Bigger the closer two squares are -- for pulling our king in.
fn push_close(a: Square, b: Square) -> i32 {
    7 - chebyshev(a, b)
}

/// Bigger the further apart -- for driving a defender away from its piece.
fn push_away(a: Square, b: Square) -> i32 {
    chebyshev(a, b)
}

/// Bigger the nearer the edge. A king with a whole board to run in is much
/// harder to mate than one pinned against the side.
fn push_to_edge(sq: Square) -> i32 {
    let rd = rank_of(sq).min(7 - rank_of(sq)) as i32;
    let fd = file_of(sq).min(7 - file_of(sq)) as i32;
    3 - rd.min(fd)
}

/// Bigger the nearer any corner.
fn push_to_corner(sq: Square) -> i32 {
    let rd = rank_of(sq).min(7 - rank_of(sq)) as i32;
    let fd = file_of(sq).min(7 - file_of(sq)) as i32;
    6 - rd - fd
}

/// Bigger the nearer the corner of ONE colour. Bishop and knight can only
/// force mate in a corner the bishop attacks, and a mate that is driven to
/// the wrong one is not a mate -- this is the whole difficulty of that
/// endgame, and without it the win is found and then not converted.
fn push_to_coloured_corner(sq: Square, dark: bool) -> i32 {
    let d = (7 - rank_of(sq) as i32 - file_of(sq) as i32).abs();
    7 - if dark { 7 - d } else { d }
}

fn is_dark(sq: Square) -> bool {
    (1u64 << sq) & DARK_SQUARES != 0
}

fn material_can_mate(board: &Board, strong: Color) -> bool {
    let q = n(board, strong, PieceType::Queen);
    let r = n(board, strong, PieceType::Rook);
    let b = n(board, strong, PieceType::Bishop);
    let kn = n(board, strong, PieceType::Knight);
    let bishops = board.pieces[strong.idx()][PieceType::Bishop.idx()];
    // Queen or rook mates alone; bishop plus knight mates; two bishops mate
    // only if they are on different colours. Two knights do NOT, and neither
    // does anything less -- which is exactly the case that was scoring +4.20.
    q > 0
        || r > 0
        || (b > 0 && kn > 0)
        || ((bishops & LIGHT_SQUARES) != 0 && (bishops & DARK_SQUARES) != 0)
}

/// Anything against a lone king. Material tells us who is better; what it
/// does not tell us is whether the position can be won at all, so the
/// known-win bonus is added only when the pieces can actually deliver mate.
fn eval_kx_vs_k(board: &Board, strong: Color, eg_material: i32) -> i32 {
    let our_king = board.king_sq(strong);
    let their_king = board.king_sq(strong.opp());
    let mut score =
        eg_material + 20 * push_close(our_king, their_king) + 20 * push_to_corner(their_king);
    if material_can_mate(board, strong) {
        score += KNOWN_WIN;
    }
    score
}

/// Bishop and knight against a lone king: won, but only into the corner the
/// bishop covers. The weights say so -- the coloured-corner term is worth
/// three times the plain edge term, because driving the king to the wrong
/// corner makes no progress at all.
fn eval_kbn_vs_k(board: &Board, strong: Color) -> i32 {
    let bishop = board.pieces[strong.idx()][PieceType::Bishop.idx()].trailing_zeros() as Square;
    let our_king = board.king_sq(strong);
    let their_king = board.king_sq(strong.opp());
    KNOWN_WIN
        + 20 * push_close(our_king, their_king)
        + 70 * push_to_edge(their_king)
        + 200 * push_to_coloured_corner(their_king, is_dark(bishop))
}

/// Rook against knight: NOT a win, and deliberately given no material bonus.
/// The defending side holds by keeping knight and king together; the only
/// real progress is separating them and pushing the king to the edge.
fn eval_kr_vs_kn(board: &Board, strong: Color) -> i32 {
    let knight = board.pieces[strong.opp().idx()][PieceType::Knight.idx()].trailing_zeros() as Square;
    let their_king = board.king_sq(strong.opp());
    10 * push_to_edge(their_king) + 15 * push_away(their_king, knight)
}

/// Queen against a pawn. Usually winning, with a real exception: a rook or
/// bishop pawn one step from queening, where the defending king can be
/// stalemated and the queen cannot approach. The material credit is withheld
/// in exactly that case.
fn eval_kq_vs_kp(board: &Board, strong: Color, q_minus_p: i32) -> i32 {
    let weak = strong.opp();
    let queen = board.pieces[strong.idx()][PieceType::Queen.idx()].trailing_zeros() as Square;
    let pawn = board.pieces[weak.idx()][PieceType::Pawn.idx()].trailing_zeros() as Square;
    let our_king = board.king_sq(strong);
    let mut score = 20 * push_close(our_king, pawn);

    // The squares the pawn still has to cross, including where it promotes.
    let file = file_of(pawn);
    let mut path: Bitboard = 0;
    if weak == Color::White {
        for r in (rank_of(pawn) + 1)..=7 {
            path |= 1u64 << (r * 8 + file);
        }
    } else {
        for r in 0..rank_of(pawn) {
            path |= 1u64 << (r * 8 + file);
        }
    }
    let blocked = (path & (1u64 << queen)) != 0 || (path & (1u64 << our_king)) != 0;
    if blocked {
        score += KNOWN_WIN;
    }
    // Relative rank of the pawn from the defender's side: 6 means one step away.
    let rel_rank = if weak == Color::White { rank_of(pawn) } else { 7 - rank_of(pawn) };
    let central_or_knight_file = matches!(file, 1 | 3 | 4 | 6);
    if rel_rank < 6 || central_or_knight_file || blocked {
        score += q_minus_p;
    }
    score
}

/// Queen against rook: won, and the technique is the same as any bare-king
/// mate -- bring the king up and squeeze the defender towards a corner.
fn eval_kq_vs_kr(board: &Board, strong: Color, eg_material: i32) -> i32 {
    let our_king = board.king_sq(strong);
    let their_king = board.king_sq(strong.opp());
    eg_material + 20 * push_close(our_king, their_king) + 20 * push_to_corner(their_king)
}

/// Pawns against a lone king, all on one rook file: drawn if the defending
/// king reaches the promotion square. Material says a clear win; the board
/// says the pawn cannot get past.
fn scale_kps_vs_k(board: &Board, strong: Color) -> i32 {
    let pawns = board.pieces[strong.idx()][PieceType::Pawn.idx()];
    if pawns & !crate::bitboard::FILE_A != 0 && pawns & !file_bb(7) != 0 {
        return SCALE_NORMAL;
    }
    let queening_rank: u8 = if strong == Color::White { 7 } else { 0 };
    let f = file_of(pawns.trailing_zeros() as Square);
    let queening_sq = (queening_rank * 8 + f) as Square;
    if chebyshev(board.king_sq(strong.opp()), queening_sq) <= 1 {
        return SCALE_DRAW;
    }
    SCALE_NORMAL
}

/// Bishop and rook-file pawns against a lone king: drawn if the bishop does
/// not cover the promotion square and the defending king can sit on it. The
/// most famous "extra piece, still a draw" in the endgame, and invisible to
/// any amount of material counting.
fn scale_kbps_vs_k(board: &Board, strong: Color) -> i32 {
    let pawns = board.pieces[strong.idx()][PieceType::Pawn.idx()];
    if pawns & !crate::bitboard::FILE_A != 0 && pawns & !file_bb(7) != 0 {
        return SCALE_NORMAL;
    }
    let bishop = board.pieces[strong.idx()][PieceType::Bishop.idx()].trailing_zeros() as Square;
    let queening_rank: u8 = if strong == Color::White { 7 } else { 0 };
    let f = file_of(pawns.trailing_zeros() as Square);
    let queening_sq = (queening_rank * 8 + f) as Square;
    if is_dark(queening_sq) != is_dark(bishop)
        && chebyshev(board.king_sq(strong.opp()), queening_sq) <= 1
    {
        return SCALE_DRAW;
    }
    SCALE_NORMAL
}

fn file_bb(f: u8) -> Bitboard {
    crate::bitboard::FILE_A << f
}

/// Push the losing king to the edge and bring ours up, whenever the material
/// is decisive -- not only when the loser is stripped bare.
///
/// The knowledge below only fires when the weak side has NOTHING: no pawn, no
/// piece. Leave it a bishop and two pawns and the engine plays a won endgame
/// with no idea that the point is to trap the king. Measured over four real
/// bullet wins: in balanced positions it did not drop 150cp once in 101 moves,
/// and in positions already won by more than fifteen pawns it dropped 150cp in
/// FOUR MOVES OUT OF FIVE, average 259. It converts, but it strolls -- and
/// those games ran to 227, 185 and 147 moves on a sixty-second clock, which is
/// how a won game becomes a race.
///
/// Deliberately small. This is a tie-break among moves that are all winning,
/// not a claim about who stands better: at 6 and 4 per unit of distance the
/// whole term spans about a fifth of a pawn, so it can reorder equal moves and
/// cannot outvote a real one. Returns zero unless the advantage is past a
/// rook and a minor, where "which move wins" has stopped being the question
/// and "which move wins soonest" has started.
pub fn conversion_drive(board: &Board, eg_material_white: i32) -> i32 {
    const DECISIVE: i32 = 800;
    let (strong, adv) = if eg_material_white >= 0 {
        (Color::White, eg_material_white)
    } else {
        (Color::Black, -eg_material_white)
    };
    if adv < DECISIVE {
        return 0;
    }
    // A king with a queen still on the board is not being mated in a corner;
    // it is being checkmated wherever it stands. The drive is for the endings
    // where the loser runs.
    let weak = strong.opp();
    if n(board, weak, PieceType::Queen) > 0 {
        return 0;
    }
    // 2026-07-30: was 6 and 4, which spans about a fifth of a pawn across the
    // whole term. Measured over 1281 games played to the finish: median moves
    // to win 79 against the champion's 78, and 598 wins against 603. It did
    // nothing at all -- in a position won by eight pawns the candidate moves
    // differ by more than the tie-break was worth, so it was swallowed before
    // it could break anything.
    //
    // Four times bigger, still under a pawn across the term, which is what a
    // tie-break can be in a position nobody is going to lose.
    let v = 24 * push_close(board.king_sq(strong), board.king_sq(weak))
        + 16 * push_to_edge(board.king_sq(weak));
    if strong == Color::White { v } else { -v }
}

/// Which side, if either, is the one with the material -- and what is known
/// about the resulting endgame.
///
/// Matching on piece counts rather than a hashed material key: with this many
/// cases it is the same work, it cannot collide, and each arm reads as the
/// endgame it describes.
pub fn probe(board: &Board, eg_material_white: i32, q_minus_p: i32) -> Option<(Color, Verdict)> {
    let c = |col: Color, pt: PieceType| n(board, col, pt);
    for &strong in &[Color::White, Color::Black] {
        let weak = strong.opp();
        let (sp, sn, sb, sr, sq) = (
            c(strong, PieceType::Pawn),
            c(strong, PieceType::Knight),
            c(strong, PieceType::Bishop),
            c(strong, PieceType::Rook),
            c(strong, PieceType::Queen),
        );
        let (wp, wn, wb, wr, wq) = (
            c(weak, PieceType::Pawn),
            c(weak, PieceType::Knight),
            c(weak, PieceType::Bishop),
            c(weak, PieceType::Rook),
            c(weak, PieceType::Queen),
        );
        let eg_mat = if strong == Color::White { eg_material_white } else { -eg_material_white };
        let weak_bare = wp + wn + wb + wr + wq == 0;

        // Theoretical draws: no mate exists, so no amount of material matters.
        if weak_bare && sp == 0 && sr == 0 && sq == 0 && (sn + sb) <= 2 {
            let bishops = board.pieces[strong.idx()][PieceType::Bishop.idx()];
            let two_colours = (bishops & LIGHT_SQUARES) != 0 && (bishops & DARK_SQUARES) != 0;
            if sb == 1 && sn == 1 {
                return Some((strong, Verdict::Exact(eval_kbn_vs_k(board, strong))));
            }
            if !two_colours {
                // K, KN, KB, KNN against a bare king: all drawn.
                return Some((strong, Verdict::Exact(0)));
            }
        }
        // Minor against minor with no pawns: nobody can force anything.
        if sp == 0 && wp == 0 && sr == 0 && wr == 0 && sq == 0 && wq == 0
            && (sn + sb) <= 1 && (wn + wb) <= 1 && (sn + sb + wn + wb) > 0
        {
            return Some((strong, Verdict::Exact(0)));
        }
        if weak_bare {
            if sq == 1 && sr == 0 && sb == 0 && sn == 0 && sp == 0 {
                return Some((strong, Verdict::Exact(eval_kx_vs_k(board, strong, eg_mat))));
            }
            if sp > 0 && sn == 0 && sb == 0 && sr == 0 && sq == 0 {
                return Some((strong, Verdict::Scale(scale_kps_vs_k(board, strong))));
            }
            if sp > 0 && sb == 1 && sn == 0 && sr == 0 && sq == 0 {
                return Some((strong, Verdict::Scale(scale_kbps_vs_k(board, strong))));
            }
            if sp == 0 {
                return Some((strong, Verdict::Exact(eval_kx_vs_k(board, strong, eg_mat))));
            }
        }
        // Rook against knight, no pawns: held, not won.
        if sr == 1 && sq == 0 && sb == 0 && sn == 0 && sp == 0
            && wn == 1 && wq == 0 && wr == 0 && wb == 0 && wp == 0
        {
            return Some((strong, Verdict::Exact(eval_kr_vs_kn(board, strong))));
        }
        // Queen against a pawn.
        if sq == 1 && sr == 0 && sb == 0 && sn == 0 && sp == 0
            && wp == 1 && wn == 0 && wb == 0 && wr == 0 && wq == 0
        {
            return Some((strong, Verdict::Exact(eval_kq_vs_kp(board, strong, q_minus_p))));
        }
        // Queen against rook.
        if sq == 1 && sr == 0 && sb == 0 && sn == 0 && sp == 0
            && wr == 1 && wn == 0 && wb == 0 && wq == 0 && wp == 0
        {
            return Some((strong, Verdict::Exact(eval_kq_vs_kr(board, strong, eg_mat))));
        }
    }
    None
}
