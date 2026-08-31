//! Move generation with a fixed, deliberate order.
//!
//! Two generators that agree on every legal move can still disagree on the
//! order they hand them over, and among moves that score the same the order is
//! what decides which one is tried fifth. That decides what late move pruning
//! removes, which decides the shape of the tree, which decides everything else.
//!
//! What that costs when it is left to chance, measured on this engine: at depth
//! two, two searches with identical rules and identical constants reached the
//! reverse futility test at exactly the same nineteen nodes and fired six times
//! against eleven, and one never pruned a move by count where the other pruned
//! sixteen. Nothing was wrong with either. They simply had a different fifth
//! move.
//!
//! So the order here is stated rather than inherited: pawn pushes, then double
//! pushes, then captures to the left, then to the right, then en passant, then
//! promotions; knights, bishops, rooks, queens; then the king, then castling.
//! Pieces are walked from the lowest square upward and their targets likewise.
//! Promotions come out queen, knight, rook, bishop.
//!
//! Removing a pinned move overwrites its slot with the LAST move rather than
//! shifting the tail down. That is not tidiness -- it reorders what remains,
//! and the order is the point of the file.

use crate::attacks::{bishop_attacks, rook_attacks, Attacks};
use crate::bitboard::*;
use crate::board::{Board, CASTLE_BK, CASTLE_BQ, CASTLE_WK, CASTLE_WQ};
use crate::moves::{Move, MoveFlag};
use crate::types::*;

pub const GEN_LEGAL: u8 = 0;
pub const GEN_CAPTURES: u8 = 1;
pub const GEN_QUIETS: u8 = 2;

const RANK_1: Bitboard = 0x0000_0000_0000_00ff;
const RANK_2: Bitboard = RANK_1 << 8;
const RANK_3: Bitboard = RANK_1 << 16;
const RANK_6: Bitboard = RANK_1 << 40;
const RANK_7: Bitboard = RANK_1 << 48;
const FILE_A: Bitboard = 0x0101_0101_0101_0101;
const FILE_H: Bitboard = FILE_A << 7;

#[inline]
fn pop_lsb(bb: &mut Bitboard) -> Square {
    let s = bb.trailing_zeros() as Square;
    *bb &= *bb - 1;
    s
}

#[inline]
fn more_than_one(bb: Bitboard) -> bool {
    bb & bb.wrapping_sub(1) != 0
}

/// Everything of `by`'s that attacks `square`, given an occupancy that may not
/// be the board's.
pub fn attackers(board: &Board, atk: &Attacks, square: Square, occ: Bitboard, by: Color) -> Bitboard {
    let p = &board.pieces[by.idx()];
    // The pawn table is indexed by the side being attacked, so it is the other
    // colour's table that finds attacking pawns.
    (atk.pawn[by.opp().idx()][square as usize] & p[PieceType::Pawn.idx()])
        | (atk.knight[square as usize] & p[PieceType::Knight.idx()])
        | (bishop_attacks(square, occ) & (p[PieceType::Bishop.idx()] | p[PieceType::Queen.idx()]))
        | (rook_attacks(square, occ) & (p[PieceType::Rook.idx()] | p[PieceType::Queen.idx()]))
        | (atk.king[square as usize] & p[PieceType::King.idx()])
}

/// Pieces of ours pinned against `square`, and the pieces doing the pinning.
fn pins(board: &Board, atk: &Attacks, square: Square, occ: Bitboard, us: Color) -> (Bitboard, Bitboard) {
    let them = us.opp();
    let bishops =
        board.pieces[them.idx()][PieceType::Bishop.idx()] | board.pieces[them.idx()][PieceType::Queen.idx()];
    let rooks =
        board.pieces[them.idx()][PieceType::Rook.idx()] | board.pieces[them.idx()][PieceType::Queen.idx()];

    // Anything that would attack the square on an empty board is a candidate.
    let mut candidates =
        (bishop_attacks(square, 0) & bishops) | (rook_attacks(square, 0) & rooks);
    let occ_excl = occ ^ candidates;

    let mut pinned = 0u64;
    let mut pinners = 0u64;
    while candidates != 0 {
        let pinner = pop_lsb(&mut candidates);
        let between = atk.between[square as usize][pinner as usize] & occ_excl;
        if between != 0 && !more_than_one(between) {
            pinned |= between;
            pinners |= bb(pinner);
        }
    }
    (pinned, pinners)
}

/// May a pinned piece go there: only along the pin, or onto the pinner.
fn pinned_move_ok(atk: &Attacks, from: Square, to: Square, mut pinners: Bitboard, king: Square) -> bool {
    while pinners != 0 {
        let pinner = pop_lsb(&mut pinners);
        let legals = atk.between[pinner as usize][king as usize] | bb(pinner);
        if legals & bb(from) != 0 && legals & bb(to) != 0 {
            return true;
        }
    }
    false
}

fn can_castle(board: &Board, atk: &Attacks, us: Color, kingside: bool, occ: Bitboard) -> bool {
    let king_sq = board.king_sq(us);
    let (rook_sq, king_target) = match (us, kingside) {
        (Color::White, true) => (7u8, 6u8),
        (Color::White, false) => (0u8, 2u8),
        (Color::Black, true) => (63u8, 62u8),
        (Color::Black, false) => (56u8, 58u8),
    };
    let rook_target = if kingside { king_target - 1 } else { king_target + 1 };

    let occ = occ & !bb(king_sq) & !bb(rook_sq);
    let king_travel = atk.between[king_sq as usize][king_target as usize] | bb(king_target);
    let rook_travel = atk.between[rook_sq as usize][rook_target as usize] | bb(rook_target);
    if occ & king_travel != 0 || occ & rook_travel != 0 {
        return false;
    }
    // Every square the king passes through, including the one it lands on.
    let mut travel = king_travel;
    while travel != 0 {
        let s = pop_lsb(&mut travel);
        if attackers(board, atk, s, occ, us.opp()) != 0 {
            return false;
        }
    }
    true
}

/// Everything of theirs that currently attacks our king.
pub fn checkers(board: &Board, atk: &Attacks, us: Color) -> Bitboard {
    attackers(board, atk, board.king_sq(us), board.occ_all, us.opp())
}

pub fn generate(board: &Board, atk: &Attacks, kind: u8) -> Vec<Move> {
    let us = board.side;
    let them = us.opp();
    let occ = board.occ_all;
    let king_sq = board.king_sq(us);
    let (pinned, pinners) = pins(board, atk, king_sq, occ, us);
    let check = checkers(board, atk, us);

    let filter = match kind {
        GEN_CAPTURES => board.occ_color[them.idx()],
        GEN_QUIETS => !occ,
        _ => !board.occ_color[us.idx()],
    };

    let mut list: Vec<Move> = Vec::with_capacity(64);

    if !more_than_one(check) {
        let mut f = filter;
        if check != 0 {
            f &= check | atk.between[king_sq as usize][check.trailing_zeros() as usize];
        }
        gen_pawns(board, atk, us, f, occ, &mut list);
        gen_piece(board, atk, us, PieceType::Knight, f, occ, &mut list);
        gen_piece(board, atk, us, PieceType::Bishop, f, occ, &mut list);
        gen_piece(board, atk, us, PieceType::Rook, f, occ, &mut list);
        gen_piece(board, atk, us, PieceType::Queen, f, occ, &mut list);
    }

    gen_king(board, atk, us, filter, occ, check, kind, &mut list);

    // Pinned pieces last, and removed by overwriting with the tail.
    if pinned != 0 {
        let mut i = 0;
        while i < list.len() {
            let m = list[i];
            if bb(m.from) & pinned != 0 && !pinned_move_ok(atk, m.from, m.to, pinners, king_sq) {
                let last = list.pop().unwrap();
                if i < list.len() {
                    list[i] = last;
                }
            } else {
                i += 1;
            }
        }
    }

    list
}

fn gen_piece(
    board: &Board,
    atk: &Attacks,
    us: Color,
    pt: PieceType,
    filter: Bitboard,
    occ: Bitboard,
    list: &mut Vec<Move>,
) {
    let mut pieces = board.pieces[us.idx()][pt.idx()];
    while pieces != 0 {
        let from = pop_lsb(&mut pieces);
        let mut targets = match pt {
            PieceType::Knight => atk.knight[from as usize],
            PieceType::Bishop => bishop_attacks(from, occ),
            PieceType::Rook => rook_attacks(from, occ),
            PieceType::Queen => bishop_attacks(from, occ) | rook_attacks(from, occ),
            _ => 0,
        } & filter;
        while targets != 0 {
            let to = pop_lsb(&mut targets);
            list.push(Move {
                from,
                to,
                promotion: None,
                flag: if occ & bb(to) != 0 { MoveFlag::Capture } else { MoveFlag::Quiet },
            });
        }
    }
}

fn promo(from: Square, to: Square, capture: bool, list: &mut Vec<Move>) {
    // Queen, knight, rook, bishop, in that order.
    for p in [PieceType::Queen, PieceType::Knight, PieceType::Rook, PieceType::Bishop] {
        list.push(Move {
            from,
            to,
            promotion: Some(p),
            flag: if capture { MoveFlag::Capture } else { MoveFlag::Quiet },
        });
    }
}

fn gen_pawns(
    board: &Board,
    atk: &Attacks,
    us: Color,
    filter: Bitboard,
    occ: Bitboard,
    list: &mut Vec<Move>,
) {
    let white = us == Color::White;
    let them = us.opp();
    let rank3 = if white { RANK_3 } else { RANK_6 };
    let rank7 = if white { RANK_7 } else { RANK_2 };

    let up = |b: Bitboard| if white { b << 8 } else { b >> 8 };
    let up_left = |b: Bitboard| if white { b << 7 } else { b >> 9 };
    let up_right = |b: Bitboard| if white { b << 9 } else { b >> 7 };
    let back = |s: Square, d: i32| -> Square { (s as i32 - d) as Square };
    let d_up: i32 = if white { 8 } else { -8 };
    let d_left: i32 = d_up - 1;
    let d_right: i32 = d_up + 1;

    let enemy = board.occ_color[them.idx()] & filter;
    let empty = !occ;
    let pawns = board.pieces[us.idx()][PieceType::Pawn.idx()];
    let promoting = pawns & rank7;
    let plain = pawns & !rank7;

    let mut single = up(plain) & empty;
    let mut double = up(single & rank3) & empty & filter;
    single &= filter;

    while single != 0 {
        let to = pop_lsb(&mut single);
        list.push(Move::quiet(back(to, d_up), to));
    }
    while double != 0 {
        let to = pop_lsb(&mut double);
        list.push(Move {
            from: back(to, 2 * d_up),
            to,
            promotion: None,
            flag: MoveFlag::DoublePush,
        });
    }

    let mut left = up_left(plain & !FILE_A) & enemy;
    let mut right = up_right(plain & !FILE_H) & enemy;
    while left != 0 {
        let to = pop_lsb(&mut left);
        list.push(Move::capture(back(to, d_left), to));
    }
    while right != 0 {
        let to = pop_lsb(&mut right);
        list.push(Move::capture(back(to, d_right), to));
    }

    // En passant needs its own pin test, done here rather than left to the
    // general one: the capture takes two pawns off a single rank at once, and
    // ordinary pin detection looks at one piece leaving one square.
    if board.ep_square != NO_SQUARE && filter & bb(back(board.ep_square, d_up)) != 0 {
        let king_sq = board.king_sq(us);
        let mut ep_attackers = atk.pawn[them.idx()][board.ep_square as usize] & plain;
        while ep_attackers != 0 {
            let from = pop_lsb(&mut ep_attackers);
            if rank_of(from) == rank_of(king_sq) {
                let mut rooks_queens = (board.pieces[them.idx()][PieceType::Rook.idx()]
                    | board.pieces[them.idx()][PieceType::Queen.idx()])
                    & (RANK_1 << (8 * rank_of(from)));
                if rooks_queens != 0 {
                    let new_occ = occ & !bb(from) & !bb(back(board.ep_square, d_up));
                    let mut in_check = false;
                    while rooks_queens != 0 && !in_check {
                        let r = pop_lsb(&mut rooks_queens);
                        in_check = rook_attacks(r, new_occ) & bb(king_sq) != 0;
                    }
                    if in_check {
                        continue;
                    }
                }
            }
            list.push(Move {
                from,
                to: board.ep_square,
                promotion: None,
                flag: MoveFlag::EnPassant,
            });
        }
    }

    let mut fwd = up(promoting) & empty & filter;
    let mut lcap = up_left(promoting & !FILE_A) & enemy & filter;
    let mut rcap = up_right(promoting & !FILE_H) & enemy & filter;
    while fwd != 0 {
        let to = pop_lsb(&mut fwd);
        promo(back(to, d_up), to, false, list);
    }
    while lcap != 0 {
        let to = pop_lsb(&mut lcap);
        promo(back(to, d_left), to, true, list);
    }
    while rcap != 0 {
        let to = pop_lsb(&mut rcap);
        promo(back(to, d_right), to, true, list);
    }
}

fn gen_king(
    board: &Board,
    atk: &Attacks,
    us: Color,
    filter: Bitboard,
    occ: Bitboard,
    check: Bitboard,
    kind: u8,
    list: &mut Vec<Move>,
) {
    let king_sq = board.king_sq(us);
    let mut targets = atk.king[king_sq as usize] & filter;
    // The king is lifted off the board for the attack test, or a slider through
    // its own square looks blocked by it.
    let occ_noking = occ ^ bb(king_sq);
    while targets != 0 {
        let to = pop_lsb(&mut targets);
        if attackers(board, atk, to, occ_noking, us.opp()) == 0 {
            list.push(Move {
                from: king_sq,
                to,
                promotion: None,
                flag: if occ & bb(to) != 0 { MoveFlag::Capture } else { MoveFlag::Quiet },
            });
        }
    }

    if check == 0 && kind != GEN_CAPTURES {
        let (ks, qs) = match us {
            Color::White => (CASTLE_WK, CASTLE_WQ),
            Color::Black => (CASTLE_BK, CASTLE_BQ),
        };
        if board.castling & ks != 0 && can_castle(board, atk, us, true, occ) {
            let to = if us == Color::White { 6 } else { 62 };
            list.push(Move { from: king_sq, to, promotion: None, flag: MoveFlag::CastleKing });
        }
        if board.castling & qs != 0 && can_castle(board, atk, us, false, occ) {
            let to = if us == Color::White { 2 } else { 58 };
            list.push(Move { from: king_sq, to, promotion: None, flag: MoveFlag::CastleQueen });
        }
    }
}
