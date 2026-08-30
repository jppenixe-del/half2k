use crate::attacks::*;
use crate::bitboard::*;
use crate::board::*;
use crate::moves::*;
use crate::types::*;

pub fn generate_pseudo_legal(board: &Board, atk: &Attacks, out: &mut Vec<Move>) {
    gera_pseudo::<false>(board, atk, out)
}

/// Captures, en passant and queen promotions only.
///
/// Exists for quiescence, which searches nothing else: it used to generate
/// every move and then throw the quiet ones away, so roughly thirty moves were
/// built per node to keep about five, at the ~30% of nodes that are quiescence.
/// The generation is the same code with one bitboard changed -- targets are
/// intersected with the enemy pieces instead of with everything that is not
/// ours -- so the two cannot drift apart.
///
/// Quiet queen promotions are included even though they capture nothing: a
/// pawn stepping to the last rank changes material as much as a capture does,
/// and quiescence has always considered them.
pub fn generate_pseudo_legal_caps(board: &Board, atk: &Attacks, out: &mut Vec<Move>) {
    gera_pseudo::<true>(board, atk, out)
}

fn gera_pseudo<const APENAS_CAPTURAS: bool>(board: &Board, atk: &Attacks, out: &mut Vec<Move>) {
    let us = board.side;
    let them = us.opp();
    let own = board.occ_color[us.idx()];
    let occ = board.occ_all;
    // The one line that separates the two generators: everything that is not
    // ours, or only what is theirs.
    let alvos = if APENAS_CAPTURAS { board.occ_color[them.idx()] } else { !own };

    // Pawns
    let pawns = board.pieces[us.idx()][PieceType::Pawn.idx()];
    let (push_dir, start_rank, from_promo_rank): (i32, Bitboard, u8) = if us == Color::White {
        (8, RANK_2, 6) // pawn on rank 7 (index 6) promotes on push/capture
    } else {
        (-8, RANK_7, 1) // pawn on rank 2 (index 1) promotes
    };
    let mut p = pawns;
    while p != 0 {
        let from = pop_lsb(&mut p);
        let will_promote = rank_of(from) == from_promo_rank;
        let one = ((from as i32) + push_dir) as u8;
        if one < 64 && bb(one) & occ == 0 {
            if will_promote {
                add_promotions(out, from, one, false);
            } else if !APENAS_CAPTURAS {
                out.push(Move::quiet(from, one));
            }
            if !APENAS_CAPTURAS && bb(from) & start_rank != 0 {
                let two = ((from as i32) + push_dir * 2) as u8;
                if bb(two) & occ == 0 {
                    out.push(Move { from, to: two, promotion: None, flag: MoveFlag::DoublePush });
                }
            }
        }
        let attacks = atk.pawn[us.idx()][from as usize];
        let mut caps = attacks & board.occ_color[them.idx()];
        while caps != 0 {
            let to = pop_lsb(&mut caps);
            if will_promote {
                add_promotions(out, from, to, true);
            } else {
                out.push(Move::capture(from, to));
            }
        }
        if board.ep_square != NO_SQUARE && attacks & bb(board.ep_square) != 0 {
            out.push(Move { from, to: board.ep_square, promotion: None, flag: MoveFlag::EnPassant });
        }
    }

    // Knights
    let mut n = board.pieces[us.idx()][PieceType::Knight.idx()];
    while n != 0 {
        let from = pop_lsb(&mut n);
        gen_from_attacks(out, from, atk.knight[from as usize] & alvos, board, them);
    }
    // Bishops
    let mut b = board.pieces[us.idx()][PieceType::Bishop.idx()];
    while b != 0 {
        let from = pop_lsb(&mut b);
        gen_from_attacks(out, from, bishop_attacks(from, occ) & alvos, board, them);
    }
    // Rooks
    let mut r = board.pieces[us.idx()][PieceType::Rook.idx()];
    while r != 0 {
        let from = pop_lsb(&mut r);
        gen_from_attacks(out, from, rook_attacks(from, occ) & alvos, board, them);
    }
    // Queens
    let mut q = board.pieces[us.idx()][PieceType::Queen.idx()];
    while q != 0 {
        let from = pop_lsb(&mut q);
        gen_from_attacks(out, from, queen_attacks(from, occ) & alvos, board, them);
    }
    // King
    let king_sq = board.king_sq(us);
    gen_from_attacks(out, king_sq, atk.king[king_sq as usize] & alvos, board, them);

    // Castling. Never a capture, so it has no place in the quiescence
    // generator -- and skipping it also skips the in_check test and the four
    // is_square_attacked calls behind it, at every quiescence node.
    if !APENAS_CAPTURAS && !board.in_check(us, atk) {
        if us == Color::White {
            if board.castling & CASTLE_WK != 0
                && occ & (bb(5) | bb(6)) == 0
                && !board.is_square_attacked(5, them, atk)
                && !board.is_square_attacked(6, them, atk)
            {
                out.push(Move { from: 4, to: 6, promotion: None, flag: MoveFlag::CastleKing });
            }
            if board.castling & CASTLE_WQ != 0
                && occ & (bb(1) | bb(2) | bb(3)) == 0
                && !board.is_square_attacked(3, them, atk)
                && !board.is_square_attacked(2, them, atk)
            {
                out.push(Move { from: 4, to: 2, promotion: None, flag: MoveFlag::CastleQueen });
            }
        } else {
            if board.castling & CASTLE_BK != 0
                && occ & (bb(61) | bb(62)) == 0
                && !board.is_square_attacked(61, them, atk)
                && !board.is_square_attacked(62, them, atk)
            {
                out.push(Move { from: 60, to: 62, promotion: None, flag: MoveFlag::CastleKing });
            }
            if board.castling & CASTLE_BQ != 0
                && occ & (bb(57) | bb(58) | bb(59)) == 0
                && !board.is_square_attacked(59, them, atk)
                && !board.is_square_attacked(58, them, atk)
            {
                out.push(Move { from: 60, to: 58, promotion: None, flag: MoveFlag::CastleQueen });
            }
        }
    }
}

fn add_promotions(out: &mut Vec<Move>, from: Square, to: Square, _capture: bool) {
    for p in [PieceType::Queen, PieceType::Rook, PieceType::Bishop, PieceType::Knight] {
        out.push(Move { from, to, promotion: Some(p), flag: if _capture { MoveFlag::Capture } else { MoveFlag::Quiet } });
    }
}

fn gen_from_attacks(out: &mut Vec<Move>, from: Square, targets: Bitboard, board: &Board, them: Color) {
    let mut t = targets;
    while t != 0 {
        let to = pop_lsb(&mut t);
        if board.occ_color[them.idx()] & bb(to) != 0 {
            out.push(Move::capture(from, to));
        } else {
            out.push(Move::quiet(from, to));
        }
    }
}

/// Legality check via make/unmake on the SAME board (found in review,
/// 2026-07-21) instead of a full `Board::clone()` + `make_move()` per
/// pseudo-legal candidate -- every other legality check in the search
/// (negamax/quiescence's own move loops) already uses make/unmake on
/// one board; this was the one remaining spot still paying a full
/// struct clone (mailbox, accumulators, castling/ep state, everything)
/// per candidate move, at every single node. Requires `&mut Board`
/// instead of `&Board` -- every call site already holds a `&mut Board`
/// or an owned `Board`, so this is a mechanical signature change.
/// Bitboard of our pieces that are absolutely pinned to our king by an
/// enemy slider (found in review, 2026-07-21). A non-pinned, non-king
/// move that isn't en passant can never expose our own king, so its
/// legality needs no make/unmake test -- that's the fast path in
/// generate_legal below. Computed once per position: for each enemy
/// slider that would hit our king on an otherwise-empty board (a
/// "sniper"), if exactly one piece sits between it and the king, that
/// piece is pinned.
pub fn compute_pinned(board: &Board, atk: &Attacks, us: Color, king_sq: Square) -> Bitboard {
    let them = us.opp();
    let occ = board.occ_all;
    let enemy_rq = board.pieces[them.idx()][PieceType::Rook.idx()]
        | board.pieces[them.idx()][PieceType::Queen.idx()];
    let enemy_bq = board.pieces[them.idx()][PieceType::Bishop.idx()]
        | board.pieces[them.idx()][PieceType::Queen.idx()];
    // Snipers: sliders aligned with the king as if the board were empty.
    let snipers = (rook_attacks(king_sq, 0) & enemy_rq) | (bishop_attacks(king_sq, 0) & enemy_bq);
    let mut pinned = 0u64;
    let mut s = snipers;
    while s != 0 {
        let sniper_sq = s.trailing_zeros() as Square;
        s &= s - 1;
        let between = atk.between[king_sq as usize][sniper_sq as usize];
        let blockers = between & occ;
        // Exactly one piece between king and sniper, and it's ours ->
        // pinned. (If that one piece is theirs it's not our pin; if two+
        // pieces, no pin.)
        if blockers.count_ones() == 1 && (blockers & board.occ_color[us.idx()]) != 0 {
            pinned |= blockers;
        }
    }
    pinned
}

/// Legal captures, en passant and queen promotions -- the quiescence move list.
///
/// Same legality logic as `generate_legal`, over a generator that never built
/// the quiet moves in the first place. Quiescence used to call the full
/// generator and then `retain` the captures, which meant building about thirty
/// moves per node to keep five, at the ~30% of nodes that are quiescence.
///
/// Only correct when NOT in check: a side in check must consider every legal
/// evasion, quiet ones included, so quiescence keeps calling `generate_legal`
/// there.
pub fn generate_legal_caps(board: &mut Board, atk: &Attacks) -> Vec<Move> {
    gera_legal::<true>(board, atk)
}

pub fn generate_legal(board: &mut Board, atk: &Attacks) -> Vec<Move> {
    gera_legal::<false>(board, atk)
}

fn gera_legal<const APENAS_CAPTURAS: bool>(board: &mut Board, atk: &Attacks) -> Vec<Move> {
    // `pseudo`, the intermediate, is a reused thread-local scratch buffer, so
    // that a Vec is not allocated and freed once per NODE -- malloc/free was
    // ~4% of the profile. `legal` stays fresh, because the caller holds it
    // across the search recursion. There is no re-entrancy here: the recursion
    // calls gera_legal only AFTER this one returns, and the RefCell would
    // panic outright if some re-entrant path had been missed.
    std::thread_local! {
        static PSEUDO: std::cell::RefCell<Vec<Move>> =
            std::cell::RefCell::new(Vec::with_capacity(64));
    }
    PSEUDO.with(|cell| {
    let mut pseudo = cell.borrow_mut();
    pseudo.clear();
    gera_pseudo::<APENAS_CAPTURAS>(board, atk, &mut pseudo);
    let us = board.side;
    let king_sq = board.king_sq(us);
    let in_check = board.in_check(us, atk);
    let pinned = compute_pinned(board, atk, us, king_sq);
    let mut legal = Vec::with_capacity(pseudo.len());
    for &mv in pseudo.iter() {
        // Fast path: when not in check, a move by a non-pinned piece
        // that is neither a king move nor en passant cannot leave our
        // own king in check -- accept it without a make/unmake test.
        // Everything else (in check, pinned piece, king move, en
        // passant) falls back to the exact make/unmake check, which is
        // known-correct and validated by perft. Pinned pieces are NOT
        // simply rejected -- a pinned piece can still move along the
        // pin ray (or capture the pinner), so those go through the
        // fallback too rather than being wrongly dropped.
        let needs_check = in_check
            || mv.from == king_sq
            || mv.flag == MoveFlag::EnPassant
            || (bb(mv.from) & pinned) != 0;
        if !needs_check {
            legal.push(mv);
            continue;
        }
        // Second fast path: an ordinary king move is settled by asking
        // whether the destination is attacked, with the king lifted off its
        // old square and anything captured on the destination removed --
        // see `Board::king_move_leaves_check`. This is the largest group
        // that used to need a make/unmake, since a king has up to eight
        // moves and they were tested this way at every node, in check or
        // not.
        //
        // Castling is deliberately left on the slow path even though it is
        // a king move: `generate_pseudo_legal` already proves the squares it
        // crosses are unattacked, so the remaining question is not the same
        // one this answers, and the rook moves too.
        if mv.from == king_sq
            && mv.flag != MoveFlag::CastleKing
            && mv.flag != MoveFlag::CastleQueen
        {
            if !board.king_move_leaves_check(mv.from, mv.to, atk) {
                legal.push(mv);
            }
            continue;
        }
        let undo = board.make_move(&mv);
        let illegal = board.in_check(us, atk);
        board.unmake_move(&mv, &undo);
        if !illegal {
            legal.push(mv);
        }
    }
    legal
    })
}
