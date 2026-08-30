use crate::attacks::Attacks;
use crate::board::Board;
use crate::movegen::generate_legal;

pub fn perft(board: &mut Board, depth: u32, atk: &Attacks) -> u64 {
    if depth == 0 {
        return 1;
    }
    let moves = generate_legal(board, atk);
    if depth == 1 {
        return moves.len() as u64;
    }
    let mut nodes = 0u64;
    for mv in moves {
        let undo = board.make_move(&mv);
        nodes += perft(board, depth - 1, atk);
        board.unmake_move(&mv, &undo);
    }
    nodes
}

/// Like `perft`, but at every node it compares the incrementally updated
/// accumulator against one rebuilt from scratch.
///
/// This is the test that catches a castling rook moved without telling the
/// accumulator, an en passant capture removed from the wrong square, or a
/// promotion that adds the pawn back instead of the queen. Those are exactly
/// the bugs that never surface in a normal game until they decide one, because
/// the evaluation stays plausible while being wrong.
///
/// Returns (nodes, mismatches).
pub fn verify_accumulator(board: &mut Board, depth: u32, atk: &Attacks) -> (u64, u64) {
    let mut wrong = 0u64;
    if let (Some(net), Some(acc)) = (crate::nnue::net(), board.acc.as_ref()) {
        let kings = [
            board.king_sq(crate::types::Color::White),
            board.king_sq(crate::types::Color::Black),
        ];
        let fresh = crate::nnue::Accumulator::fresh(net, &board.pieces, kings);
        for p in 0..2 {
            if fresh.half[p].values != acc.half[p].values
                || fresh.half[p].psqt != acc.half[p].psqt
                || fresh.half[p].bucket != acc.half[p].bucket
                || fresh.half[p].mirror != acc.half[p].mirror
            {
                wrong += 1;
                if wrong <= 5 {
                    eprintln!("MISMATCH perspective {} at {}", p, board.to_fen());
                }
            }
        }
    }
    if depth == 0 {
        return (1, wrong);
    }
    let moves = generate_legal(board, atk);
    let mut nodes = 1u64;
    for mv in moves {
        let undo = board.make_move(&mv);
        let (n, w) = verify_accumulator(board, depth - 1, atk);
        nodes += n;
        wrong += w;
        board.unmake_move(&mv, &undo);
    }
    (nodes, wrong)
}
