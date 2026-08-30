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

/// Like perft(), but at every node it compares the network's incrementally
/// updated accumulator against one rebuilt from scratch.
///
/// This is the test that catches a castling rook that was moved without
/// telling the accumulator, an en-passant capture removed from the wrong
/// square, or a promotion that added the pawn back instead of the queen. Those
/// are exactly the bugs that never show up in a normal game until they decide
/// one, because the evaluation stays plausible while being wrong.
///
/// It replaces the same check that used to guard the hand-written
/// evaluation's incremental score. The evaluation changed; the class of bug
/// did not.
pub fn verify_incremental_eval(board: &mut Board, depth: u32, atk: &Attacks) -> (u64, u64) {
    let mut mismatches = 0u64;
    if let (Some(net), Some(acc)) = (crate::nnue::rede(), board.acc.as_ref()) {
        let fresco = crate::nnue::Accumulator::fresh(net, board);
        if fresco.white != acc.white || fresco.black != acc.black {
            mismatches += 1;
            eprintln!("MISMATCH acumulador fen={}", board.to_fen());
        }
    }
    if board.phase != {
        let mut f = board.clone();
        f.recompute_eval_accumulators();
        f.phase
    } {
        mismatches += 1;
        eprintln!("MISMATCH fase fen={}", board.to_fen());
    }
    if depth == 0 {
        return (1, mismatches);
    }
    let moves = generate_legal(board, atk);
    let mut nodes = 1u64;
    for mv in moves {
        let undo = board.make_move(&mv);
        let (n, m) = verify_incremental_eval(board, depth - 1, atk);
        nodes += n;
        mismatches += m;
        board.unmake_move(&mv, &undo);
    }
    (nodes, mismatches)
}
