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

// The companion to this -- a perft that also compares the incrementally
// updated accumulator against one rebuilt from scratch at every node -- lands
// with the network. It is the test that catches a castling rook moved without
// telling the accumulator, an en passant capture removed from the wrong square,
// or a promotion that adds the pawn back instead of the queen: the bugs that
// never surface in a normal game until they decide one, because the evaluation
// stays plausible while being wrong.
