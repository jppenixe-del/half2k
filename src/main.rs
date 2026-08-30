// half2k ultra
//
// Copyright (C) 2026  the half2k authors
//
// This program is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version. See COPYING.
//
// Provenance of anything not written here is recorded in NOTICES.md.

mod attacks;
mod bitboard;
mod board;
mod magic;
mod movegen;
mod moves;
mod perft;
mod tt;
mod types;
mod zobrist;

use attacks::Attacks;
use board::Board;

/// Positions whose node counts are published and agreed on, so a disagreement
/// is ours and not a matter of interpretation. Between them they exercise
/// castling through attacked squares, en passant that would expose the king,
/// promotion with and without capture, and double check.
const PERFT_SUITE: &[(&str, &[u64])] = &[
    (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        &[20, 400, 8902, 197281, 4865609, 119060324],
    ),
    (
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        &[48, 2039, 97862, 4085603, 193690690],
    ),
    (
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        &[14, 191, 2812, 43238, 674624, 11030083],
    ),
    (
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        &[6, 264, 9467, 422333, 15833292],
    ),
    (
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        &[44, 1486, 62379, 2103487, 89941194],
    ),
    (
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        &[46, 2079, 89890, 3894594, 164075551],
    ),
];

/// Runs the suite and reports. Returns the number of disagreements.
fn perft_suite(max_nodes: u64) -> u32 {
    let atk = Attacks::new();
    let mut bad = 0u32;
    for (fen, expected) in PERFT_SUITE {
        let mut board = Board::from_fen(fen);
        println!("{}", fen);
        for (i, want) in expected.iter().enumerate() {
            if *want > max_nodes {
                println!("  depth {}  skipped, {} nodes", i + 1, want);
                continue;
            }
            let t = std::time::Instant::now();
            let got = perft::perft(&mut board, (i + 1) as u32, &atk);
            let secs = t.elapsed().as_secs_f64();
            let mark = if got == *want {
                "ok"
            } else {
                bad += 1;
                "WRONG"
            };
            println!(
                "  depth {}  {:>12}  want {:>12}  {:>5}  {:.2}s  {:.0} nps",
                i + 1,
                got,
                want,
                mark,
                secs,
                got as f64 / secs.max(1e-9)
            );
        }
        // The position has to come back exactly as it went in. A make/unmake
        // pair that does not cancel shows up here and nowhere else, because
        // perft itself only counts leaves and would happily count the wrong
        // tree consistently.
        if board.to_fen() != *fen {
            println!("  WRONG  board did not survive: {}", board.to_fen());
            bad += 1;
        }
    }
    bad
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("perft") => {
            // Everything up to and including the six-figure depths by default;
            // pass a bigger budget to run the long ones.
            let budget: u64 = args
                .get(2)
                .and_then(|s| s.parse().ok())
                .unwrap_or(200_000_000);
            let bad = perft_suite(budget);
            if bad == 0 {
                println!("\nall positions agree");
            } else {
                println!("\n{} disagreements", bad);
                std::process::exit(1);
            }
        }
        _ => println!("half2k 0.1.0"),
    }
}
