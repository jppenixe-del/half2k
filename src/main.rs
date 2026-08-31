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
mod nnue;
mod ordered_movegen;
mod perft;
mod refsearch;
mod search;
mod see;
mod tt;
mod types;
mod uci;
mod zobrist;

use attacks::Attacks;
use board::Board;
use types::Color;

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

/// Score one position, White's point of view.
///
/// White's rather than the side to move's because that is the frame the
/// reference prints in, and a comparison that has to flip a sign on the way is
/// a comparison with somewhere for a sign error to hide.
///
/// Reads the board's own accumulator rather than building one here, so this
/// exercises the path the search will use rather than a private shortcut.
fn eval_white(net: &nnue::Network, board: &Board) -> i32 {
    let acc = board.acc.as_ref().expect("no accumulator: install a network first");
    let stm_score = acc.eval(net, board.side, board.occ_all.count_ones());
    if board.side == Color::White {
        stm_score
    } else {
        -stm_score
    }
}

/// Load a network and install it, or die saying why.
fn install_net(path: &str) -> &'static nnue::Network {
    match nnue::load(path) {
        Ok(n) => {
            nnue::install(n);
            nnue::net().unwrap()
        }
        Err(e) => {
            eprintln!("network: {}", e);
            std::process::exit(1);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("perft") => {
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
        // `evalfens <network> <file of FENs>` -- one score per line, for
        // diffing against the oracle.
        Some("evalfens") => {
            let net = install_net(&args[2]);
            let text = std::fs::read_to_string(&args[3]).expect("fen file");
            for line in text.lines() {
                let fen = line.trim();
                if fen.is_empty() {
                    continue;
                }
                let board = Board::from_fen(fen);
                println!("{}", eval_white(net, &board));
            }
        }
        // `genverify [depth]` -- the ordered generator against the old one.
        Some("genverify") => {
            let depth: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);
            let atk = Attacks::new();
            let mut bad = 0u64;
            for (fen, _) in PERFT_SUITE {
                let mut board = Board::from_fen(fen);
                let (n, w) = perft::verify_generator(&mut board, depth, &atk);
                bad += w;
                println!("  {:>10} nodes  {:>4} wrong  {}", n, w, fen);
            }
            if bad == 0 {
                println!("\nthe two generators produce the same moves everywhere");
            } else {
                println!("\n{} disagreements", bad);
                std::process::exit(1);
            }
        }
        // `verify <network> [depth]` -- walk the same positions perft does, and
        // at every node compare the incrementally updated accumulator against
        // one rebuilt from scratch. Depth 4 already covers a few million nodes
        // and every kind of move that touches more than one square.
        Some("verify") => {
            install_net(&args[2]);
            let depth: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4);
            let atk = Attacks::new();
            let mut wrong_total = 0u64;
            for (fen, _) in PERFT_SUITE {
                let mut board = Board::from_fen(fen);
                let t = std::time::Instant::now();
                let (nodes, wrong) = perft::verify_accumulator(&mut board, depth, &atk);
                wrong_total += wrong;
                println!(
                    "  {:>10} nodes  {:>6} wrong  {:.1}s  {}",
                    nodes,
                    wrong,
                    t.elapsed().as_secs_f64(),
                    fen
                );
            }
            if wrong_total == 0 {
                println!("\nincremental matches a full rebuild everywhere");
            } else {
                println!("\n{} mismatches", wrong_total);
                std::process::exit(1);
            }
        }
        // `wdl <score>` -- the training win-rate model, for eyeballing.
        Some("wdl") => {
            let sc: i32 = args[2].parse().expect("score");
            let (w, d, l) = nnue::wdl(sc);
            println!("{} -> w {} d {} l {}", sc, w, d, l);
        }
        _ => uci::main_loop(),
    }
}
