// The UCI front end.
//
// The search runs on this thread rather than a worker, so `stop` is only acted
// on between moves. That is deliberate for now: the engine limits its own time
// and never relies on being told to stop, which is the behaviour that matters
// at a time control. Pondering and `go infinite` need the worker and will get
// it when there is something to ponder with.

use crate::board::Board;
use crate::movegen::generate_legal;
use crate::nnue;
use crate::search::{Limits, Searcher};
use crate::types::Color;
use std::io::BufRead;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

pub const NAME: &str = "half2k";
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Where to look for the weights, in order: the UCI option, the environment,
/// then next to the binary. The last one is what lets a worker that only knows
/// how to build and run the binary find them without being told.
fn find_network(explicit: &str) -> Option<String> {
    let mut tries: Vec<String> = Vec::new();
    if !explicit.is_empty() && explicit != "<empty>" {
        tries.push(explicit.to_string());
    }
    if let Ok(p) = std::env::var("HALF2K_NET") {
        tries.push(p);
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            tries.push(dir.join("network.dat").to_string_lossy().into_owned());
        }
    }
    tries.into_iter().find(|p| std::path::Path::new(p).exists())
}

pub fn main_loop() {
    let stop = Arc::new(AtomicBool::new(false));
    let mut hash_mb = 16usize;
    let mut searcher = Searcher::new(hash_mb, stop.clone());
    let mut board = Board::startpos();
    let mut net_path = String::new();
    let mut net_loaded = false;

    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        // End of input ends the program. Reading a failed line as an empty
        // command and going round again would spin a core forever, which is a
        // real way to make a build system hang with no error at all.
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let mut parts = line.split_whitespace();
        let cmd = match parts.next() {
            Some(c) => c,
            None => continue,
        };

        match cmd {
            "uci" => {
                println!("id name {} {}", NAME, VERSION);
                println!("id author the half2k authors");
                println!("option name Hash type spin default 16 min 1 max 65536");
                println!("option name Threads type spin default 1 min 1 max 1");
                println!(
                    "option name Move Overhead type spin default {} min 0 max 5000",
                    searcher.move_overhead
                );
                println!("option name EvalFile type string default <empty>");
                // Off by default, every one of them: out of the box the search
                // uses the smaller, settled set of ideas, so anything switched
                // on has a number of its own rather than being lost in a pile
                // of simultaneous changes.
                for name in crate::search::Features::EXTRA {
                    println!("option name {} type check default false", name);
                }
                // These are part of the baseline and start on. The switch is
                // for measuring them, not for leaving them out.
                for name in crate::search::Features::BASELINE {
                    println!("option name {} type check default true", name);
                }
                // Every number the search compares against. Not one was
                // measured, so every one is a candidate.
                let d = crate::search::Params::default();
                for (name, get, _, lo, hi) in crate::search::PARAM_SPECS {
                    println!(
                        "option name {} type spin default {} min {} max {}",
                        name,
                        get(&d),
                        lo,
                        hi
                    );
                }
                println!("uciok");
            }
            "isready" => {
                if !net_loaded {
                    match find_network(&net_path) {
                        Some(p) => match nnue::load(&p) {
                            Ok(n) => {
                                nnue::install(n);
                                net_loaded = true;
                                // Rebuild: the position was made before there
                                // was a network to build an accumulator with.
                                board = Board::from_fen(&board.to_fen());
                            }
                            Err(e) => eprintln!("info string network {}: {}", p, e),
                        },
                        None => eprintln!("info string no network found"),
                    }
                }
                println!("readyok");
            }
            "setoption" => {
                // `setoption name <words> value <words>` -- the name can carry
                // spaces, so it is everything between the two keywords.
                let rest: Vec<&str> = line.split_whitespace().collect();
                let name_at = rest.iter().position(|w| *w == "name");
                let value_at = rest.iter().position(|w| *w == "value");
                if let (Some(n), Some(v)) = (name_at, value_at) {
                    let name = rest[n + 1..v].join(" ").to_lowercase();
                    let value = rest[v + 1..].join(" ");
                    match name.as_str() {
                        "hash" => {
                            if let Ok(mb) = value.parse::<usize>() {
                                hash_mb = mb.clamp(1, 65536);
                                let mo = searcher.move_overhead;
                                let ft = searcher.features;
                                let pr = searcher.params;
                                searcher = Searcher::new(hash_mb, stop.clone());
                                searcher.move_overhead = mo;
                                searcher.features = ft;
                                searcher.params = pr;
                                searcher.params_changed();
                            }
                        }
                        "move overhead" => {
                            if let Ok(ms) = value.parse::<u64>() {
                                searcher.move_overhead = ms.min(5000);
                            }
                        }
                        "evalfile" => {
                            net_path = value;
                            net_loaded = false;
                        }
                        other => {
                            if let Ok(n) = value.parse::<i32>() {
                                if searcher.params.set(other, n) {
                                    searcher.params_changed();
                                    continue;
                                }
                            }
                            let on = value.eq_ignore_ascii_case("true");
                            searcher.features.set(other, on);
                        }
                    }
                }
            }
            "ucinewgame" => {
                searcher.clear();
                board = Board::startpos();
                searcher.set_game_history(vec![board.hash]);
            }
            "position" => {
                let rest: Vec<&str> = line.split_whitespace().collect();
                let mut i = 1;
                if rest.get(i) == Some(&"startpos") {
                    board = Board::startpos();
                    i += 1;
                } else if rest.get(i) == Some(&"fen") {
                    let end = rest
                        .iter()
                        .position(|w| *w == "moves")
                        .unwrap_or(rest.len());
                    board = Board::from_fen(&rest[i + 1..end].join(" "));
                    i = end;
                }
                // Every position along the way is kept, because that is what a
                // repetition is measured against. Losing the history here makes
                // the search blind to a draw it is one move away from.
                let mut keys = vec![board.hash];
                let mut played: Vec<crate::moves::Move> = Vec::new();
                if rest.get(i) == Some(&"moves") {
                    let atk = &searcher.atk;
                    for token in &rest[i + 1..] {
                        let legal = generate_legal(&mut board, atk);
                        if let Some(mv) = legal.iter().find(|m| m.to_uci() == *token) {
                            board.make_move(mv);
                            keys.push(board.hash);
                            played.push(*mv);
                        } else {
                            eprintln!("info string illegal move in position: {}", token);
                            break;
                        }
                    }
                }
                searcher.set_game_history(keys);
                searcher.set_game_moves(&played);
            }
            "go" => {
                let mut limits = Limits::default();
                let rest: Vec<&str> = line.split_whitespace().collect();
                let mut i = 1;
                while i < rest.len() {
                    let val = rest.get(i + 1).and_then(|v| v.parse::<u64>().ok());
                    match rest[i] {
                        "wtime" => limits.wtime = val,
                        "btime" => limits.btime = val,
                        "winc" => limits.winc = val.unwrap_or(0),
                        "binc" => limits.binc = val.unwrap_or(0),
                        "movestogo" => limits.movestogo = val,
                        "movetime" => limits.movetime = val,
                        "depth" => limits.depth = val.map(|v| v as u32),
                        "nodes" => limits.nodes = val,
                        "infinite" => limits.infinite = true,
                        _ => {}
                    }
                    i += 1;
                }
                let best = searcher.go(&mut board, &limits, true);
                if std::env::var_os("HALF2K_DBG").is_some() {
                    eprintln!("{}", crate::refsearch::dbg_report());
                }
                match best {
                    Some(m) => println!("bestmove {}", m.to_uci()),
                    // Nothing legal: say so rather than go quiet, which reads
                    // to whoever is waiting as a hung engine.
                    None => println!("bestmove 0000"),
                }
            }
            "stop" => {}
            "quit" => break,
            "eval" => {
                let side = board.side;
                let s = crate::search::debug_eval(&board, searcher.features.rule50_fade);
                let (w, d, l) = nnue::wdl(s);
                println!(
                    "eval {} (side to move: {}) wdl {} {} {}",
                    s,
                    if side == Color::White { "white" } else { "black" },
                    w,
                    d,
                    l
                );
            }
            _ => {}
        }
    }
}
