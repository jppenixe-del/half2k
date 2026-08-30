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
mod magic;
mod moves;
mod tt;
mod types;

// Still to come, in gate order: `board` needs its evaluation coupling cut before
// `movegen`, `zobrist` and `perft` can join it, and the network reader lands
// after that.

fn main() {
    println!("half2k 0.1.0");
}
