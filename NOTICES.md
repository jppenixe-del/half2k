# Notices

half2k ultra is distributed under the GNU General Public License, version 3 or
later. The full text is in `COPYING`.

This file records provenance: code that came from somewhere else, and under what
terms. Ideas are not code and are not listed here — a technique that many
programs implement is not anyone's property, and the source comments explain how
each one works rather than where it was first seen.

## Board representation, move generation, and the surrounding machinery

`src/types.rs`, `src/bitboard.rs`, `src/moves.rs`, `src/magic.rs`,
`src/attacks.rs`, `src/movegen.rs`, `src/board.rs`, `src/zobrist.rs`,
`src/perft.rs` and `src/tt.rs` were lifted from Kestrel, our own engine, and
carry its GPL-3-or-later terms. `src/board.rs` has since been decoupled from
Kestrel's evaluation and rebuilt around this program's network.

## The network

`nnue-15a31a27e1d7.dat` is ours. It was trained in bullet on relabelled Leela
data filtered by our own recipe, and its file format is the one used by the
program the network was first built for — see below.

## The network file format

The layout of the weight file, the feature indexing, and the quantisation follow
the format of `pawn` by Rui Coelho (<https://github.com/ruicoelhopedro/pawn>),
GPL-3. The network we ship was trained against that exact format, so the reader
in `src/nnue.rs` implements it deliberately and must stay bit-compatible with it.
No code was copied; the format was reimplemented from its specification and is
verified against that program as an oracle.

## Tablebases

Syzygy probing uses Fathom (<https://github.com/jdart1/Fathom>), MIT licensed.
Its licence text travels with the vendored sources.
