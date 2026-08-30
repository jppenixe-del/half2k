# Notices

half2k ultra is distributed under the GNU General Public License, version 3 or
later. The full text is in `COPYING`.

This file records provenance: code that came from elsewhere, and under what
terms. Ideas are not code and are not listed here — a technique that many
programs implement is not anyone's property, and the source comments explain how
each one works rather than where it was first seen.

## Board representation, move generation, and the surrounding machinery

`src/types.rs`, `src/bitboard.rs`, `src/moves.rs`, `src/magic.rs`,
`src/attacks.rs`, `src/movegen.rs`, `src/board.rs`, `src/zobrist.rs`,
`src/perft.rs` and `src/tt.rs` come from Kestrel, our own engine, and carry its
GPL-3-or-later terms. `src/board.rs` has since been decoupled from Kestrel's
evaluation and rebuilt around this program's network.

## The network and its file format

The network is ours. Its file layout, feature indexing and quantisation follow
the format used by `pawn` by Rui Coelho
(<https://github.com/ruicoelhopedro/pawn>), GPL-3, which is the format the
network was trained against. No code was copied — the reader in `src/nnue.rs`
was written from the format's specification and is checked against that program
as an oracle.

## Tablebases

Syzygy probing uses Fathom (<https://github.com/jdart1/Fathom>), MIT licensed.
Its licence text travels with the vendored sources.
