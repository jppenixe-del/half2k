# half2k ultra

A UCI chess engine in Rust, with an efficiently updatable neural network for
evaluation.

## Building

```
cargo build --release
```

The binary looks for its weights, in order: the `EvalFile` UCI option, the
`HALF2K_NET` environment variable, then a file called `network.dat` beside the
binary. It reports `no network found` and evaluates everything as a draw if it
finds none.

## Checking it

Two commands exist because two different things can be wrong.

```
half2k perft            # move generation against published node counts
half2k verify <net> 4   # the accumulator, against a full rebuild at every node
```

`perft` walks six positions whose node counts are published and agreed on, so a
disagreement is ours rather than a matter of interpretation, and checks that
each position comes back byte for byte afterwards — which is what catches a
make/unmake pair that does not cancel, since perft alone would count a wrong
tree perfectly consistently.

`verify` compares the accumulator carried through make and unmake against one
rebuilt from nothing, at every node. It is the test that catches a castling rook
moved without telling the accumulator, an en passant capture removed from the
wrong square, or a promotion that adds the pawn back instead of the queen. Those
bugs never surface in a game until they decide one, because the evaluation stays
plausible while being wrong.

## Options

| | |
|---|---|
| `Hash` | table size in MB |
| `Move Overhead` | milliseconds held back from every time allocation |
| `EvalFile` | path to the network |

`Move Overhead` is worth setting deliberately. Being wrong about it is
asymmetric: too large costs a little strength, too small costs whole games.

## Licence

GPL-3.0-or-later; see `COPYING`. Provenance of anything not written here is in
`NOTICES.md`.
