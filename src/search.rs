// The search.
//
// Deliberately a small, sound search rather than a full one: iterative
// deepening with aspiration windows, principal variation search, quiescence,
// transposition table, killers and history, null move and late move
// reductions. Everything beyond that is added one at a time and measured, so
// that what each addition is worth is a number and not an opinion.
//
// Time management is here from the start rather than bolted on later, because
// an engine that loses on the clock is worth nothing whatever it scores at
// infinite time. See `allocate`.

use crate::attacks::Attacks;
use crate::board::Board;
use crate::moves::{Move, MoveFlag};
use crate::movegen::{generate_legal, generate_legal_caps};
use crate::nnue;
use crate::see;
use crate::tt::{Bound, TranspositionTable, TT_EVAL_NONE};
use crate::types::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// How much to reduce a late quiet move by, indexed by depth and by how many
/// moves have already been tried.
///
/// A table rather than an expression because the expression wanted logarithms,
/// and the first version computed them in integers -- `ln(3)` and `ln(4)` both
/// truncate to 1, so the reduction was very nearly a constant and the whole
/// point of reducing later moves harder was lost.
///
/// Rebuilt when its two numbers change rather than computed per node: four
/// thousand logarithms is nothing once a move, and real work once a node.
fn build_lmr_table(base: i32, div: i32) -> [[i32; 64]; 64] {
    let base = base as f64 / 100.0;
    let div = (div as f64 / 100.0).max(0.01);
    let mut t = [[0i32; 64]; 64];
    for d in 1..64usize {
        for m in 1..64usize {
            t[d][m] = (base + (d as f64).ln() * (m as f64).ln() / div) as i32;
        }
    }
    t
}

/// Correction history: how wrong the static evaluation usually is here.
///
/// The pruning margins are fixed numbers compared against the static score.
/// When that score is *systematically* wrong for a family of positions -- and
/// it is, because the network cannot see what only the search finds -- those
/// margins bite in the wrong place, the same way every time. This keeps a
/// running average of what the search ended up saying minus what the static
/// score said, indexed by pawn structure, and feeds it back next time that
/// structure appears.
///
/// The key is the two pawn bitboards mixed together rather than an incremental
/// key. Derived from the state, it cannot drift out of sync with it, and a
/// heuristic table tolerates collisions by construction.
const CORR_SIZE: usize = 16384;

/// Where one table's entry saturates.
///
/// It was 8192 with a grain of 256, which meant the whole correction, summed
/// over every table, could reach thirty two units -- sixteen centipawns. The
/// reverse futility margin is a hundred and fifty per ply. A correction that
/// small cannot move a pruning decision, which is the only thing it exists to
/// do, and measuring it found exactly the nothing that implies.
const CORR_MAX: i32 = 1024;

/// The most one update may move an entry.
const CORR_MAX_UPDATE: i32 = CORR_MAX / 4;

/// How much each reading is trusted, out of the divisor below.
///
/// Not an average. The tables answer different questions and are not equally
/// good at it: the pawn structure says the most by a wide margin, what remains
/// of each side's force says less, and the shape of the last move says less
/// again. Proportions taken from an engine where they were tuned over games.
const CORR_WEIGHT: [i32; CORR_KINDS] = [203, 109, 109, 121, 72];
const CORR_DIVISOR: i32 = 2048;

/// How many separate readings of "what kind of position is this" the
/// correction is spread over.
///
/// Five, and which five matters more than how many. Two of the first set were
/// a rook-and-queen key and a knight-and-bishop key, and both were dropped:
/// each is a strict subset of the non-pawn key, so they answered a question
/// already answered and were only adding their own collisions to it. The
/// non-pawn key took their place twice, once per colour -- how much force each
/// side has left are different facts and were being merged into one.
///
/// The fifth is the last move as a change rather than as a destination: the
/// difference between the position key before and after it, which carries
/// where the piece came from, what it took and whose move it was. What it
/// learns is "a change of this shape tends to be misread", which is not
/// something the piece-and-square key can express.
pub const CORR_KINDS: usize = 5;

#[inline]
fn mix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// A Zobrist key over one subset of the pieces.
///
/// Built from the same per-piece randoms the position key uses, so two boards
/// share a key only when the chosen pieces stand on the same squares in the
/// same colours. That is the whole point and it was what the previous version
/// threw away: it keyed on the occupancy bitboard, which cannot tell a white
/// queen on d1 from a black knight on d1, and on `both()` bitboards that merged
/// the colours outright. A table indexed like that files a correction learned
/// in one position where an unrelated position will read it, and the two teach
/// each other nothing -- measured, switching it on cost fourteen Elo.
#[inline]
fn subset_key(board: &Board, colors: &[Color], types: &[PieceType]) -> u64 {
    let z = crate::zobrist::tabelas();
    let mut k = 0u64;
    for &c in colors {
        for &t in types {
            let mut bb = board.pieces[c.idx()][t.idx()];
            while bb != 0 {
                let sq = bb.trailing_zeros() as usize;
                bb &= bb - 1;
                k ^= z.piece_sq[c.idx()][t.idx()][sq];
            }
        }
    }
    k
}

/// The five indices for this position: pawns, White's remaining force, Black's
/// remaining force, the last move by piece and destination, and the last move
/// as a change of position key.
#[inline]
fn corr_indices(
    board: &Board,
    last: Option<(usize, usize)>,
    hash_delta: u64,
) -> [usize; CORR_KINDS] {
    use Color::{Black, White};
    use PieceType::{Bishop, Knight, Pawn, Queen, Rook};

    const BOTH: [Color; 2] = [White, Black];
    const FORCE: [PieceType; 4] = [Knight, Bishop, Rook, Queen];

    let pawns = subset_key(board, &BOTH, &[Pawn]);
    let np_white = subset_key(board, &[White], &FORCE);
    let np_black = subset_key(board, &[Black], &FORCE);

    let cont = match last {
        Some((pc, to)) => mix((pc as u64) << 8 | to as u64 | 0x5eed_0000_0000),
        None => 0,
    };

    // Zero when there is no previous position to differ from, which files
    // everything at the root in one slot rather than in a random one.
    let trans = if hash_delta == 0 { 0 } else { mix(hash_delta) };

    // The subset keys are XORs of randoms, so mix once more before taking the
    // low bits as an index.
    let m = (CORR_SIZE - 1) as u64;
    [
        (mix(pawns) & m) as usize,
        (mix(np_white) & m) as usize,
        (mix(np_black) & m) as usize,
        (cont & m) as usize,
        (trans & m) as usize,
    ]
}

/// How many continuation tables, and how far back each looks.
/// Quiet moves remembered per ply as having caused a cutoff. Three rather
/// than two: the extra one costs a comparison and catches the case where two
/// different refutations alternate, which two slots lose to immediately.
pub const NUM_KILLERS: usize = 3;

pub const CONT_SLOTS: usize = 3;
pub const CONT_BACK: [usize; CONT_SLOTS] = [1, 2, 4];
/// Weight per slot when scoring, the reply carrying twice the rest.
pub const CONT_WEIGHT: [i32; CONT_SLOTS] = [2, 1, 1];

/// Where a history entry settles. Separate ceilings because the two tables
/// answer different questions and the continuation one is asked more precisely,
/// so it is allowed to be more emphatic.
const HIST_MAX_MAIN: i32 = 15000;
const HIST_MAX_CONT: i32 = 30000;

/// What one cutoff is worth, by the depth that found it.
///
/// Linear and generous. It was `min(1200, d*d)` -- at depth ten that is a
/// hundred against two thousand, so the tables took eighty cutoffs to reach
/// where they now reach in six, and move ordering spent most of its time acting
/// on information that had gone stale. Ordering is what decides whether late
/// move reductions are reducing the right moves, so this is not a small thing.
#[inline]
fn hist_bonus(depth: i32) -> i32 {
    (200 * depth).min(4000)
}

/// What one capture cutoff is worth.
///
/// Its own curve, steeper and with an offset, rather than the quiet one. A
/// capture that works is a stronger statement than a quiet move that works --
/// there were fewer of them to choose from and the material says whether it
/// paid -- so the table should move further on it. Reusing the quiet bonus made
/// this three times too small at the depths where it matters.
#[inline]
fn capt_bonus(depth: i32) -> i32 {
    (depth * 680 - 250).clamp(0, 2400)
}

/// Where a capture history entry settles.
const HIST_MAX_CAPT: i32 = 16384;

/// Move towards the ceiling by an amount that shrinks as it is approached, so
/// an entry saturates instead of running away.
#[inline]
fn hist_add(entry: &mut i32, bonus: i32, max: i32) {
    *entry += bonus - *entry * bonus.abs() / max;
}

/// How much of the plan this search has earned, judged between iterations.
///
/// Four readings of "does this position still need thinking", multiplied.
///
/// `effort` is the share of nodes that went into the move we mean to play. A
/// search that has poured almost everything into one move has found its answer
/// and is confirming it; one still splitting nodes across rivals has not
/// decided. This is the sturdiest of the four, being a ratio over millions of
/// nodes rather than a verdict that can turn on one.
///
/// `settle` counts iterations that kept the same root move, and decays fast. It
/// is deliberately the weakest term. Keying elastic time on stability ALONE was
/// tried in an earlier engine of ours and reverted the same day: on a forced
/// recapture, where there is nothing to decide, the root move still changed at
/// three separate depths and each change threw the multiplier back to maximum.
/// Stability may lengthen a search, never on its own, and never without the
/// wall standing behind it.
///
/// `falling` is a score dropping between iterations, which neither of the
/// others can see: effort stays high while a position collapses under it, and
/// stability watches whether the move changed rather than whether it got worse.
/// Only falls count -- paying extra for good news is how a clock is spent on
/// won positions.
///
/// `instability` is a move the search keeps overturning. It separates a hard
/// position from a slow one, which the first two cannot: both read "several
/// moves are equally good" as difficulty, so they fire on quiet positions with
/// many reasonable answers.
fn time_scale(effort_frac: f64, settle: u32, score_drop: i32, changes: u32) -> f64 {
    let effort = (1.40 - effort_frac) * 1.55;
    let settle = (0.95 + 2.2 * (settle as f64 + 2.6).powf(-1.5)).max(1.0);
    let falling = if score_drop > 0 {
        (1.0 + score_drop as f64 * 0.004).min(1.5)
    } else {
        1.0
    };
    let instability = (1.0 + changes as f64 * 0.22).min(2.4);
    (effort * settle * falling * instability).clamp(0.65, 3.4)
}

/// How far above the root the continuation tables may reach into the game.
pub const PRE_MOVES: usize = 6;

/// Floor of the base two logarithm, which is what the alternative reduction
/// formula is built on.
#[inline]
fn ilog2i(v: i32) -> i32 {
    if v <= 0 {
        0
    } else {
        31 - (v as u32).leading_zeros() as i32
    }
}

pub const MAX_PLY: usize = 128;
pub const INF: i32 = 32_000;
pub const MATE: i32 = 31_000;
/// Anything at least this large is a mate score, not an evaluation.
pub const MATE_IN_MAX: i32 = MATE - MAX_PLY as i32;


/// Every number the search compares something against.
///
/// They are options rather than constants because not one of them was measured
/// -- each was picked to be sane and then left alone, which is a different
/// thing from being right. Exposed, they can be walked over by a tuner playing
/// games, which is the only process that has ever produced good ones.
///
/// Two are stored multiplied by a hundred, because the shape they belong to is
/// a logarithm and the option protocol only carries integers.
#[derive(Clone, Copy)]
pub struct Params {
    pub rfp_margin: i32,
    pub rfp_improving: i32,
    pub rfp_depth: i32,
    pub razor_margin: i32,
    pub razor_depth: i32,
    pub nmp_base: i32,
    pub nmp_div: i32,
    pub lmp_base: i32,
    pub lmp_depth: i32,
    pub fut_base: i32,
    pub fut_slope: i32,
    pub fut_depth: i32,
    /// Divisor applied to the history score inside the forward futility
    /// margin. In OUR history units, which run about five and a half times
    /// smaller than the scale this value was originally written for.
    pub fut_hist_div: i32,
    pub hist_prune: i32,
    /// Static exchange threshold for quiet moves, as `-x * (d + d*d)`.
    pub see_prune_quiet: i32,
    /// A check is only worth extending for when the position is not already
    /// decided.
    pub check_ext_eval: i32,
    /// What a draw is worth to the side to move, in the units the network
    /// speaks, where two are a centipawn.
    ///
    /// Zero means a draw is a draw. Above zero means the engine would rather
    /// keep playing than repeat, which is worth something when the opponent
    /// evaluates positions the same way we do -- against an engine sharing our
    /// network, agreement about what is equal turns into a repetition, and half
    /// the games end that way.
    ///
    /// It is not free: an engine that refuses a draw it should take loses games
    /// it should have halved. Small, and measured.
    pub contempt: i32,
    pub see_prune: i32,
    /// x100
    pub lmr_base: i32,
    /// x100
    pub lmr_div: i32,
    pub lmr_cut: i32,
    pub lmr_hist_div: i32,
    pub asp_delta: i32,
    pub asp_depth: i32,
    pub sing_depth: i32,
    pub sing_margin: i32,
    /// How far below the singular window a move has to fall to earn a second
    /// ply rather than one.
    pub double_ext: i32,
    /// Reductions accumulate in 1024ths and divide at the end, so a term can be
    /// worth a third of a ply instead of all or nothing. These are in those
    /// units.
    pub lmr_cut_f: i32,
    pub lmr_nonpv_f: i32,
    pub lmr_ttpv_f: i32,
    /// A stored lower bound this far above beta already answers the question.
    ///
    /// In OUR units. The value it came from belongs to a program whose pawn is
    /// about 255 where ours is 200, so it is scaled by that ratio -- the third
    /// time today a constant has been carried across a scale boundary, and the
    /// first two both silently disabled the thing they were meant to control.
    pub probcut_margin: i32,
    pub tm_mtg: i32,
    /// percent of the increment spent each move
    pub tm_inc_pct: i32,
    pub tm_hard_mult: i32,
    /// percent of what is left that the wall may reach
    pub tm_hard_pct: i32,
    /// Percent of the base allowance by game phase. The opening is played
    /// rather than calculated, and a simplified position has less to find.
    pub tm_open_pct: i32,
    pub tm_early_pct: i32,
    pub tm_mid_pct: i32,
    pub tm_late_pct: i32,
    pub tm_simple_pct: i32,
    /// What to do about the other clock: more when comfortably ahead on it,
    /// less when behind.
    pub tm_ahead_pct: i32,
    pub tm_behind_pct: i32,
    /// Never think for less than this, so a low clock still buys a move that
    /// was looked at rather than one that was guessed.
    pub tm_floor_ms: i32,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            rfp_margin: 150,
            rfp_improving: 150,
            rfp_depth: 9,
            razor_margin: 348,
            razor_depth: 5,
            nmp_base: 4,
            nmp_div: 6,
            lmp_base: 3,
            lmp_depth: 6,
            fut_base: 100,
            fut_slope: 150,
            fut_depth: 12,
            fut_hist_div: 75,
            hist_prune: 600,
            see_prune: 70,
            see_prune_quiet: 5,
            check_ext_eval: 75,
            contempt: 0,
            lmr_base: 77,
            lmr_div: 236,
            lmr_cut: 2,
            lmr_hist_div: 22000,
            asp_delta: 25,
            asp_depth: 4,
            sing_depth: 5,
            sing_margin: 2,
            double_ext: 40,
            lmr_cut_f: 2048,
            lmr_nonpv_f: 1024,
            lmr_ttpv_f: 1024,
            probcut_margin: 294,
            tm_mtg: 46,
            tm_inc_pct: 75,
            tm_hard_mult: 4,
            tm_hard_pct: 25,
            tm_open_pct: 30,
            tm_early_pct: 70,
            tm_mid_pct: 110,
            tm_late_pct: 100,
            tm_simple_pct: 60,
            tm_ahead_pct: 115,
            tm_behind_pct: 85,
            tm_floor_ms: 10,
        }
    }
}

/// Name, current value, and the range a tuner may walk it over.
pub type ParamSpec = (&'static str, fn(&Params) -> i32, fn(&mut Params, i32), i32, i32);

pub const PARAM_SPECS: &[ParamSpec] = &[
    ("RfpMargin", |p| p.rfp_margin, |p, v| p.rfp_margin = v, 40, 300),
    ("RfpImproving", |p| p.rfp_improving, |p, v| p.rfp_improving = v, 0, 300),
    ("RfpDepth", |p| p.rfp_depth, |p, v| p.rfp_depth = v, 2, 12),
    ("RazorMargin", |p| p.razor_margin, |p, v| p.razor_margin = v, 50, 900),
    ("RazorDepth", |p| p.razor_depth, |p, v| p.razor_depth = v, 1, 10),
    ("NmpBase", |p| p.nmp_base, |p, v| p.nmp_base = v, 2, 8),
    ("NmpDiv", |p| p.nmp_div, |p, v| p.nmp_div = v, 2, 12),
    ("LmpBase", |p| p.lmp_base, |p, v| p.lmp_base = v, 1, 10),
    ("LmpDepth", |p| p.lmp_depth, |p, v| p.lmp_depth = v, 2, 12),
    ("FutBase", |p| p.fut_base, |p, v| p.fut_base = v, 20, 400),
    ("FutSlope", |p| p.fut_slope, |p, v| p.fut_slope = v, 30, 300),
    ("FutDepth", |p| p.fut_depth, |p, v| p.fut_depth = v, 2, 16),
    ("FutHistDiv", |p| p.fut_hist_div, |p, v| p.fut_hist_div = v, 20, 200),
    ("HistPrune", |p| p.hist_prune, |p, v| p.hist_prune = v, 100, 2000),
    ("SeePrune", |p| p.see_prune, |p, v| p.see_prune = v, 20, 250),
    ("SeePruneQuiet", |p| p.see_prune_quiet, |p, v| p.see_prune_quiet = v, 1, 40),
    ("CheckExtEval", |p| p.check_ext_eval, |p, v| p.check_ext_eval = v, 0, 400),
    ("Contempt", |p| p.contempt, |p, v| p.contempt = v, 0, 100),
    ("LmrBase", |p| p.lmr_base, |p, v| p.lmr_base = v, 0, 200),
    ("LmrDiv", |p| p.lmr_div, |p, v| p.lmr_div = v, 120, 400),
    ("LmrCut", |p| p.lmr_cut, |p, v| p.lmr_cut = v, 0, 4),
    ("LmrHistDiv", |p| p.lmr_hist_div, |p, v| p.lmr_hist_div = v, 4096, 65536),
    ("AspDelta", |p| p.asp_delta, |p, v| p.asp_delta = v, 8, 80),
    ("AspDepth", |p| p.asp_depth, |p, v| p.asp_depth = v, 2, 10),
    ("SingDepth", |p| p.sing_depth, |p, v| p.sing_depth = v, 4, 12),
    ("SingMargin", |p| p.sing_margin, |p, v| p.sing_margin = v, 1, 8),
    ("DoubleExt", |p| p.double_ext, |p, v| p.double_ext = v, 5, 200),
    ("LmrCutF", |p| p.lmr_cut_f, |p, v| p.lmr_cut_f = v, 0, 2048),
    ("LmrNonPvF", |p| p.lmr_nonpv_f, |p, v| p.lmr_nonpv_f = v, 0, 2048),
    ("LmrTtPvF", |p| p.lmr_ttpv_f, |p, v| p.lmr_ttpv_f = v, 0, 2048),
    ("ProbcutMargin", |p| p.probcut_margin, |p, v| p.probcut_margin = v, 100, 1024),
    ("TmMtg", |p| p.tm_mtg, |p, v| p.tm_mtg = v, 15, 70),
    ("TmIncPct", |p| p.tm_inc_pct, |p, v| p.tm_inc_pct = v, 20, 95),
    ("TmHardMult", |p| p.tm_hard_mult, |p, v| p.tm_hard_mult = v, 1, 8),
    ("TmHardPct", |p| p.tm_hard_pct, |p, v| p.tm_hard_pct = v, 15, 70),
    ("TmOpenPct", |p| p.tm_open_pct, |p, v| p.tm_open_pct = v, 20, 120),
    ("TmEarlyPct", |p| p.tm_early_pct, |p, v| p.tm_early_pct = v, 40, 140),
    ("TmMidPct", |p| p.tm_mid_pct, |p, v| p.tm_mid_pct = v, 60, 160),
    ("TmLatePct", |p| p.tm_late_pct, |p, v| p.tm_late_pct = v, 60, 160),
    ("TmSimplePct", |p| p.tm_simple_pct, |p, v| p.tm_simple_pct = v, 30, 140),
    ("TmAheadPct", |p| p.tm_ahead_pct, |p, v| p.tm_ahead_pct = v, 100, 180),
    ("TmBehindPct", |p| p.tm_behind_pct, |p, v| p.tm_behind_pct = v, 40, 100),
    ("TmFloorMs", |p| p.tm_floor_ms, |p, v| p.tm_floor_ms = v, 1, 200),
];

impl Params {
    pub fn set(&mut self, name: &str, value: i32) -> bool {
        for (n, _, put, lo, hi) in PARAM_SPECS {
            if n.eq_ignore_ascii_case(name) {
                put(self, value.clamp(*lo, *hi));
                return true;
            }
        }
        false
    }
}

/// Techniques that are switched off until they have earned their place.
///
/// Every one is off by default, so the engine out of the box searches with the
/// smaller, settled set of ideas. What each is worth then has an answer
/// rather than an opinion: turn exactly one on, play a match, read the number.
/// A feature that cannot be switched off is a feature nobody ever measured.
///
/// Measured at 8+0.08 against this engine with everything off, roughly two
/// hundred games each, so the interval on any one of them is about seventy Elo
/// wide and only the extremes below mean anything:
///
///   NmpCutNode  +24    RfpDamp  +20    Razoring  +14
///   Rule50Fade   -4    TtCutCredit -4   CheckExt   -6
///   CaptureHist  -8    Probcut -12     CorrHist  -14
///   LmpImproving -24   TtPvLmr -25     QsFutility -130
///
/// The last one is the only result that needed no second sample: three wins
/// against forty-eight losses. It was not the idea, it was a line of it, and
/// the same turned out to be true of the correction history -- both are
/// commented where they are implemented. That is the lesson these numbers
/// actually carry: a technique that is worth Elo in every strong engine and
/// negative here is a bug report, not a measurement.
#[derive(Clone, Copy)]
pub struct Features {
    /// Learn how wrong the static evaluation usually is for a pawn structure,
    /// and feed it back.
    pub corr_hist: bool,
    /// Ask quiescence directly when a node is far enough behind.
    pub razoring: bool,
    /// Fade the evaluation towards a draw as the fifty move counter runs out.
    pub rule50_fade: bool,
    /// Reduce less at a node that once earned a full window.
    pub ttpv_lmr: bool,
    /// Search a ply shallower when the table has no move to try first.
    pub iir: bool,
    /// Cut the late move count in half when things are not improving.
    pub lmp_improving: bool,
    /// In quiescence, skip a capture that cannot come near alpha even if it
    /// wins everything it takes.
    pub qs_futility: bool,
    /// Spend less when most of the tree went to the move that won anyway.
    pub tm_node_effort: bool,
    /// Trust a stored lower bound far enough above beta without re-searching.
    pub probcut: bool,
    /// Only try the null move where the node is expected to fail high.
    pub nmp_cut_node: bool,
    /// On a reverse futility cutoff, return part of the way to the estimate
    /// rather than all of it.
    pub rfp_damp: bool,
    /// Extend a move that gives check.
    pub check_ext: bool,
    /// Credit the stored move when the table itself produces the cutoff.
    pub tt_cut_credit: bool,
    /// Remember which captures worked, not only what they take.
    pub capture_hist: bool,
    /// Use the transcribed search instead of this one.
    ///
    /// Not a feature but an experiment: same board, same network, same table,
    /// same clock, and the search swapped whole. Whichever way it comes out
    /// says where the difference lives.
    pub ref_search: bool,
    /// Use the integer-logarithm reduction formula instead of the table.
    ///
    /// The last place where a number in this search is an invention rather
    /// than something measured. Everything else came across with its value;
    /// the reduction shape did not, because it was written before the
    /// reference was read closely, and it is the highest-leverage part of a
    /// search to be guessing at.
    pub log_lmr: bool,

    /// Reduce harder at a node that is expected to fail high.
    ///
    /// Unlike the rest, this is ON by default: it belongs in the baseline
    /// rather than on top of it, and the switch is here to measure it, not to
    /// leave it out.
    pub cut_node_lmr: bool,

    /// Skip a quiet move the history has consistently disliked.
    pub history_prune: bool,
    /// Spend longer when the score is falling, less when the best move has
    /// stopped changing.
    pub tm_stability: bool,

    /// Reduce late captures too, not only late quiet moves.
    pub lmr_captures: bool,
}

impl Default for Features {
    fn default() -> Self {
        Features {
            corr_hist: false,
            razoring: false,
            rule50_fade: false,
            ttpv_lmr: false,
            iir: true,
            lmp_improving: false,
            qs_futility: false,
            tm_node_effort: true,
            probcut: false,
            nmp_cut_node: false,
            rfp_damp: false,
            check_ext: false,
            tt_cut_credit: false,
            capture_hist: false,
            ref_search: false,
            log_lmr: false,
            cut_node_lmr: true,
            history_prune: true,
            tm_stability: true,
            lmr_captures: true,
        }
    }
}

impl Features {
    /// UCI option name to field, for `setoption`.
    pub fn set(&mut self, name: &str, on: bool) -> bool {
        match name {
            "corrhist" => self.corr_hist = on,
            "razoring" => self.razoring = on,
            "rule50fade" => self.rule50_fade = on,
            "ttpvlmr" => self.ttpv_lmr = on,
            "iir" => self.iir = on,
            "lmpimproving" => self.lmp_improving = on,
            "qsfutility" => self.qs_futility = on,
            "tmnodeeffort" => self.tm_node_effort = on,
            "probcut" => self.probcut = on,
            "nmpcutnode" => self.nmp_cut_node = on,
            "rfpdamp" => self.rfp_damp = on,
            "checkext" => self.check_ext = on,
            "ttcutcredit" => self.tt_cut_credit = on,
            "capturehist" => self.capture_hist = on,
            "refsearch" => self.ref_search = on,
            "loglmr" => self.log_lmr = on,
            "cutnodelmr" => self.cut_node_lmr = on,
            "historyprune" => self.history_prune = on,
            "tmstability" => self.tm_stability = on,
            "lmrcaptures" => self.lmr_captures = on,
            _ => return false,
        }
        true
    }

    /// The ones outside the settled set. All default off.
    pub const EXTRA: [&'static str; 14] = [
        "CorrHist",
        "Razoring",
        "Rule50Fade",
        "TtPvLmr",
        "LmpImproving",
        "QsFutility",
        "Probcut",
        "NmpCutNode",
        "RfpDamp",
        "CheckExt",
        "TtCutCredit",
        "CaptureHist",
        "RefSearch",
        "LogLmr",
    ];

    /// The ones it does have, so they are in the baseline. All default on.
    pub const BASELINE: [&'static str; 6] =
        ["CutNodeLmr", "HistoryPrune", "LmrCaptures", "TmStability", "IIR", "TmNodeEffort"];
}

#[derive(Default, Clone)]
pub struct Limits {
    pub wtime: Option<u64>,
    pub btime: Option<u64>,
    pub winc: u64,
    pub binc: u64,
    pub movestogo: Option<u64>,
    pub movetime: Option<u64>,
    pub depth: Option<u32>,
    pub nodes: Option<u64>,
    pub infinite: bool,
}

pub struct Searcher {
    pub tt: TranspositionTable,
    pub atk: Attacks,
    pub stop: Arc<AtomicBool>,
    /// Milliseconds held back from every allocation to cover the time between
    /// deciding on a move and the move being seen by whoever is counting.
    ///
    /// Not a nicety. Measured over sixty games at 5+0.05 without it, thirty-one
    /// were lost on the clock; with it, none of twenty-eight were. The default
    /// is deliberately generous, because the cost of being wrong is asymmetric:
    /// too large loses a little strength, too small loses whole games.
    pub move_overhead: u64,
    pub features: Features,
    pub params: Params,
    lmr: [[i32; 64]; 64],

    pub(crate) nodes: u64,
    start: Instant,
    soft: Duration,
    hard: Duration,
    pub(crate) stopped: bool,

    killers: [[Option<Move>; NUM_KILLERS]; MAX_PLY],
    history: [[[i32; 64]; 64]; 2],
    /// `[side][pawn structure]`.
    /// `[kind][side][key]`.
    corr: Vec<Vec<[i32; CORR_SIZE]>>,
    /// What was played at each ply, as (piece, destination). Continuation
    /// history is indexed by this: a move is good or bad largely in reply to
    /// something, and a table that ignores what came before cannot say which.
    played: [Option<(usize, usize)>; MAX_PLY],
    /// `[slot][prev piece * 64 + prev to][piece][to]`, one table per distance
    /// back.
    conthist: Vec<Vec<[[i32; 64]; 6]>>,
    /// `[moving piece][destination][captured piece]`.
    ///
    /// What a capture takes is known before it is played; whether taking it
    /// works is not. Most valuable victim answers the first question and calls
    /// it the second -- so a queen recapture that always loses to a pin keeps
    /// being tried first, forever, because the queen is still the biggest piece
    /// on the square.
    capthist: Vec<[[i32; 6]; 64]>,
    /// Zobrist keys along the path plus the game so far, for repetition.
    pub(crate) keys: Vec<u64>,
    /// How many of `keys` are game history rather than search path.
    root_keys: usize,

    pub(crate) pv: [[Option<Move>; MAX_PLY]; MAX_PLY],
    pub(crate) pv_len: [usize; MAX_PLY],
    /// The static score at each ply, so a node can ask whether things have
    /// been getting better for the side to move. A position that is improving
    /// deserves a tighter margin than one that is falling apart, because the
    /// reason to prune is confidence and there is less of it on the way down.
    pub(crate) eval_stack: [i32; MAX_PLY],
    /// A move this ply is pretending does not exist, while it finds out
    /// whether that move was the only one holding the position up.
    pub(crate) excluded: [Option<Move>; MAX_PLY],
    /// The move played at each ply, for the transcribed search's continuation
    /// tables, which index by a move rather than by a piece and square.
    pub(crate) played_moves: Vec<Option<Move>>,
    /// The transcribed search keeps its own tables: same shapes and ceilings as
    /// the transcribed search, which are not the shapes the other one uses.
    pub(crate) ref_hist: crate::refsearch::Histories,
    /// The last few moves of the game, for the plies above the root.
    pub(crate) pre_moves: [Option<Move>; PRE_MOVES],
    /// How many nodes each root move cost this iteration. A move that took
    /// most of the tree and still came out best was not a close call, and time
    /// management can read that.
    root_effort: Vec<(Move, u64)>,
    /// Every root move with what this iteration thought of it.
    ///
    /// Keeping only the best one leaves nothing to fall back on when the best
    /// one repeats: there is no second opinion, only a move and no reason to
    /// prefer anything else. With the whole list scored, a repetition can be
    /// declined in favour of something that is nearly as good, and how much
    /// worse we are willing to accept is a number rather than an accident.
    root_scores: Vec<(Move, i32)>,
    /// Which plies got there by passing. Two passes in a row prove nothing:
    /// the side to move has effectively been given a free tempo twice, and the
    /// position being searched is not one that can occur.
    pub(crate) null_at: [bool; MAX_PLY],
}

/// The score of a position from the side to move's point of view.
fn evaluate(board: &Board, fade: bool) -> i32 {
    let net = match nnue::net() {
        Some(n) => n,
        None => return 0,
    };
    let acc = match board.acc.as_ref() {
        Some(a) => a,
        None => return 0,
    };
    let raw = acc.eval(net, board.side, board.occ_all.count_ones());

    // Fade towards a draw as the fifty move counter runs out. A network trained
    // on positions is confident about a position that is about to stop counting
    // for anything, and without this the search happily walks into a draw it
    // thinks it is winning.
    if fade {
        raw * (200 - board.halfmove.min(100) as i32) / 200
    } else {
        raw
    }
}

/// A piece value in the units the network speaks.
///
/// The table that travels with the board is in ordinary centipawns, where a
/// pawn is 100. This network answers on a scale with two units to the
/// centipawn, so anything that compares a piece against an evaluation has to
/// convert. Not converting made the quiescence margin twice as harsh as
/// intended -- the same class of mistake that made history pruning never fire
/// at all, in the other direction. Static exchange is exempt: it is centipawns
/// end to end, input and output, so a threshold handed to it belongs in
/// centipawns too.
#[inline]
fn value_in_eval_units(pt: PieceType) -> i32 {
    pt.value() * 2
}

/// Does this side have anything but pawns and a king?
///
/// The question null move pruning asks: with only pawns left, having to move is
/// often a disadvantage, so a side that passes and still looks fine proves
/// nothing about a side that has to play.
pub(crate) fn has_pieces_pub(board: &Board, side: Color) -> bool {
    has_pieces(board, side)
}

fn has_pieces(board: &Board, side: Color) -> bool {
    let p = &board.pieces[side.idx()];
    p[PieceType::Knight.idx()]
        | p[PieceType::Bishop.idx()]
        | p[PieceType::Rook.idx()]
        | p[PieceType::Queen.idx()]
        != 0
}

/// The evaluation, exposed for the UCI `eval` command.
pub fn debug_eval(board: &Board, fade: bool) -> i32 {
    evaluate(board, fade)
}

fn mate_score(ply: usize) -> i32 {
    -MATE + ply as i32
}

pub fn is_mate(score: i32) -> bool {
    score.abs() >= MATE_IN_MAX
}

/// Moving a mate score in and out of the table: stored relative to the node it
/// was found at, used relative to the root. Without this a mate found deep in
/// one branch is reported as being that many moves away from wherever the entry
/// is read next.
pub(crate) fn score_to_tt(score: i32, ply: usize) -> i32 {
    if score >= MATE_IN_MAX {
        score + ply as i32
    } else if score <= -MATE_IN_MAX {
        score - ply as i32
    } else {
        score
    }
}

pub(crate) fn score_from_tt(score: i32, ply: usize) -> i32 {
    if score >= MATE_IN_MAX {
        score - ply as i32
    } else if score <= -MATE_IN_MAX {
        score + ply as i32
    } else {
        score
    }
}

impl Searcher {
    pub fn new(hash_mb: usize, stop: Arc<AtomicBool>) -> Self {
        Searcher {
            tt: TranspositionTable::new(hash_mb),
            atk: Attacks::new(),
            stop,
            move_overhead: 30,
            features: Features::default(),
            params: Params::default(),
            lmr: build_lmr_table(Params::default().lmr_base, Params::default().lmr_div),
            nodes: 0,
            start: Instant::now(),
            soft: Duration::from_secs(0),
            hard: Duration::from_secs(0),
            stopped: false,
            killers: [[None; NUM_KILLERS]; MAX_PLY],
            history: [[[0; 64]; 64]; 2],
            corr: vec![vec![[0; CORR_SIZE]; 2]; CORR_KINDS],
            played: [None; MAX_PLY],
            conthist: vec![vec![[[0; 64]; 6]; 6 * 64]; CONT_SLOTS],
            capthist: vec![[[0; 6]; 64]; 6],
            keys: Vec::with_capacity(1024),
            root_keys: 0,
            pv: [[None; MAX_PLY]; MAX_PLY],
            pv_len: [0; MAX_PLY],
            eval_stack: [0; MAX_PLY],
            excluded: [None; MAX_PLY],
            played_moves: vec![None; MAX_PLY],
            ref_hist: crate::refsearch::Histories::new(MAX_PLY),
            pre_moves: [None; PRE_MOVES],
            root_effort: Vec::with_capacity(256),
            root_scores: Vec::with_capacity(256),
            null_at: [false; MAX_PLY],
        }
    }

    /// Call after changing any parameter, so anything derived from one is
    /// rebuilt rather than left describing the old value.
    pub fn params_changed(&mut self) {
        self.lmr = build_lmr_table(self.params.lmr_base, self.params.lmr_div);
    }

    pub fn set_game_history(&mut self, keys: Vec<u64>) {
        self.keys = keys;
        self.root_keys = self.keys.len();
    }

    /// The moves actually played before this search started.
    ///
    /// Continuation history asks "what is a good reply to what just happened",
    /// and at the top of the tree what just happened is in the GAME, not in the
    /// search. Without this the tables are empty for the first plies of every
    /// search -- which is where most of the tree is -- and the ordering there
    /// runs on the butterfly table alone.
    ///
    /// Measured against the program this search was transcribed from: its
    /// average history score per reduced move was -7757 against ours at -1219,
    /// six times less opinionated, and this was why.
    ///
    /// Stored in the slots below zero, so that `ply - 1` at the root reaches
    /// the last move of the game rather than nothing.
    pub fn set_game_moves(&mut self, moves: &[Move]) {
        self.pre_moves = [None; PRE_MOVES];
        for (i, mv) in moves.iter().rev().take(PRE_MOVES).enumerate() {
            self.pre_moves[i] = Some(*mv);
        }
    }

    /// The move `back` plies before `ply`, reaching into the game when the
    /// search runs out.
    #[inline]
    pub(crate) fn move_back(&self, ply: usize, back: usize) -> Option<Move> {
        if ply >= back {
            self.played_moves[ply - back]
        } else {
            self.pre_moves.get(back - ply - 1).copied().flatten()
        }
    }

    pub fn clear(&mut self) {
        self.tt.clear();
        self.killers = [[None; NUM_KILLERS]; MAX_PLY];
        self.history = [[[0; 64]; 64]; 2];
        self.corr = vec![vec![[0; CORR_SIZE]; 2]; CORR_KINDS];
        self.ref_hist.clear();
        self.conthist = vec![vec![[[0; 64]; 6]; 6 * 64]; CONT_SLOTS];
        self.capthist = vec![[[0; 6]; 64]; 6];
    }

    /// What the last move changed about the position key.
    ///
    /// The key stack holds every position back to the start of the game, with
    /// the current one on top, so the difference between the top two is the
    /// move that was just made -- from, to, what it took and whose it was, all
    /// in one number. Zero when there is nothing above.
    #[inline]
    fn hash_delta(&self) -> u64 {
        let n = self.keys.len();
        if n >= 2 {
            self.keys[n - 1] ^ self.keys[n - 2]
        } else {
            0
        }
    }

    /// The static score, adjusted by what the search has been saying about
    /// positions with this pawn structure.
    #[inline]
    fn corrected(&self, board: &Board, raw: i32, ply: usize) -> i32 {
        let last = if ply > 0 { self.played[ply - 1] } else { None };
        let idx = corr_indices(board, last, self.hash_delta());
        let side = board.side.idx();
        let mut total = 0i32;
        for k in 0..CORR_KINDS {
            total += self.corr[k][side][idx[k]] * CORR_WEIGHT[k];
        }
        let c = total / CORR_DIVISOR;
        (raw + c).clamp(-MATE_IN_MAX + 1, MATE_IN_MAX - 1)
    }

    /// Learn from the difference, weighted by how deep the search that found
    /// it went.
    /// Move each reading towards what the search actually said.
    ///
    /// The amount is the miss scaled by how deep the search that found it went,
    /// capped, and then applied so that an entry approaches its ceiling instead
    /// of slamming into it. The previous version multiplied the miss by 256
    /// before capping, so any miss above thirty two units saturated the target
    /// and every update looked the same size.
    #[inline]
    fn learn_correction(&mut self, board: &Board, diff: i32, depth: i32, ply: usize) {
        let last = if ply > 0 { self.played[ply - 1] } else { None };
        let idx = corr_indices(board, last, self.hash_delta());
        let side = board.side.idx();
        let bonus = (diff * depth / 8).clamp(-CORR_MAX_UPDATE, CORR_MAX_UPDATE);
        for k in 0..CORR_KINDS {
            let e = &mut self.corr[k][side][idx[k]];
            *e += bonus - *e * bonus.abs() / CORR_MAX;
            *e = (*e).clamp(-CORR_MAX, CORR_MAX);
        }
    }

    /// Which continuation table each slot points at, this ply.
    ///
    /// One, two and four plies back. The first is the move being replied to and
    /// carries twice the weight of the others: what makes a quiet move good is
    /// most often what the opponent just did, and only after that what we were
    /// doing before.
    #[inline]
    fn cont_slots(&self, ply: usize) -> [Option<usize>; CONT_SLOTS] {
        let mut out = [None; CONT_SLOTS];
        for (k, back) in CONT_BACK.iter().enumerate() {
            if ply >= *back {
                if let Some((pc, to)) = self.played[ply - back] {
                    out[k] = Some(pc * 64 + to);
                }
            } else if let Some(mv) = self.pre_moves.get(back - ply - 1).copied().flatten() {
                // Above the root: the piece is unknown here, so the move itself
                // stands in for it. Consistent within the table, which is all
                // an index has to be.
                out[k] = Some((mv.from as usize % 6) * 64 + mv.to as usize);
            }
        }
        out
    }

    /// Decide how long this move may take.
    ///
    /// Two limits, because they answer different questions. `soft` is checked
    /// only between iterations: passing it means there is not enough left to
    /// make another depth worthwhile, and the move we have is the move we play.
    /// `hard` is checked inside the search and is a wall -- crossing it means
    /// abandoning the iteration in progress and using the last completed one.
    ///
    /// Everything is taken from the clock AFTER the overhead is removed, and
    /// `hard` is capped so that even the wall cannot spend what we do not have.
    fn allocate(&mut self, limits: &Limits, board: &Board) {
        let side = board.side;
        self.start = Instant::now();

        if limits.infinite || limits.depth.is_some() || limits.nodes.is_some() {
            self.soft = Duration::from_secs(86_400);
            self.hard = Duration::from_secs(86_400);
            return;
        }

        if let Some(mt) = limits.movetime {
            let usable = mt.saturating_sub(self.move_overhead).max(1);
            self.soft = Duration::from_millis(usable);
            self.hard = Duration::from_millis(usable);
            return;
        }

        let (time, inc, opp_time) = match side {
            Color::White => (limits.wtime, limits.winc, limits.btime),
            Color::Black => (limits.btime, limits.binc, limits.wtime),
        };
        let time = match time {
            Some(t) => t,
            None => {
                self.soft = Duration::from_secs(86_400);
                self.hard = Duration::from_secs(86_400);
                return;
            }
        };

        // What is actually ours to spend. `saturating_sub` and the floor of one
        // millisecond matter: in time trouble the clock can be below the
        // overhead, and an allocation of zero would still have to make a move,
        // just without having thought about it.
        let usable = time.saturating_sub(self.move_overhead).max(1);

        // How many more moves to plan for. With a real count given, use it.
        // Without one, twenty-five is a deliberately pessimistic guess: games
        // that end sooner leave time unspent, which costs a little strength,
        // while games that run longer flag, which costs the whole point.
        let mtg = limits.movestogo.unwrap_or(self.params.tm_mtg as u64).max(1);

        // The increment is income, so most of it can be spent every move
        // without the clock moving. Not all of it: the part held back is what
        // slowly rebuilds a buffer over a long game.
        let mut base = usable / mtg + inc * self.params.tm_inc_pct as u64 / 100;

        // What the position is worth spending on, by where the game is.
        //
        // The opening is played rather than calculated: the answer is either
        // known or is one of several equally playable moves, and a quarter of
        // a clock can disappear before the game has started. A simplified
        // position has less left to find. The middle is where thinking pays,
        // so that is where the money goes.
        let ply = (board.fullmove as i32 - 1) * 2 + (side == Color::Black) as i32;
        let pieces = board.occ_all.count_ones() as i32;
        let phase = if ply < 12 {
            self.params.tm_open_pct
        } else if ply < 24 {
            self.params.tm_early_pct
        } else if ply < 45 {
            self.params.tm_mid_pct
        } else if ply < 65 {
            self.params.tm_late_pct
        } else if pieces <= 10 {
            self.params.tm_simple_pct
        } else {
            100
        };
        base = base * phase as u64 / 100;

        // And by the other clock, which is half of the game.
        //
        // A comfortable lead on the clock is an asset to spend; being behind on
        // it is a reason not to, because the opponent can simply keep playing
        // and let the difference do the work. Overall health rather than the
        // pace of any one move.
        if let Some(opp) = opp_time.filter(|t| *t > 0) {
            let ratio = time * 10 / opp;
            let pressure = if ratio > 15 {
                self.params.tm_ahead_pct
            } else if ratio < 7 {
                self.params.tm_behind_pct
            } else {
                100
            };
            base = base * pressure as u64 / 100;
        }

        // Two ceilings on the wall, and the second is the one that matters.
        //
        // Twice the plan lets a critical move think a little longer. Two
        // fifths of what is left stops that from becoming a way to spend the
        // clock.
        let hard = (base * self.params.tm_hard_mult as u64)
            .min(usable * self.params.tm_hard_pct as u64 / 100);
        // A floor under both, so that a clock this low still buys a move that
        // was looked at rather than one that was guessed. It cannot make the
        // engine spend what it does not have: the floor is itself capped by
        // what is actually left.
        let floor = (self.params.tm_floor_ms as u64).min(usable);
        let soft = base.min(hard).max(floor);
        let hard = hard.max(soft);

        self.soft = Duration::from_millis(soft.max(1));
        self.hard = Duration::from_millis(hard.max(1));
    }

    #[inline]
    pub(crate) fn out_of_time(&mut self) -> bool {
        if self.stopped {
            return true;
        }
        // Checking the clock is a syscall, so it is not done every node. But
        // the interval is a floor on how long the search can run without
        // noticing, and it has to be small enough to fit inside the smallest
        // allocation we will ever make. At 2048 it was not: a first iteration
        // in a middle game position is under two thousand nodes, so in real
        // time trouble the whole of it ran without the clock being read once,
        // and the engine sailed past a forty millisecond wall by taking a
        // hundred. At 512 the blind spot is a couple of milliseconds.
        if self.nodes & 511 == 0
            && (self.start.elapsed() >= self.hard || self.stop.load(Ordering::Relaxed))
        {
            self.stopped = true;
        }
        self.stopped
    }

    /// Would playing this move land straight back on a position already seen?
    #[inline]
    fn repeats_at_once(&self, board: &Board, mv: Move) -> bool {
        let mut b = board.clone();
        let undo = b.make_move(&mv);
        let h = b.hash;
        b.unmake_move(&mv, &undo);
        self.keys.iter().rev().take(64).any(|k| *k == h)
    }

    /// Has this position already occurred? One earlier occurrence is enough to
    /// treat it as drawn inside the search -- waiting for the third makes the
    /// search miss the repetition it is about to walk into.
    fn is_repetition(&self, board: &Board) -> bool {
        let back = (board.halfmove as usize).min(self.keys.len());
        // Walking back from the top, offset zero is this position, so the
        // positions with the same side to move are the EVEN offsets: two plies
        // ago, four, six. Skipping one instead of two sampled the odd ones,
        // every one of which has the other side to move, and the side to move
        // is part of the key -- so the test could not match and never did.
        //
        // Measured before the fix: three hundred and thirty four thousand
        // calls across three positions, zero detections. The engine had no
        // repetition detection at all. It would announce eight pawns of
        // advantage while playing the move that made a threefold, because for
        // the search that line was not a draw.
        self.keys
            .iter()
            .rev()
            .take(back)
            .skip(2)
            .step_by(2)
            .any(|k| *k == board.hash)
    }

    pub(crate) fn is_draw(&self, board: &Board) -> bool {
        board.halfmove >= 100 || self.is_repetition(board)
    }

    /// What a drawn position is worth, seen from the side to move at `ply`.
    ///
    /// The value belongs to the root, not to whoever happens to be on the move.
    /// Returning `-contempt` at every node makes both sides reluctant, and in a
    /// negamax that is not a preference, it is a contradiction: the root reads
    /// its own draws as costing `contempt` and the opponent's draws as gaining
    /// it, so the same drawn position is worth two different things depending
    /// on the parity of the ply it was found at.
    ///
    /// Measured with the old version, at a contempt of twenty over 176 games:
    /// fifty wins, forty-two draws and eighty-four losses, which is 68 Elo
    /// worse and the whole interval below zero. The engine was declining draws
    /// in positions it was losing, which is where it wanted them.
    ///
    /// Alternating with the ply is what makes the root read a draw as costing
    /// `contempt` everywhere.
    #[inline]
    pub(crate) fn draw_score(&self, ply: usize) -> i32 {
        if ply & 1 == 1 {
            self.params.contempt
        } else {
            -self.params.contempt
        }
    }

    pub fn go(&mut self, board: &mut Board, limits: &Limits, info: bool) -> Option<Move> {
        self.allocate(limits, board);
        self.nodes = 0;
        self.stopped = false;
        self.stop.store(false, Ordering::Relaxed);
        self.tt.increase_gen();
        self.keys.truncate(self.root_keys);

        let max_depth = limits.depth.unwrap_or(MAX_PLY as u32 - 2).min(MAX_PLY as u32 - 2);

        let mut best: Option<Move> = None;
        let mut best_score = 0;

        // Time management state that only makes sense across iterations.
        let mut last_best: Option<Move> = None;
        let mut best_move_changes = 0i32;
        let mut iters_since_change = 0i32;
        let mut average_score = 0i32;
        let base_soft = self.soft;

        for depth in 1..=max_depth {
            let iter_start = self.start.elapsed();
            self.root_effort.clear();
            self.root_scores.clear();
            let score = self.aspiration(board, depth as i32, best_score);

            // An aborted iteration has searched only part of the move list, so
            // its best move is not the best move -- it is whatever happened to
            // come first. Keep the previous depth.
            if self.stopped && depth > 1 {
                break;
            }

            best_score = score;
            if self.pv_len[0] > 0 {
                best = self.pv[0][0];
            }

            // Decline a repetition when something else is nearly as good.
            //
            // Contempt already makes a repeated position score badly INSIDE
            // the search, but only where the search reaches one. A move that
            // repeats immediately -- straight back into a position already on
            // the board twice -- is decided here, where the whole root list is
            // in hand and the alternatives have scores.
            if self.params.contempt > 0 {
                if let Some(b) = best {
                    if self.repeats_at_once(board, b) {
                        let margin = self.params.contempt;
                        let alt = self
                            .root_scores
                            .iter()
                            .filter(|(m, _)| *m != b && !self.repeats_at_once(board, *m))
                            .max_by_key(|(_, sc)| *sc);
                        if let Some((m, sc)) = alt {
                            if *sc >= best_score - margin {
                                best = Some(*m);
                            }
                        }
                    }
                }
            }

            if info {
                self.print_info(depth, score);
            }

            if limits.nodes.map_or(false, |n| self.nodes >= n) {
                break;
            }

            // What the position is telling us about how long to keep going.
            //
            // Three signals, and they answer different questions. A score that
            // is falling means the move we have is worse than we thought and
            // the alternatives deserve another look. A best move that has
            // stopped changing means the answer has settled and more time buys
            // nothing. And a move that took most of the tree to itself and
            // still came out on top was never a close call.
            if depth == 1 {
                average_score = score;
            } else {
                average_score = (score + 9 * average_score) / 10;
            }
            if best != last_best {
                last_best = best;
                iters_since_change = 0;
                best_move_changes += 1;
            } else {
                iters_since_change += 1;
            }

            let spent: u64 = self.root_effort.iter().map(|(_, n)| *n).sum();
            let on_best = best
                .and_then(|b| self.root_effort.iter().find(|(m, _)| *m == b))
                .map(|(_, n)| *n)
                .unwrap_or(0);
            // With nothing measured yet, claim the middle rather than either
            // end: an unknown share should neither buy time nor spend it.
            let effort_frac = if spent > 0 && self.features.tm_node_effort {
                on_best as f64 / spent as f64
            } else {
                0.4
            };
            let drop = if self.features.tm_stability {
                (average_score - score).max(0)
            } else {
                0
            };
            let settle = if self.features.tm_stability {
                iters_since_change as u32
            } else {
                0
            };
            let changes = if self.features.tm_stability {
                best_move_changes.max(0) as u32
            } else {
                0
            };
            let factor = time_scale(effort_frac, settle, drop, changes);

            // The scaling moves the plan, never the wall. Whatever the position
            // says, a move cannot spend more than the clock allows.
            let soft = base_soft.mul_f64(factor).min(self.hard);
            self.soft = soft;

            // Is there room for another iteration, not is there room for the
            // one just finished.
            //
            // Each depth costs roughly twice the one before, so stopping only
            // once the plan is already spent means routinely starting an
            // iteration that cannot fit and letting the wall end it. The spend
            // then settles at about twice the plan, which is enough to break
            // even against the increment: measured over a fifty-nine move game
            // at 8+0.08, the engine used 12.69 seconds of a 12.72 second
            // budget and flagged. Nothing looked wrong move by move -- the
            // longest was 0.72 seconds -- because nothing was wrong move by
            // move.
            //
            // Predicting the next one instead leaves the margin the increment
            // is supposed to build.
            let elapsed = self.start.elapsed();
            let last = elapsed.saturating_sub(iter_start);
            if elapsed + last * 2 >= self.soft {
                break;
            }
        }

        // Never return nothing: if even depth one was cut short, play the first
        // legal move rather than forfeit.
        best.or_else(|| generate_legal(board, &self.atk).into_iter().next())
    }

    fn print_info(&self, depth: u32, score: i32) {
        let ms = self.start.elapsed().as_millis().max(1) as u64;
        let nps = self.nodes * 1000 / ms;
        let score_str = if is_mate(score) {
            let plies = MATE - score.abs();
            let moves = (plies + 1) / 2;
            format!("mate {}", if score > 0 { moves } else { -moves })
        } else {
            // Two internal units to the centipawn, from the training
            // quantisation. The win/draw/loss figures below are NOT converted:
            // that model was fitted against the internal units and its offset
            // and scaling are in them, so handing it centipawns would quietly
            // halve every probability it reports.
            format!("cp {}", score / 2)
        };
        let (w, d, l) = nnue::wdl(score);
        let mut pv = String::new();
        for i in 0..self.pv_len[0] {
            if let Some(m) = self.pv[0][i] {
                pv.push(' ');
                pv.push_str(&m.to_uci());
            }
        }
        println!(
            "info depth {} score {} wdl {} {} {} nodes {} nps {} time {} pv{}",
            depth, score_str, w, d, l, self.nodes, nps, ms, pv
        );
    }

    /// Search the root with a window around the last score, widening on a
    /// failure rather than starting wide every time.
    fn aspiration(&mut self, board: &mut Board, depth: i32, prev: i32) -> i32 {
        // Wider at low depth, where the previous score is a poor guide, and
        // narrowing as it becomes a good one.
        let mut delta = 5 + self.params.asp_delta * 8 / depth.max(1);
        let (mut alpha, mut beta) = if depth <= self.params.asp_depth || is_mate(prev) {
            (-INF, INF)
        } else {
            (prev - delta, prev + delta)
        };

        loop {
            // Once a bound is this far from level, the window has stopped being
            // a guess worth narrowing and has become an obstacle: a position
            // that decided is going to keep failing in the same direction, and
            // each failure costs a whole re-search to widen by a step.
            if alpha < -1000 {
                alpha = -INF;
            }
            if beta > 1000 {
                beta = INF;
            }

            let score = if self.features.ref_search {
                self.negamax_ref(board, depth, alpha, beta, 0, true, false)
            } else {
                self.negamax(board, depth, alpha, beta, 0, true, false)
            };
            if self.stopped {
                return score;
            }
            if score <= alpha {
                // Failing low means the position is worse than believed, and
                // the upper bound has to move with the lower one or the next
                // attempt fails low again at once.
                beta = (alpha + beta) / 2;
                alpha = (score - delta).max(-INF);
            } else if score >= beta {
                beta = (score + delta).min(INF);
            } else {
                return score;
            }
            delta += delta / 2;
        }
    }

    /// `cut_node` says this node is expected to fail high.
    ///
    /// It is not a guess made here: it is handed down. The first child of a
    /// principal variation node is another one; every later child of a
    /// principal variation node is expected to fail high; the children of a
    /// node expected to fail high are expected to fail low, and the other way
    /// round. Knowing which kind of node you are in is worth something, because
    /// a node that is expected to fail high will do it on one of the first
    /// moves or not at all, so the late ones there can be reduced harder than
    /// the same moves somewhere else.
    fn negamax(
        &mut self,
        board: &mut Board,
        mut depth: i32,
        mut alpha: i32,
        beta: i32,
        ply: usize,
        pv_node: bool,
        cut_node: bool,
    ) -> i32 {
        self.pv_len[ply] = 0;

        if depth <= 0 {
            return self.quiescence(board, alpha, beta, ply);
        }

        self.nodes += 1;
        if self.out_of_time() {
            return 0;
        }

        let root = ply == 0;
        let in_check = board.in_check(board.side, &self.atk);

        if !root {
            if self.is_draw(board) {
                return self.draw_score(ply);
            }
            if ply >= MAX_PLY - 1 {
                return evaluate(board, self.features.rule50_fade);
            }
            // Mate distance pruning: no line from here can beat a mate already
            // found closer to the root.
            let a = alpha.max(mate_score(ply));
            let b = beta.min(-mate_score(ply + 1));
            if a >= b {
                return a;
            }
            alpha = a;
        }

        // A node searching with a move excluded is asking a different question
        // from the one the table answered, so it must not take the answer --
        // nor leave its own answer behind for a node that is asking the
        // ordinary question.
        let excluded = self.excluded[ply];
        let entry = self.tt.probe(board.hash);
        let mut tt_move = None;
        let mut tt_pv = pv_node;
        if let Some(e) = entry {
            tt_move = e.best;
            tt_pv |= e.pv;
            if excluded.is_none() && !pv_node && e.depth >= depth && e.has_bound() {
                let s = score_from_tt(e.score, ply);
                let usable = match e.bound {
                    Bound::Exact => true,
                    Bound::Lower => s >= beta,
                    Bound::Upper => s <= alpha,
                    Bound::NoBound => false,
                };
                // Not near the fifty move wall: there the same position is
                // worth different things depending on how much counter is left,
                // and the table does not know which one it stored.
                if usable && board.halfmove < 90 {
                    // Credit the stored move on the way out. It just caused a
                    // cutoff, which is the same evidence a searched move would
                    // have produced, and returning without recording it lets
                    // the tables go cold in exactly the positions that come
                    // back most often -- the ones the table keeps answering.
                    if self.features.tt_cut_credit && s >= beta {
                        if let Some(m) = tt_move {
                            // As close to a legality test as is affordable here:
                            // one of ours on the origin square, and nothing of
                            // ours on the destination. It does not prove the
                            // move is legal, which is why this is off.
                            let ours = board
                                .piece_at(m.from)
                                .is_some_and(|(_, c)| c == board.side);
                            let free = board
                                .piece_at(m.to)
                                .is_none_or(|(_, c)| c != board.side);
                            if ours && free && !m.is_capture() && m.promotion.is_none() {
                                let side = board.side.idx();
                                let slots = self.cont_slots(ply);
                                self.credit(board, m, side, &slots, hist_bonus(depth));
                                if !self.killers[ply].iter().any(|k| *k == Some(m)) {
                                    for j in (1..NUM_KILLERS).rev() {
                                        self.killers[ply][j] = self.killers[ply][j - 1];
                                    }
                                    self.killers[ply][0] = Some(m);
                                }
                            }
                        }
                    }
                    return s;
                }
            }
        }

        // The raw number is kept separately, because it is what goes back into
        // the table at the bottom of this node.
        //
        // Storing the corrected value there instead is a quiet disaster: the
        // next visit reads it, applies the correction a second time, stores
        // that, and the error compounds every time the position is reached.
        // Nothing about it looks wrong from outside -- the evaluation stays
        // plausible while drifting.
        let raw_static_eval = if in_check {
            TT_EVAL_NONE as i32
        } else {
            match entry {
                Some(e) if e.static_eval != TT_EVAL_NONE => e.static_eval as i32,
                _ => {
                    let e = evaluate(board, self.features.rule50_fade);
                    self.tt.store_eval_only(board.hash, e as i16);
                    e
                }
            }
        };
        let static_eval = raw_static_eval;
        // The table keeps the raw number, the search uses the corrected one.
        // Deliberately different: the table is shared, and whoever reads it
        // later applies their own correction.
        let static_eval = if in_check || !self.features.corr_hist {
            static_eval
        } else {
            self.corrected(board, static_eval, ply)
        };

        self.eval_stack[ply] = static_eval;
        // Is the side to move better off than it was two plies ago?
        //
        // A ply spent in check has no static evaluation and its slot holds a
        // sentinel, not a score. Comparing against the sentinel made every
        // position for two plies after any check look like it was improving,
        // because anything beats minus thirty two thousand -- so reverse
        // futility pruned harder and late move pruning cut later, both on a
        // fact that was not one. Step back four plies when two are not usable,
        // and claim nothing when neither is.
        let usable = |v: i32| v != TT_EVAL_NONE as i32;
        let improving = if in_check {
            false
        } else if ply >= 2 && usable(self.eval_stack[ply - 2]) {
            static_eval > self.eval_stack[ply - 2]
        } else if ply >= 4 && usable(self.eval_stack[ply - 4]) {
            static_eval > self.eval_stack[ply - 4]
        } else {
            false
        };

        // Two evaluations from here on, and they are not the same number.
        //
        // `static_eval` is what the network says, corrected, and it is what
        // `improving` and the forward futility margin compare against -- both
        // want a value that means the same thing at every ply, which a score
        // borrowed from a search does not.
        //
        // `pruning_eval` is that value improved by what the table already
        // knows. A stored lower bound above the static score, or an upper
        // bound below it, is a better estimate than the static score by
        // definition: a search went and found out. Whole-node pruning should
        // use the better one, and it was using the worse one.
        let mut pruning_eval = static_eval;
        if !in_check {
            if let Some(e) = entry {
                if e.has_bound() {
                    let ts = score_from_tt(e.score, ply);
                    let better = match e.bound {
                        Bound::Exact => true,
                        Bound::Lower => ts > static_eval,
                        Bound::Upper => ts < static_eval,
                        Bound::NoBound => false,
                    };
                    if better {
                        pruning_eval = ts;
                    }
                }
            }
        }

        if !pv_node && !in_check {
            // Reverse futility: so far ahead that giving away the margin still
            // beats beta, and the opponent has no way to take it all back in
            // the remaining depth. A ply that is improving can afford a
            // narrower margin, since the trend is evidence in the same
            // direction as the score.
            let margin = self.params.rfp_margin * depth
                - self.params.rfp_improving * improving as i32;
            if depth < self.params.rfp_depth
                && pruning_eval - margin >= beta
                && pruning_eval.abs() < MATE_IN_MAX
            {
                // Part of the way to the estimate rather than all of it. The
                // margin establishes that the node is above beta, not by how
                // much, and returning the whole distance passes upwards a
                // confidence that was never earned.
                return if self.features.rfp_damp {
                    beta + (pruning_eval - beta) / 3
                } else {
                    pruning_eval
                };
            }

            // Razoring: so far behind that even the quiescence search is
            // unlikely to find enough, so ask it directly instead of spending
            // a full width on the answer. If it turns out to be wrong the
            // score comes back above alpha and the node is searched properly.
            // The plain static score here, not the one the table improved.
            // Razoring asks whether the position is so far behind that only a
            // capture sequence could save it; a bound borrowed from a search
            // has already priced those in, and asking with it is asking a
            // question that has been answered.
            //
            // And not when alpha is already decisive -- a margin has nothing to
            // say about a position that is being mated.
            if self.features.razoring
                && depth <= self.params.razor_depth
                && alpha.abs() < 2000
                && static_eval + self.params.razor_margin * depth <= alpha
            {
                let q = self.quiescence(board, alpha, alpha + 1, ply);
                if q < alpha {
                    return q;
                }
            }

            // Null move: hand the opponent a free move and see whether the
            // position still holds. Not with only pawns left, where passing is
            // often the best move there is and the conclusion would be wrong.
            // The reduction grows with how far above beta we already are,
            // rather than with depth: the question null move asks is whether
            // the position is so good it survives giving away a move, and how
            // good it is answers that better than how deep we are.
            //
            // The extra conditions matter: the
            // raw static score has to be at least as good as the uncorrected
            // one, and the uncorrected one has to be within reach of beta. A
            // position that only looks good because the table said so is not
            // one to hand a free move away in.
            // Only where the node is expected to fail high. Elsewhere the
            // question null move asks -- is this so good it survives giving a
            // move away -- is not the question the node is there to answer.
            if (!self.features.nmp_cut_node || cut_node)
                && depth >= 3
                && pruning_eval >= beta
                && pruning_eval >= self.eval_stack[ply]
                && self.eval_stack[ply]
                    >= beta - 20 * depth - 40 * improving as i32 + 100
                && has_pieces(board, board.side)
                && !(ply > 0 && self.null_at[ply - 1])
            {
                let r = self.params.nmp_base
                    + ((pruning_eval - beta) / 200).min(self.params.nmp_div);
                let undo = board.make_null_move();
                self.keys.push(board.hash);
                self.null_at[ply] = true;
                // Passing the position over expects the opposite of whatever
                // this node expects.
                let score =
                    -self.negamax(board, depth - r, -beta, -beta + 1, ply + 1, false, !cut_node);
                self.null_at[ply] = false;
                self.keys.pop();
                board.unmake_null_move(&undo);
                if score >= beta {
                    // A mate score from a null move search is an artefact of
                    // the free move; report the bound instead.
                    return if is_mate(score) { beta } else { score };
                }
            }
        }

        // Nothing in the table for a node this deep means no move worth
        // trying first, and searching at full depth to discover one costs more
        // than finding it a ply shallower and coming back.
        if self.features.iir && depth >= 4 && tt_move.is_none() {
            depth -= 1;
        }

        // A stored lower bound far enough above beta already answers the
        // question this node was about to ask, even at a depth we would not
        // normally trust. It cost a search once; there is no reason to pay
        // again to be told the same thing by a smaller margin.
        if self.features.probcut && !pv_node && !in_check && excluded.is_none() {
            if let Some(e) = entry {
                if matches!(e.bound, Bound::Lower | Bound::Exact)
                    && e.depth >= depth - 2
                    && beta.abs() < MATE_IN_MAX
                {
                    let ts = score_from_tt(e.score, ply);
                    if !is_mate(ts) && ts >= beta + self.params.probcut_margin {
                        return ts;
                    }
                }
            }
        }

        let mut moves = generate_legal(board, &self.atk);
        if moves.is_empty() {
            return if in_check { mate_score(ply) } else { 0 };
        }
        let (mut scores, mut hist) = self.score_moves(board, &moves, tt_move, ply, depth);

        let mut best_score = -INF;
        let mut best_move = None;
        let alpha_orig = alpha;
        let mut searched_quiets: Vec<Move> = Vec::new();
        let mut searched_captures: Vec<Move> = Vec::new();

        let tt_score_for_singular = entry
            .filter(|e| e.has_bound())
            .map(|e| score_from_tt(e.score, ply));

        for i in 0..moves.len() {
            Self::pick(&mut moves, &mut scores, &mut hist, i);
            let mv = moves[i];
            if Some(mv) == excluded {
                continue;
            }
            let is_quiet = !mv.is_capture() && mv.promotion.is_none();
            let mut extension = 0;

            // Singular extension. If the table says this move is good enough to
            // fail high, search every OTHER move against a window just below
            // that. If they all fall short, this move is the only one holding
            // the position up, and a line that hangs on one move deserves
            // another ply to be sure of it.
            if !root
                && excluded.is_none()
                && Some(mv) == tt_move
                && depth >= self.params.sing_depth
                && ply < MAX_PLY - 8
            {
                if let Some(ts) = tt_score_for_singular {
                    let e = entry.unwrap();
                    if e.depth >= depth - 3
                        && matches!(e.bound, Bound::Lower | Bound::Exact)
                        && !is_mate(ts)
                    {
                        let target = ts - self.params.sing_margin * depth;
                        self.excluded[ply] = Some(mv);
                        let s = self.negamax(
                            board,
                            (depth - 1) / 2,
                            target - 1,
                            target,
                            ply,
                            false,
                            cut_node,
                        );
                        self.excluded[ply] = None;
                        if self.stopped {
                            return 0;
                        }
                        if s < target {
                            extension = 1;
                            // Not merely singular but singular by a distance:
                            // every alternative fell a long way short, so the
                            // line is even narrower than one ply of extension
                            // says. Outside the principal variation only, where
                            // being wrong costs a subtree rather than the move
                            // we play.
                            if !pv_node && s < target - self.params.double_ext {
                                extension = 2;
                            }
                        } else if target >= beta {
                            // Every other move also beats beta, so the position
                            // is winning for reasons that do not depend on this
                            // one and the whole subtree can go.
                            return target;
                        } else if !pv_node && !is_mate(s) && s >= beta {
                            return s;
                        } else if ts >= beta {
                            // The table says this move fails high, and the
                            // search just said it is not the only one that
                            // does. A node with several good answers is the
                            // opposite of the case worth extending, so take a
                            // ply off rather than adding one.
                            extension = -1;
                        }
                    }
                }
            }

            // Everything here needs a score already in hand: without one, the
            // node has nothing to compare a margin against and skipping moves
            // risks reporting a mate that is not there.
            if !root
                && !pv_node
                && !in_check
                && best_score > -MATE_IN_MAX
                && has_pieces(board, board.side)
            {
                if is_quiet {
                    // Late move pruning: past a certain count at low depth,
                    // the ordering has been wrong often enough that the rest
                    // are not worth the nodes.
                    let full = self.params.lmp_base + depth * depth;
                    let count = if !self.features.lmp_improving || improving {
                        full
                    } else {
                        full / 2
                    };
                    if depth <= self.params.lmp_depth && i >= count as usize {
                        break;
                    }

                    // Futility: even handed the margin, this move does not
                    // reach alpha, and a quiet move does not change the
                    // material to make up the difference.
                    // The history term belongs here:
                    // a move the tables like is worth trying even when the
                    // margin says otherwise, and one they dislike is worth
                    // less than the margin suggests. Its divisor is in OUR
                    // history units, which run about five and a half times
                    // smaller.
                    let hist_term = hist[i] / self.params.fut_hist_div.max(1);
                    if depth <= self.params.fut_depth
                        && static_eval
                            + self.params.fut_base
                            + self.params.fut_slope * depth
                            + hist_term
                            <= alpha
                    {
                        break;
                    }

                    // History pruning. A quiet move the tables have disliked
                    // this consistently, at a depth this shallow, is not worth
                    // the node. The threshold grows with the square of the
                    // depth so that it only bites where being wrong is cheap.
                    //
                    // The constant is in OUR history units and had to be. Taken
                    // straight from a design whose tables run to about
                    // 105000, against ours that cap near 24500, it never once
                    // fired -- the two runs came back with byte-identical node
                    // counts, which is what a dead branch looks like from
                    // outside.
                    if self.features.history_prune
                        && depth <= 4
                        && hist[i] < -self.params.hist_prune * depth * depth
                    {
                        continue;
                    }

                    // A quiet move can still lose material -- walking a piece
                    // onto a square where it is taken for nothing. Static
                    // exchange says so before the search has to find out.
                    if depth <= 8
                        && !see::see_ge(
                            &self.atk,
                            board,
                            &mv,
                            -self.params.see_prune_quiet * (depth + depth * depth),
                        )
                    {
                        continue;
                    }
                } else if depth <= 8
                    && !see::see_ge(&self.atk, board, &mv, -self.params.see_prune * depth)
                {
                    // A capture that loses more than the depth could plausibly
                    // win back.
                    continue;
                }
            }

            let nodes_before = self.nodes;
            self.played[ply] = board
                .piece_at(mv.from)
                .map(|(pt, _)| (pt.idx(), mv.to as usize));
            let undo = board.make_move(&mv);
            // Ask for the child's entry now. The probe happens a function call
            // and a check detection later, which is enough to cover part of the
            // trip to memory -- and that trip was a fifth of the whole search.
            self.tt.prefetch(board.hash);
            self.keys.push(board.hash);

            // A move that gives check is forcing: the reply is constrained and
            // the line is worth another ply. Only while the score says the game
            // is still a contest, since a check in a decided position extends
            // something that changes nothing.
            if self.features.check_ext
                && board.in_check(board.side, &self.atk)
                && static_eval != TT_EVAL_NONE as i32
                && static_eval.abs() > self.params.check_ext_eval
            {
                extension = extension.max(1);
            }

            let new_depth = depth - 1 + extension;

            let mut did_lmr = false;
            let mut searched_again = false;
            let mut score;
            if i == 0 {
                // The first move of a principal variation node leads to another
                // one; anywhere else the child expects the opposite of us.
                let child_cut = if pv_node { false } else { !cut_node };
                score =
                    -self.negamax(board, new_depth, -beta, -alpha, ply + 1, pv_node, child_cut);
            } else {
                // Late move reductions: the ordering has already put the moves
                // most likely to be best first, so the ones at the back are
                // searched shallower until one of them proves otherwise.
                // Late captures are reduced too, outside the principal
                // variation. A capture is not automatically worth a full look
                // just for being a capture -- the ones that were worth it are
                // already at the front of the list, and the ones down here have
                // been sorted below quiet moves by static exchange for a
                // reason.
                let reducible = is_quiet
                    || (self.features.lmr_captures && !pv_node && depth >= 3);

                did_lmr = true;
                let mut r = 0;
                if self.features.log_lmr {
                    // The other shape, whole. Integer logarithms of the
                    // move number and the depth, a ply back for anything
                    // tactical or on the principal variation, the history
                    // divided by the ceiling one table reaches, two plies at a
                    // node expected to fail high, and one always.
                    //
                    // Our history tables were rebuilt to its scale earlier, so
                    // the divisor transfers without conversion.
                    if depth > 2 && i >= 1 + 2 * root as usize && (!pv_node || is_quiet) {
                        let n = i as i32 + 1;
                        r = ilog2i(n) / 2 + ilog2i(depth) / 2
                            - (!is_quiet || pv_node) as i32
                            - (hist[i] + 15000) / 30000
                            + 2 * cut_node as i32
                            + 1;
                        r = r.clamp(0, (new_depth - 1).max(0));
                    }
                } else if depth >= 3 && reducible && !in_check {
                    // Accumulated in 1024ths and divided at the end, so a term
                    // can be worth a third of a ply instead of all or nothing.
                    // The first version added whole plies, and the cut node term
                    // alone was two of them where a third of one is the right
                    // size. Five times too much, which is exactly why measuring
                    // it found it doing no good.
                    let mut r1024 = self.lmr[(depth as usize).min(63)][i.min(63)] * 1024;
                    if !is_quiet {
                        // Half as hard for a capture: it changes the material,
                        // so a mistake about it costs more than a mistake about
                        // a quiet move.
                        r1024 /= 2;
                    }
                    if self.features.cut_node_lmr && cut_node {
                        r1024 += self.params.lmr_cut_f;
                    }
                    if !pv_node {
                        r1024 += self.params.lmr_nonpv_f;
                    }
                    // A position that once earned a full window is less likely
                    // to be the throwaway this reduction assumes.
                    if self.features.ttpv_lmr && tt_pv {
                        r1024 -= self.params.lmr_ttpv_f;
                    }
                    // A move the history likes gets the benefit of the doubt,
                    // one it dislikes gets less of it.
                    r1024 -= (hist[i] * 1024 / self.params.lmr_hist_div.max(1))
                        .clamp(-2048, 2048);
                    r = (r1024 / 1024).clamp(0, (new_depth - 1).max(0));
                }
                // A reduced scout search is looking for a reason to stop, so
                // the child is treated as expecting to fail high.
                score =
                    -self.negamax(board, new_depth - r, -alpha - 1, -alpha, ply + 1, false, true);
                if score > alpha && r > 0 {
                    searched_again = true;
                    score = -self.negamax(
                        board,
                        new_depth,
                        -alpha - 1,
                        -alpha,
                        ply + 1,
                        false,
                        !cut_node,
                    );
                }
                if score > alpha && score < beta {
                    score = -self.negamax(board, new_depth, -beta, -alpha, ply + 1, true, false);
                }
            }

            self.keys.pop();
            board.unmake_move(&mv, &undo);

            if root {
                self.root_effort.push((mv, self.nodes - nodes_before));
                // A move searched with a null window that failed low has no
                // real score, only a bound. Recorded anyway: it is still
                // evidence of the ordering, and the fallback below only ever
                // looks at moves close to the best, which fail-lows are not.
                self.root_scores.push((mv, score));
            }

            // A quiet move that was reduced and then had to be searched again
            // has told us something either way: it was worth the second look,
            // or it was not. Both are worth recording, and neither shows up in
            // the cutoff update, which only ever sees the move that ended the
            // node.
            if did_lmr && searched_again && is_quiet {
                let credit = if score > best_score {
                    hist_bonus(depth)
                } else {
                    -hist_bonus(depth)
                };
                let side = board.side.idx();
                let slots = self.cont_slots(ply);
                self.credit(board, mv, side, &slots, credit);
            }

            if self.stopped {
                return 0;
            }

            // At the root, and only there, give up on an iteration that has
            // already gone well past what was planned for the whole move. The
            // soft limit is otherwise consulted only between iterations, so a
            // single long one sails past it and the wall is all that catches
            // it -- much later, and much more expensively.
            //
            // Safe to stop here because a root move that has finished has a
            // real score: the best so far is a genuine best-so-far, not
            // whatever happened to be first.
            if root && i > 0 && self.start.elapsed() >= self.soft * 2 {
                self.stopped = true;
            }

            if score > best_score {
                best_score = score;
                best_move = Some(mv);

                if score > alpha {
                    alpha = score;
                    self.update_pv(ply, mv);

                    if alpha >= beta {
                        if is_quiet {
                            self.on_beta_cutoff(board, mv, ply, depth, &searched_quiets);
                        } else {
                            self.credit_capture(board, mv, capt_bonus(depth));
                        }
                        // Captures tried and passed over were wrong here
                        // whatever ended the node, quiet or not.
                        for c in &searched_captures {
                            if *c != mv {
                                self.credit_capture(board, *c, -capt_bonus(depth));
                            }
                        }
                        break;
                    }
                }
            }

            if is_quiet {
                searched_quiets.push(mv);
            } else {
                searched_captures.push(mv);
            }
        }

        if best_score == -INF {
            return alpha;
        }

        // Learn the correction from what the search ended up saying, but only
        // where it contradicts the static score in a direction worth trusting:
        // below it and below beta, so there is an upper bound proving the
        // static score was optimistic, or above it with a move to show why.
        // Anywhere else the number is the product of a cutoff, not a reading of
        // the position. Captures are excluded: there the jump comes from
        // material rather than from misreading, and it would teach the table
        // the wrong thing.
        if self.features.corr_hist
            && !in_check
            && best_score.abs() < MATE_IN_MAX
            && !best_move.map_or(false, |m| m.is_capture())
            && ((best_score < static_eval && best_score < beta)
                || (best_score > static_eval && best_move.is_some()))
        {
            self.learn_correction(board, best_score - static_eval, depth, ply);
        }

        let bound = if best_score >= beta {
            Bound::Lower
        } else if best_score > alpha_orig {
            Bound::Exact
        } else {
            Bound::Upper
        };
        let se = if in_check {
            TT_EVAL_NONE
        } else {
            raw_static_eval as i16
        };
        if excluded.is_none() {
            self.tt.store(
                board.hash,
                depth,
                score_to_tt(best_score, ply),
                bound,
                best_move,
                tt_pv,
                se,
            );
        }

        best_score
    }

    fn quiescence(&mut self, board: &mut Board, mut alpha: i32, beta: i32, ply: usize) -> i32 {
        self.nodes += 1;
        if self.out_of_time() {
            return 0;
        }
        if ply >= MAX_PLY - 1 {
            return evaluate(board, self.features.rule50_fade);
        }
        if self.is_draw(board) {
            return self.draw_score(ply);
        }

        let in_check = board.in_check(board.side, &self.atk);
        let alpha_orig = alpha;

        // The table is worth reading here too. Quiescence is most of the tree,
        // the same capture sequences transpose constantly, and every hit that
        // returns saves not just a node but the whole tail behind it.
        let entry = self.tt.probe(board.hash);
        let mut tt_move = None;
        if let Some(e) = entry {
            tt_move = e.best;
            if e.has_bound() {
                let sc = score_from_tt(e.score, ply);
                let usable = match e.bound {
                    Bound::Exact => true,
                    Bound::Lower => sc >= beta,
                    Bound::Upper => sc <= alpha,
                    Bound::NoBound => false,
                };
                if usable {
                    return sc;
                }
            }
        }

        // Standing pat: the side to move is not obliged to capture, so the
        // static score is a floor. Not while in check, where every move is
        // forced and there is nothing to stand on.
        let mut static_eval = TT_EVAL_NONE as i32;
        let mut stand = TT_EVAL_NONE as i32;
        if !in_check {
            static_eval = match entry {
                Some(e) if e.static_eval != TT_EVAL_NONE => e.static_eval as i32,
                _ => evaluate(board, self.features.rule50_fade),
            };

            // The floor to stand on is the better of the static score and
            // whatever the table already established, for the same reason the
            // full search prefers it: a stored bound on the right side of the
            // static score came from a search that went and found out. Standing
            // on the worse number means searching captures to reach a value
            // already in hand.
            let mut floor = static_eval;
            if let Some(e) = entry {
                if e.has_bound() {
                    let ts = score_from_tt(e.score, ply);
                    let better = match e.bound {
                        Bound::Exact => true,
                        Bound::Lower => ts > static_eval,
                        Bound::Upper => ts < static_eval,
                        Bound::NoBound => false,
                    };
                    if better {
                        floor = ts;
                    }
                }
            }
            if floor >= beta {
                return floor;
            }
            if floor > alpha {
                alpha = floor;
            }
            stand = floor;

        }

        let mut moves = if in_check {
            generate_legal(board, &self.atk)
        } else {
            generate_legal_caps(board, &self.atk)
        };
        if in_check && moves.is_empty() {
            return mate_score(ply);
        }
        let (mut scores, _) = self.score_moves(board, &moves, tt_move, ply, 1);

        let mut best = if in_check { -INF } else { stand };
        let mut best_move = None;

        let mut hist_unused: Vec<i32> = vec![0; moves.len()];
        for i in 0..moves.len() {
            Self::pick(&mut moves, &mut scores, &mut hist_unused, i);
            let mv = moves[i];

            // A capture that loses material cannot raise the floor we are
            // already standing on, and following it is how quiescence chases
            // every recapture to the horizon instead of settling.
            if !in_check && !see::see_ge(&self.atk, board, &mv, 0) {
                continue;
            }

            // And a capture that wins everything it takes and is still nowhere
            // near alpha cannot help either. Quiescence is most of the tree, so
            // this is the cheapest place in the search to stop looking at moves
            // that were never going to matter.
            //
            // The margin is generous on purpose: the price of being wrong here
            // is missing a tactic, and the whole point of quiescence is not
            // missing tactics.
            if self.features.qs_futility && !in_check && best.abs() < MATE_IN_MAX {
                let taken = if mv.flag == MoveFlag::EnPassant {
                    value_in_eval_units(PieceType::Pawn)
                } else {
                    board
                        .piece_at(mv.to)
                        .map(|(pt, _)| value_in_eval_units(pt))
                        .unwrap_or(0)
                };
                let promo = mv
                    .promotion
                    .map(|p| value_in_eval_units(p) - value_in_eval_units(PieceType::Pawn))
                    .unwrap_or(0);
                // Measured against the static score, not against the floor
                // the table raised or lowered. What this test knows how to
                // correct is a piece value, and a piece value only means
                // something added to a static evaluation of the same position.
                // A stored upper bound below the static score makes the floor
                // pessimistic, and pruning a capture against a pessimistic
                // floor throws away captures that were good: measured, the
                // difference was three wins against thirty-four losses.
                if static_eval != TT_EVAL_NONE as i32
                    && static_eval + taken + promo + 200 <= alpha
                {
                    continue;
                }
            }

            let undo = board.make_move(&mv);
            self.tt.prefetch(board.hash);
            self.keys.push(board.hash);
            let score = -self.quiescence(board, -beta, -alpha, ply + 1);
            self.keys.pop();
            board.unmake_move(&mv, &undo);

            if self.stopped {
                return 0;
            }
            if score > best {
                best = score;
                if score > alpha {
                    alpha = score;
                    best_move = Some(mv);
                    if alpha >= beta {
                        break;
                    }
                }
            }
        }

        let bound = if best >= beta {
            Bound::Lower
        } else if best > alpha_orig {
            Bound::Exact
        } else {
            Bound::Upper
        };
        self.tt.store(
            board.hash,
            0,
            score_to_tt(best, ply),
            bound,
            best_move,
            false,
            if in_check { TT_EVAL_NONE } else { static_eval as i16 },
        );

        best
    }

    pub(crate) fn update_pv_pub(&mut self, ply: usize, mv: Move) {
        self.update_pv(ply, mv);
    }

    fn update_pv(&mut self, ply: usize, mv: Move) {
        self.pv[ply][0] = Some(mv);
        let child = self.pv_len[ply + 1].min(MAX_PLY - ply - 2);
        for i in 0..child {
            self.pv[ply][i + 1] = self.pv[ply + 1][i];
        }
        self.pv_len[ply] = child + 1;
    }

    fn on_beta_cutoff(
        &mut self,
        board: &Board,
        mv: Move,
        ply: usize,
        depth: i32,
        searched: &[Move],
    ) {
        if !self.killers[ply].iter().any(|k| *k == Some(mv)) {
            for i in (1..NUM_KILLERS).rev() {
                self.killers[ply][i] = self.killers[ply][i - 1];
            }
            self.killers[ply][0] = Some(mv);
        }
        let side = board.side.idx();
        let bonus = hist_bonus(depth);
        let slots = self.cont_slots(ply);

        self.credit(board, mv, side, &slots, bonus);
        for q in searched {
            if *q != mv {
                self.credit(board, *q, side, &slots, -bonus);
            }
        }
    }

    /// What a capture is worth by past results, zero when the table is off.
    #[inline]
    fn capt_score(&self, board: &Board, mv: &Move) -> i32 {
        if !self.features.capture_hist {
            return 0;
        }
        let pc = match board.piece_at(mv.from) {
            Some((pt, _)) => pt.idx(),
            None => return 0,
        };
        let victim = if mv.flag == MoveFlag::EnPassant {
            PieceType::Pawn.idx()
        } else {
            match board.piece_at(mv.to) {
                Some((pt, _)) => pt.idx(),
                None => return 0,
            }
        };
        self.capthist[pc][mv.to as usize][victim]
    }

    /// Move a capture for or against by past results.
    #[inline]
    fn credit_capture(&mut self, board: &Board, mv: Move, bonus: i32) {
        if !self.features.capture_hist {
            return;
        }
        let pc = match board.piece_at(mv.from) {
            Some((pt, _)) => pt.idx(),
            None => return,
        };
        let victim = if mv.flag == MoveFlag::EnPassant {
            PieceType::Pawn.idx()
        } else {
            match board.piece_at(mv.to) {
                Some((pt, _)) => pt.idx(),
                None => return,
            }
        };
        hist_add(&mut self.capthist[pc][mv.to as usize][victim], bonus, HIST_MAX_CAPT);
    }

    /// Move one move for and every other move against, by the same amount.
    #[inline]
    fn credit(
        &mut self,
        board: &Board,
        mv: Move,
        side: usize,
        slots: &[Option<usize>; CONT_SLOTS],
        bonus: i32,
    ) {
        hist_add(
            &mut self.history[side][mv.from as usize][mv.to as usize],
            bonus,
            HIST_MAX_MAIN,
        );
        if let Some(pc) = board.piece_at(mv.from).map(|(pt, _)| pt.idx()) {
            for (k, slot) in slots.iter().enumerate() {
                if let Some(idx) = slot {
                    hist_add(
                        &mut self.conthist[k][*idx][pc][mv.to as usize],
                        bonus,
                        HIST_MAX_CONT,
                    );
                }
            }
        }
    }

    /// Score every move once. The caller then takes the best remaining one at
    /// a time, because most nodes cut off after two or three and sorting the
    /// other thirty-seven is work thrown away.
    fn score_moves(
        &self,
        board: &Board,
        moves: &[Move],
        tt_move: Option<Move>,
        ply: usize,
        depth: i32,
    ) -> (Vec<i32>, Vec<i32>) {
        let side = board.side.idx();
        // Hoisted: the continuation slots depend on the ply, not on the move,
        // and looking them up per move turned a couple of array reads into a
        // couple per move in the hottest loop there is.
        let slots = self.cont_slots(ply);

        // What the tables actually think of each move, kept apart from where it
        // goes in the list.
        //
        // These were the same number, and it was wrong. Ordering uses large
        // sentinels -- a million for the table move, four hundred thousand for
        // a killer, plus or minus six hundred thousand for a capture -- so any
        // reduction scaled by "the score" saturated on the sentinel and had
        // nothing to do with history at all. Every killer and table move was
        // being reduced two plies less, and every losing capture two plies
        // more, on the strength of a tag rather than a fact.
        let hist: Vec<i32> = moves
            .iter()
            .map(|mv| {
                if mv.is_capture() || mv.promotion.is_some() {
                    return 0;
                }
                let mut h = self.history[side][mv.from as usize][mv.to as usize];
                if let Some(pc) = board.piece_at(mv.from).map(|(pt, _)| pt.idx()) {
                    for (k, slot) in slots.iter().enumerate() {
                        if let Some(idx) = slot {
                            h += CONT_WEIGHT[k] * self.conthist[k][*idx][pc][mv.to as usize];
                        }
                    }
                }
                h
            })
            .collect();

        let order = moves
            .iter()
            .enumerate()
            .map(|(n, mv)| {
                if Some(*mv) == tt_move {
                    1_000_000
                } else if mv.is_capture() {
                    let victim = if mv.flag == MoveFlag::EnPassant {
                        PieceType::Pawn.value()
                    } else {
                        board
                            .piece_at(mv.to)
                            .map(|(pt, _)| pt.value())
                            .unwrap_or(0)
                    };
                    let attacker = board
                        .piece_at(mv.from)
                        .map(|(pt, _)| pt.value())
                        .unwrap_or(0);
                    let mvv = victim * 16 - attacker;
                    // A capture that loses material is not a good move that
                    // happens to be violent, and putting it ahead of the quiet
                    // moves on the strength of what it takes is how a search
                    // spends its first three tries on refuted sacrifices.
                    // Below everything, then, but still ahead of nothing.
                    // The bar for a capture counting as good drops with depth.
                    // Deep in the tree there is room to find out whether a
                    // capture that looks slightly losing actually is; near the
                    // leaves there is not, so only the clearly good ones go
                    // first.
                    let bar = (-50 * (depth - 1)).max(-250);
                    let past = self.capt_score(board, mv) / 16;
                    if see::see_ge(&self.atk, board, mv, bar) {
                        600_000 + mvv + past
                    } else {
                        -600_000 + mvv + past
                    }
                } else if mv.promotion == Some(PieceType::Queen) {
                    500_000
                } else if let Some(k) =
                    self.killers[ply].iter().position(|k| *k == Some(*mv))
                {
                    400_000 - 10_000 * k as i32
                } else {
                    hist[n]
                }
            })
            .collect();

        (order, hist)
    }

    /// Bring the best remaining move to `at`, keeping scores in step.
    #[inline]
    fn pick(moves: &mut [Move], scores: &mut [i32], hist: &mut [i32], at: usize) {
        let mut best = at;
        for j in at + 1..moves.len() {
            if scores[j] > scores[best] {
                best = j;
            }
        }
        moves.swap(at, best);
        scores.swap(at, best);
        hist.swap(at, best);
    }
}
