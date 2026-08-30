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
fn lmr_table() -> &'static [[i32; 64]; 64] {
    static T: std::sync::OnceLock<[[i32; 64]; 64]> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = [[0i32; 64]; 64];
        for d in 1..64usize {
            for m in 1..64usize {
                t[d][m] = (0.77 + (d as f64).ln() * (m as f64).ln() / 2.36) as i32;
            }
        }
        t
    })
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
const CORR_GRAIN: i32 = 256;
const CORR_MAX: i32 = 32 * CORR_GRAIN;

#[inline]
fn corr_index(board: &Board) -> usize {
    let w = board.pieces[Color::White.idx()][PieceType::Pawn.idx()];
    let b = board.pieces[Color::Black.idx()][PieceType::Pawn.idx()];
    let mut z = w.wrapping_mul(0x9e37_79b9_7f4a_7c15).wrapping_add(b);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    ((z ^ (z >> 31)) as usize) & (CORR_SIZE - 1)
}

pub const MAX_PLY: usize = 128;
pub const INF: i32 = 32_000;
pub const MATE: i32 = 31_000;
/// Anything at least this large is a mate score, not an evaluation.
pub const MATE_IN_MAX: i32 = MATE - MAX_PLY as i32;

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

    nodes: u64,
    start: Instant,
    soft: Duration,
    hard: Duration,
    stopped: bool,

    killers: [[Option<Move>; 2]; MAX_PLY],
    history: [[[i32; 64]; 64]; 2],
    /// `[side][pawn structure]`.
    corr: Vec<[i32; CORR_SIZE]>,
    /// What was played at each ply, as (piece, destination). Continuation
    /// history is indexed by this: a move is good or bad largely in reply to
    /// something, and a table that ignores what came before cannot say which.
    played: [Option<(usize, usize)>; MAX_PLY],
    /// `[prev piece * 64 + prev to][piece][to]`.
    conthist: Vec<[[i32; 64]; 6]>,
    /// Zobrist keys along the path plus the game so far, for repetition.
    keys: Vec<u64>,
    /// How many of `keys` are game history rather than search path.
    root_keys: usize,

    pv: [[Option<Move>; MAX_PLY]; MAX_PLY],
    pv_len: [usize; MAX_PLY],
    /// Which plies got there by passing. Two passes in a row prove nothing:
    /// the side to move has effectively been given a free tempo twice, and the
    /// position being searched is not one that can occur.
    null_at: [bool; MAX_PLY],
}

/// The score of a position from the side to move's point of view.
fn evaluate(board: &Board) -> i32 {
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
    raw * (200 - board.halfmove.min(100) as i32) / 200
}

/// Does this side have anything but pawns and a king?
///
/// The question null move pruning asks: with only pawns left, having to move is
/// often a disadvantage, so a side that passes and still looks fine proves
/// nothing about a side that has to play.
fn has_pieces(board: &Board, side: Color) -> bool {
    let p = &board.pieces[side.idx()];
    p[PieceType::Knight.idx()]
        | p[PieceType::Bishop.idx()]
        | p[PieceType::Rook.idx()]
        | p[PieceType::Queen.idx()]
        != 0
}

/// The evaluation, exposed for the UCI `eval` command.
pub fn debug_eval(board: &Board) -> i32 {
    evaluate(board)
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
fn score_to_tt(score: i32, ply: usize) -> i32 {
    if score >= MATE_IN_MAX {
        score + ply as i32
    } else if score <= -MATE_IN_MAX {
        score - ply as i32
    } else {
        score
    }
}

fn score_from_tt(score: i32, ply: usize) -> i32 {
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
            nodes: 0,
            start: Instant::now(),
            soft: Duration::from_secs(0),
            hard: Duration::from_secs(0),
            stopped: false,
            killers: [[None; 2]; MAX_PLY],
            history: [[[0; 64]; 64]; 2],
            corr: vec![[0; CORR_SIZE]; 2],
            played: [None; MAX_PLY],
            conthist: vec![[[0; 64]; 6]; 6 * 64],
            keys: Vec::with_capacity(1024),
            root_keys: 0,
            pv: [[None; MAX_PLY]; MAX_PLY],
            pv_len: [0; MAX_PLY],
            null_at: [false; MAX_PLY],
        }
    }

    pub fn set_game_history(&mut self, keys: Vec<u64>) {
        self.keys = keys;
        self.root_keys = self.keys.len();
    }

    pub fn clear(&mut self) {
        self.tt.clear();
        self.killers = [[None; 2]; MAX_PLY];
        self.history = [[[0; 64]; 64]; 2];
        self.corr = vec![[0; CORR_SIZE]; 2];
        self.conthist = vec![[[0; 64]; 6]; 6 * 64];
    }

    /// The static score, adjusted by what the search has been saying about
    /// positions with this pawn structure.
    #[inline]
    fn corrected(&self, board: &Board, raw: i32) -> i32 {
        let c = self.corr[board.side.idx()][corr_index(board)] / CORR_GRAIN;
        (raw + c).clamp(-MATE_IN_MAX + 1, MATE_IN_MAX - 1)
    }

    /// Learn from the difference, weighted by how deep the search that found
    /// it went.
    #[inline]
    fn learn_correction(&mut self, board: &Board, diff: i32, depth: i32) {
        let e = &mut self.corr[board.side.idx()][corr_index(board)];
        let target = (diff * CORR_GRAIN).clamp(-CORR_MAX, CORR_MAX);
        let w = (depth + 1).min(16);
        *e = ((*e * (256 - w) + target * w) / 256).clamp(-CORR_MAX, CORR_MAX);
    }

    /// The continuation tables in reach at this ply: one ply back and two.
    #[inline]
    fn cont_slots(&self, ply: usize) -> [Option<usize>; 2] {
        let mut out = [None; 2];
        for (k, back) in [1usize, 2].iter().enumerate() {
            if ply >= *back {
                if let Some((pc, to)) = self.played[ply - back] {
                    out[k] = Some(pc * 64 + to);
                }
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
    fn allocate(&mut self, limits: &Limits, side: Color) {
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

        let (time, inc) = match side {
            Color::White => (limits.wtime, limits.winc),
            Color::Black => (limits.btime, limits.binc),
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
        let mtg = limits.movestogo.unwrap_or(25).max(1);

        // The increment is income, so most of it can be spent every move
        // without the clock moving. Not all of it: the part held back is what
        // slowly rebuilds a buffer over a long game.
        let base = usable / mtg + inc * 3 / 4;

        // Two ceilings on the wall, and the second is the one that matters.
        //
        // Twice the plan lets a critical move think a little longer. Two
        // fifths of what is left stops that from becoming a way to spend the
        // clock.
        let hard = (base * 2).min(usable * 2 / 5);
        let soft = base.min(hard);

        self.soft = Duration::from_millis(soft.max(1));
        self.hard = Duration::from_millis(hard.max(1));
    }

    #[inline]
    fn out_of_time(&mut self) -> bool {
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

    /// Has this position already occurred? One earlier occurrence is enough to
    /// treat it as drawn inside the search -- waiting for the third makes the
    /// search miss the repetition it is about to walk into.
    fn is_repetition(&self, board: &Board) -> bool {
        let back = (board.halfmove as usize).min(self.keys.len());
        // Same side to move means stepping back two at a time.
        self.keys
            .iter()
            .rev()
            .take(back)
            .skip(1)
            .step_by(2)
            .any(|k| *k == board.hash)
    }

    fn is_draw(&self, board: &Board) -> bool {
        board.halfmove >= 100 || self.is_repetition(board)
    }

    pub fn go(&mut self, board: &mut Board, limits: &Limits, info: bool) -> Option<Move> {
        self.allocate(limits, board.side);
        self.nodes = 0;
        self.stopped = false;
        self.stop.store(false, Ordering::Relaxed);
        self.tt.increase_gen();
        self.keys.truncate(self.root_keys);

        let max_depth = limits.depth.unwrap_or(MAX_PLY as u32 - 2).min(MAX_PLY as u32 - 2);

        let mut best: Option<Move> = None;
        let mut best_score = 0;

        for depth in 1..=max_depth {
            let iter_start = self.start.elapsed();
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

            if info {
                self.print_info(depth, score);
            }

            if limits.nodes.map_or(false, |n| self.nodes >= n) {
                break;
            }

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
            // Reported in the units the network was trained in, which is what
            // makes the win/draw/loss figures below mean anything.
            format!("cp {}", score)
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
        let mut delta = 25;
        let (mut alpha, mut beta) = if depth <= 4 || is_mate(prev) {
            (-INF, INF)
        } else {
            (prev - delta, prev + delta)
        };

        loop {
            let score = self.negamax(board, depth, alpha, beta, 0, true);
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

    fn negamax(
        &mut self,
        board: &mut Board,
        mut depth: i32,
        mut alpha: i32,
        beta: i32,
        ply: usize,
        pv_node: bool,
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
                return 0;
            }
            if ply >= MAX_PLY - 1 {
                return evaluate(board);
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

        // A king already in check has no quiet evaluation and the branch is
        // forcing, so it is worth another ply.
        if in_check {
            depth += 1;
        }

        let entry = self.tt.probe(board.hash);
        let mut tt_move = None;
        let mut tt_pv = pv_node;
        if let Some(e) = entry {
            tt_move = e.best;
            tt_pv |= e.pv;
            if !pv_node && e.depth >= depth && e.has_bound() {
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
                    return s;
                }
            }
        }

        let static_eval = if in_check {
            TT_EVAL_NONE as i32
        } else {
            match entry {
                Some(e) if e.static_eval != TT_EVAL_NONE => e.static_eval as i32,
                _ => {
                    let e = evaluate(board);
                    self.tt.store_eval_only(board.hash, e as i16);
                    e
                }
            }
        };
        // The table keeps the raw number, the search uses the corrected one.
        // Deliberately different: the table is shared, and whoever reads it
        // later applies their own correction.
        let static_eval = if in_check {
            static_eval
        } else {
            self.corrected(board, static_eval)
        };

        if !pv_node && !in_check {
            // Reverse futility: so far ahead that giving away the margin still
            // beats beta, and the opponent has no way to take it all back in
            // the remaining depth.
            if depth < 7 && static_eval - 80 * depth >= beta && static_eval.abs() < MATE_IN_MAX {
                return static_eval;
            }

            // Razoring: so far behind that even the quiescence search is
            // unlikely to find enough, so ask it directly instead of spending
            // a full width on the answer. If it turns out to be wrong the
            // score comes back above alpha and the node is searched properly.
            if depth <= 3 && static_eval + 300 * depth < alpha {
                let q = self.quiescence(board, alpha, alpha + 1, ply);
                if q < alpha {
                    return q;
                }
            }

            // Null move: hand the opponent a free move and see whether the
            // position still holds. Not with only pawns left, where passing is
            // often the best move there is and the conclusion would be wrong.
            if depth >= 3
                && static_eval >= beta
                && has_pieces(board, board.side)
                && !(ply > 0 && self.null_at[ply - 1])
            {
                let r = 3 + depth / 4;
                let undo = board.make_null_move();
                self.keys.push(board.hash);
                self.null_at[ply] = true;
                let score = -self.negamax(board, depth - r, -beta, -beta + 1, ply + 1, false);
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
        if depth >= 4 && tt_move.is_none() {
            depth -= 1;
        }

        let mut moves = generate_legal(board, &self.atk);
        if moves.is_empty() {
            return if in_check { mate_score(ply) } else { 0 };
        }
        let mut scores = self.score_moves(board, &moves, tt_move, ply);

        let mut best_score = -INF;
        let mut best_move = None;
        let alpha_orig = alpha;
        let mut searched_quiets: Vec<Move> = Vec::new();

        for i in 0..moves.len() {
            Self::pick(&mut moves, &mut scores, i);
            let mv = moves[i];
            let is_quiet = !mv.is_capture() && mv.promotion.is_none();

            // Everything here needs a score already in hand: without one, the
            // node has nothing to compare a margin against and skipping moves
            // risks reporting a mate that is not there.
            if !root && !pv_node && !in_check && best_score > -MATE_IN_MAX {
                if is_quiet {
                    // Late move pruning: past a certain count at low depth,
                    // the ordering has been wrong often enough that the rest
                    // are not worth the nodes.
                    if depth <= 6 && i >= (3 + depth * depth) as usize {
                        break;
                    }

                    // Futility: even handed the margin, this move does not
                    // reach alpha, and a quiet move does not change the
                    // material to make up the difference.
                    if depth <= 6 && static_eval + 120 + 130 * depth <= alpha {
                        break;
                    }
                } else if depth <= 8 && !see::see_ge(&self.atk, board, &mv, -90 * depth) {
                    // A capture that loses more than the depth could plausibly
                    // win back.
                    continue;
                }
            }

            self.played[ply] = board
                .piece_at(mv.from)
                .map(|(pt, _)| (pt.idx(), mv.to as usize));
            let undo = board.make_move(&mv);
            self.keys.push(board.hash);

            let mut score;
            if i == 0 {
                score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1, pv_node);
            } else {
                // Late move reductions: the ordering has already put the moves
                // most likely to be best first, so the ones at the back are
                // searched shallower until one of them proves otherwise.
                let mut r = 0;
                if depth >= 3 && is_quiet && !in_check {
                    r = lmr_table()[(depth as usize).min(63)][i.min(63)];
                    // A position that once earned a full window is less likely
                    // to be the throwaway this reduction assumes.
                    if tt_pv {
                        r -= 1;
                    }
                    if !pv_node {
                        r += 1;
                    }
                    // A move the history likes gets the benefit of the doubt,
                    // one it dislikes gets less of it.
                    r -= (scores[i] / 4096).clamp(-2, 2);
                    r = r.clamp(0, depth - 2);
                }
                score = -self.negamax(board, depth - 1 - r, -alpha - 1, -alpha, ply + 1, false);
                if score > alpha && r > 0 {
                    score = -self.negamax(board, depth - 1, -alpha - 1, -alpha, ply + 1, false);
                }
                if score > alpha && score < beta {
                    score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1, pv_node);
                }
            }

            self.keys.pop();
            board.unmake_move(&mv, &undo);

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
                        }
                        break;
                    }
                }
            }

            if is_quiet {
                searched_quiets.push(mv);
            }
        }

        // Learn the correction from what the search ended up saying, but only
        // where it contradicts the static score in a direction worth trusting:
        // below it and below beta, so there is an upper bound proving the
        // static score was optimistic, or above it with a move to show why.
        // Anywhere else the number is the product of a cutoff, not a reading of
        // the position. Captures are excluded: there the jump comes from
        // material rather than from misreading, and it would teach the table
        // the wrong thing.
        if !in_check
            && best_score.abs() < MATE_IN_MAX
            && !best_move.map_or(false, |m| m.is_capture())
            && ((best_score < static_eval && best_score < beta)
                || (best_score > static_eval && best_move.is_some()))
        {
            self.learn_correction(board, best_score - static_eval, depth);
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
            static_eval as i16
        };
        self.tt.store(
            board.hash,
            depth,
            score_to_tt(best_score, ply),
            bound,
            best_move,
            tt_pv,
            se,
        );

        best_score
    }

    fn quiescence(&mut self, board: &mut Board, mut alpha: i32, beta: i32, ply: usize) -> i32 {
        self.nodes += 1;
        if self.out_of_time() {
            return 0;
        }
        if ply >= MAX_PLY - 1 {
            return evaluate(board);
        }
        if self.is_draw(board) {
            return 0;
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
        if !in_check {
            static_eval = match entry {
                Some(e) if e.static_eval != TT_EVAL_NONE => e.static_eval as i32,
                _ => evaluate(board),
            };
            if static_eval >= beta {
                return static_eval;
            }
            if static_eval > alpha {
                alpha = static_eval;
            }
        }

        let mut moves = if in_check {
            generate_legal(board, &self.atk)
        } else {
            generate_legal_caps(board, &self.atk)
        };
        if in_check && moves.is_empty() {
            return mate_score(ply);
        }
        let mut scores = self.score_moves(board, &moves, tt_move, ply);

        let mut best = if in_check { -INF } else { static_eval };
        let mut best_move = None;

        for i in 0..moves.len() {
            Self::pick(&mut moves, &mut scores, i);
            let mv = moves[i];

            // A capture that loses material cannot raise the floor we are
            // already standing on, and following it is how quiescence chases
            // every recapture to the horizon instead of settling.
            if !in_check && !see::see_ge(&self.atk, board, &mv, 0) {
                continue;
            }

            let undo = board.make_move(&mv);
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
        if self.killers[ply][0] != Some(mv) {
            self.killers[ply][1] = self.killers[ply][0];
            self.killers[ply][0] = Some(mv);
        }
        let side = board.side.idx();
        let bonus = (depth * depth).min(1200);
        let slots = self.cont_slots(ply);

        // Saturating: without the pull towards zero a few deep cutoffs pin an
        // entry at the ceiling and it stops carrying information.
        let h = &mut self.history[side][mv.from as usize][mv.to as usize];
        *h += bonus - *h * bonus / 8192;
        if let Some(pc) = board.piece_at(mv.from).map(|(pt, _)| pt.idx()) {
            for slot in slots.into_iter().flatten() {
                let c = &mut self.conthist[slot][pc][mv.to as usize];
                *c += bonus - *c * bonus / 8192;
            }
        }

        for q in searched {
            let h = &mut self.history[side][q.from as usize][q.to as usize];
            *h += -bonus - *h * bonus / 8192;
            if let Some(pc) = board.piece_at(q.from).map(|(pt, _)| pt.idx()) {
                for slot in slots.into_iter().flatten() {
                    let c = &mut self.conthist[slot][pc][q.to as usize];
                    *c += -bonus - *c * bonus / 8192;
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
    ) -> Vec<i32> {
        let side = board.side.idx();
        // Hoisted: the continuation slots depend on the ply, not on the move,
        // and looking them up per move turned a couple of array reads into a
        // couple per move in the hottest loop there is.
        let slots = self.cont_slots(ply);
        moves
            .iter()
            .map(|mv| {
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
                    if see::see_ge(&self.atk, board, mv, 0) {
                        600_000 + mvv
                    } else {
                        -600_000 + mvv
                    }
                } else if mv.promotion == Some(PieceType::Queen) {
                    500_000
                } else if Some(*mv) == self.killers[ply][0] {
                    400_000
                } else if Some(*mv) == self.killers[ply][1] {
                    390_000
                } else {
                    let mut h = self.history[side][mv.from as usize][mv.to as usize];
                    if let Some(pc) = board.piece_at(mv.from).map(|(pt, _)| pt.idx()) {
                        for slot in slots.into_iter().flatten() {
                            h += self.conthist[slot][pc][mv.to as usize];
                        }
                    }
                    h
                }
            })
            .collect()
    }

    /// Bring the best remaining move to `at`, keeping scores in step.
    #[inline]
    fn pick(moves: &mut [Move], scores: &mut [i32], at: usize) {
        let mut best = at;
        for j in at + 1..moves.len() {
            if scores[j] > scores[best] {
                best = j;
            }
        }
        moves.swap(at, best);
        scores.swap(at, best);
    }
}
