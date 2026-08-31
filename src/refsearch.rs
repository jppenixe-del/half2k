//! A second search, transcribed rather than adapted.
//!
//! This exists to answer one question that no amount of reading can settle:
//! when two programs have the same ideas and the same constants and one is a
//! hundred Elo behind, is the difference in the search or in everything else?
//!
//! The search in `search.rs` was written from an understanding of a reference
//! and then given that reference's numbers, which is not the same as being that
//! reference. Its move ordering, its reduction formula, its history shape and
//! its aspiration all differ in structure. So this module transcribes the
//! reference faithfully instead -- same staging, same tables, same formulas --
//! on top of OUR board and OUR network.
//!
//! What each outcome would mean. If this reaches the reference's strength, the
//! difference was the search, and there is now a working version to compare
//! against line by line. If this falls short too, the difference is everything
//! underneath -- move generation, accumulator, node rate -- and no amount of
//! search work would have found it.
//!
//! It is selected by a UCI option, so the comparison is one binary, one board,
//! one network, and only the search changing.

use crate::board::Board;
use crate::moves::{Move, MoveFlag};
use crate::types::*;

/// Killers per ply, as the reference keeps them.
pub const NUM_KILLERS: usize = 3;
/// Continuation tables, indexed by the opponent's last three moves.
pub const NUM_CONTINUATION: usize = 3;

/// What one cutoff is worth. Linear, capped.
#[inline]
pub fn hist_bonus(depth: i32) -> i32 {
    (200 * depth).min(4000)
}

/// Move an entry towards a ceiling it approaches but never passes.
#[inline]
fn saturate_add(entry: &mut i32, bonus: i32, max: i32) {
    *entry += bonus - *entry * bonus.abs() / max;
}

const BUTTERFLY_MAX: i32 = 15000;
const PIECE_TO_MAX: i32 = 30000;

/// `[from][to]`, per side.
#[derive(Clone)]
pub struct Butterfly {
    data: Vec<i32>,
}

impl Butterfly {
    fn new() -> Self {
        Butterfly { data: vec![0; 64 * 64] }
    }
    #[inline]
    fn get(&self, from: Square, to: Square) -> i32 {
        self.data[from as usize * 64 + to as usize]
    }
    #[inline]
    fn add(&mut self, from: Square, to: Square, bonus: i32) {
        saturate_add(&mut self.data[from as usize * 64 + to as usize], bonus, BUTTERFLY_MAX);
    }
    fn clear(&mut self) {
        self.data.iter_mut().for_each(|v| *v = 0);
    }
}

/// `[piece][to]`, one of these per (from, to) of an earlier move.
#[derive(Clone)]
pub struct PieceTo {
    data: Vec<i32>,
}

impl PieceTo {
    fn new() -> Self {
        PieceTo { data: vec![0; 6 * 64] }
    }
    #[inline]
    fn get(&self, piece: usize, to: Square) -> i32 {
        self.data[piece * 64 + to as usize]
    }
    #[inline]
    fn add(&mut self, piece: usize, to: Square, bonus: i32) {
        saturate_add(&mut self.data[piece * 64 + to as usize], bonus, PIECE_TO_MAX);
    }
    fn clear(&mut self) {
        self.data.iter_mut().for_each(|v| *v = 0);
    }
}

/// Everything the ordering remembers.
///
/// The continuation tables are indexed by the moves at offsets 0, 2 and 4 in
/// the move list -- which, because sides alternate, are all the OPPONENT's
/// moves at increasing distance. That is the reference's choice and it is
/// deliberately kept: a quiet move is good or bad largely in reply to what the
/// opponent has been doing, and mixing our own moves into the same tables
/// answers a different question.
pub struct Histories {
    continuation: Vec<PieceTo>,
    main: [Butterfly; 2],
    killers: Vec<[Option<Move>; NUM_KILLERS]>,
}

impl Histories {
    pub fn new(max_ply: usize) -> Self {
        Histories {
            continuation: vec![PieceTo::new(); 64 * 64],
            main: [Butterfly::new(), Butterfly::new()],
            killers: vec![[None; NUM_KILLERS]; max_ply],
        }
    }

    pub fn clear(&mut self) {
        self.continuation.iter_mut().for_each(|t| t.clear());
        self.main.iter_mut().for_each(|t| t.clear());
        self.killers.iter_mut().for_each(|k| *k = [None; NUM_KILLERS]);
    }

    /// Which continuation tables are in reach, given the moves played.
    ///
    /// `played` holds the move made at each ply. Offsets 0, 2 and 4 back from
    /// the current ply are what the reference reads.
    #[inline]
    pub fn cont_slots(played: &[Option<Move>], ply: usize) -> [Option<usize>; NUM_CONTINUATION] {
        let mut out = [None; NUM_CONTINUATION];
        for (i, back) in [1usize, 3, 5].iter().enumerate() {
            if ply >= *back {
                if let Some(m) = played[ply - back] {
                    out[i] = Some(m.from as usize * 64 + m.to as usize);
                }
            }
        }
        out
    }

    #[inline]
    pub fn quiet_score(
        &self,
        board: &Board,
        mv: Move,
        slots: &[Option<usize>; NUM_CONTINUATION],
    ) -> i32 {
        let side = board.side.idx();
        let piece = match board.piece_at(mv.from) {
            Some((pt, _)) => pt.idx(),
            None => return 0,
        };
        // The reply carries double, exactly as the reference weights it.
        let mut s = self.main[side].get(mv.from, mv.to);
        for (i, slot) in slots.iter().enumerate() {
            if let Some(idx) = slot {
                let w = if i == 0 { 2 } else { 1 };
                s += w * self.continuation[*idx].get(piece, mv.to);
            }
        }
        s
    }

    #[inline]
    pub fn add_bonus(
        &mut self,
        board: &Board,
        mv: Move,
        piece: usize,
        slots: &[Option<usize>; NUM_CONTINUATION],
        bonus: i32,
    ) {
        let side = board.side.idx();
        self.main[side].add(mv.from, mv.to, bonus);
        for slot in slots.iter().flatten() {
            self.continuation[*slot].add(piece, mv.to, bonus);
        }
    }

    /// A move that caused a cutoff: credited, and remembered as a killer.
    pub fn bestmove(
        &mut self,
        board: &Board,
        mv: Move,
        piece: usize,
        slots: &[Option<usize>; NUM_CONTINUATION],
        ply: usize,
        depth: i32,
    ) {
        self.add_bonus(board, mv, piece, slots, hist_bonus(depth));
        if self.killers[ply].iter().any(|k| *k == Some(mv)) {
            return;
        }
        for i in (1..NUM_KILLERS).rev() {
            self.killers[ply][i] = self.killers[ply][i - 1];
        }
        self.killers[ply][0] = Some(mv);
    }

    #[inline]
    pub fn killers(&self, ply: usize) -> [Option<Move>; NUM_KILLERS] {
        self.killers[ply]
    }
}

/// The stages a move can come from, in the order they are tried.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    Hash,
    CapturesInit,
    Captures,
    CapturesEnd,
    Killers,
    QuietInit,
    Quiet,
    BadCapturesInit,
    BadCaptures,
    Done,
}

/// Hands out moves one at a time, generating only what is asked for.
///
/// The point of the staging is not tidiness. A node that cuts off on the stored
/// move or on a winning capture never generates a quiet move at all, and never
/// looks up a history score for one. Generating everything and sorting it, as
/// the other search does, pays that cost at every node whether or not it is
/// used.
pub struct MoveOrder {
    stage: Stage,
    hash_move: Option<Move>,
    killer: Option<Move>,
    depth: i32,
    quiescence: bool,

    captures: Vec<Move>,
    quiets: Vec<Move>,
    /// Captures that failed the exchange test, kept for the last stage.
    bad: Vec<Move>,
    idx: usize,
}

impl MoveOrder {
    pub fn new(hash_move: Option<Move>, depth: i32, quiescence: bool) -> Self {
        MoveOrder {
            stage: Stage::Hash,
            hash_move,
            killer: None,
            depth,
            quiescence,
            captures: Vec::new(),
            quiets: Vec::new(),
            bad: Vec::new(),
            idx: 0,
        }
    }

    /// Most valuable victim, as the reference orders captures -- the attacker
    /// does not enter, because the exchange test below already decides whether
    /// the capture is any good.
    #[inline]
    fn capture_score(board: &Board, mv: Move) -> i32 {
        let victim = if mv.flag == MoveFlag::EnPassant {
            PieceType::Pawn
        } else {
            match board.piece_at(mv.to) {
                Some((pt, _)) => pt,
                None => PieceType::Pawn,
            }
        };
        victim.value()
    }

    /// Insertion sort that leaves everything below `threshold` where it is.
    ///
    /// Sorting the tail is work for moves that will mostly never be reached.
    fn partial_sort(list: &mut [Move], scores: &mut [i32], threshold: i32) {
        for i in 1..list.len() {
            if scores[i] <= threshold {
                continue;
            }
            let (m, sc) = (list[i], scores[i]);
            let mut j = i;
            while j > 0 && scores[j - 1] < sc {
                list[j] = list[j - 1];
                scores[j] = scores[j - 1];
                j -= 1;
            }
            list[j] = m;
            scores[j] = sc;
        }
    }
}
impl MoveOrder {
    /// Load the move list and split it, then hand moves out by stage.
    ///
    /// The reference generates captures and quiets in separate passes and never
    /// generates the second at a node that cuts off in the first. This takes the
    /// whole list and partitions it instead, which costs that saving but keeps
    /// the ORDER identical -- and the order is the thing being tested. Doing it
    /// the other way needs a quiet-only generator, and putting a new path
    /// through move generation into an experiment about move ordering is how a
    /// result stops meaning anything.
    pub fn load(&mut self, board: &Board, moves: &[Move]) {
        for mv in moves {
            if mv.is_capture() || mv.promotion.is_some() {
                self.captures.push(*mv);
            } else {
                self.quiets.push(*mv);
            }
        }
        let mut scores: Vec<i32> =
            self.captures.iter().map(|m| Self::capture_score(board, *m)).collect();
        Self::partial_sort(&mut self.captures, &mut scores, 0);
    }

    pub fn next_move(
        &mut self,
        board: &Board,
        atk: &crate::attacks::Attacks,
        hist: &Histories,
        slots: &[Option<usize>; NUM_CONTINUATION],
        ply: usize,
    ) -> Option<Move> {
        loop {
            match self.stage {
                Stage::Hash => {
                    self.stage = Stage::CapturesInit;
                    if let Some(m) = self.hash_move {
                        if self.captures.contains(&m) || self.quiets.contains(&m) {
                            return Some(m);
                        }
                    }
                }
                Stage::CapturesInit => {
                    self.stage = Stage::Captures;
                    self.idx = 0;
                }
                Stage::Captures => {
                    // Only the ones that survive the exchange test go now. The
                    // bar falls with depth: deep in the tree there is room to
                    // find out whether a slightly losing capture actually is.
                    while self.idx < self.captures.len() {
                        let m = self.captures[self.idx];
                        self.idx += 1;
                        if Some(m) == self.hash_move {
                            continue;
                        }
                        let bar = (-50 * (self.depth - 1)).max(-250);
                        if crate::see::see_ge(atk, board, &m, bar) {
                            return Some(m);
                        }
                        self.bad.push(m);
                    }
                    self.stage = Stage::CapturesEnd;
                }
                Stage::CapturesEnd => {
                    // Quiescence stops here unless in check, where every move is
                    // a reply to the check and none of them are optional.
                    if self.quiescence && !board.in_check(board.side, atk) {
                        return None;
                    }
                    self.stage = Stage::Killers;
                }
                Stage::Killers => {
                    self.stage = Stage::QuietInit;
                    for k in hist.killers(ply) {
                        if let Some(m) = k {
                            if Some(m) != self.hash_move && self.quiets.contains(&m) {
                                self.killer = Some(m);
                                return Some(m);
                            }
                        }
                    }
                }
                Stage::QuietInit => {
                    self.stage = Stage::Quiet;
                    let mut scores: Vec<i32> = self
                        .quiets
                        .iter()
                        .map(|m| hist.quiet_score(board, *m, slots))
                        .collect();
                    Self::partial_sort(&mut self.quiets, &mut scores, -1000 * self.depth);
                    self.idx = 0;
                }
                Stage::Quiet => {
                    while self.idx < self.quiets.len() {
                        let m = self.quiets[self.idx];
                        self.idx += 1;
                        if Some(m) != self.hash_move && Some(m) != self.killer {
                            return Some(m);
                        }
                    }
                    self.stage = Stage::BadCapturesInit;
                }
                Stage::BadCapturesInit => {
                    self.stage = Stage::BadCaptures;
                    self.idx = 0;
                }
                Stage::BadCaptures => {
                    while self.idx < self.bad.len() {
                        let m = self.bad[self.idx];
                        self.idx += 1;
                        if Some(m) != self.hash_move {
                            return Some(m);
                        }
                    }
                    self.stage = Stage::Done;
                }
                Stage::Done => return None,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The transcribed search itself.
//
// Structure and constants follow the reference; the plumbing around it -- the
// table, the clock, the principal variation, repetition detection -- is shared
// with the other search on purpose, so that an A/B between them changes the
// search and nothing else.
// ---------------------------------------------------------------------------

use crate::search::{
    is_mate, score_from_tt, score_to_tt, Searcher, INF, MATE_IN_MAX, MAX_PLY,
};
use crate::tt::{Bound, TT_EVAL_NONE};

impl Searcher {
    /// One node of the transcribed search.
    pub fn negamax_ref(
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
            return self.quiescence_ref(board, alpha, beta, ply);
        }

        self.nodes += 1;
        if self.out_of_time() {
            return 0;
        }

        let root = ply == 0;
        let in_check = board.in_check(board.side, &self.atk);
        let excluded = self.excluded[ply];

        if !root {
            if self.is_draw(board) {
                return 0;
            }
            if ply >= MAX_PLY - 1 {
                return crate::search::debug_eval(board, false);
            }
            let a = alpha.max(-crate::search::MATE + ply as i32);
            let b = beta.min(crate::search::MATE - ply as i32 - 1);
            if a >= b {
                return a;
            }
            alpha = a;
        }

        let slots = crate::refsearch::Histories::cont_slots(&self.played_moves, ply);

        let entry = if excluded.is_some() {
            None
        } else {
            self.tt.probe(board.hash)
        };
        let mut tt_move = None;
        let mut tt_pv = pv_node;
        let mut tt_score = TT_EVAL_NONE as i32;
        let mut tt_depth = 0;
        let mut tt_bound = Bound::NoBound;
        if let Some(e) = entry {
            tt_move = e.best;
            tt_pv |= e.pv;
            tt_depth = e.depth;
            tt_bound = e.bound;
            if e.has_bound() {
                tt_score = score_from_tt(e.score, ply);
                let usable = match e.bound {
                    Bound::Exact => true,
                    Bound::Lower => tt_score >= beta,
                    Bound::Upper => tt_score <= alpha,
                    Bound::NoBound => false,
                };
                if !pv_node && tt_depth >= depth && usable && board.halfmove < 90 {
                    return tt_score;
                }
            }
        }

        // The static evaluation, then the same value improved by anything the
        // table already established.
        let raw = if in_check {
            TT_EVAL_NONE as i32
        } else {
            match entry {
                Some(e) if e.static_eval != TT_EVAL_NONE => e.static_eval as i32,
                _ => {
                    let e = crate::search::debug_eval(board, false);
                    self.tt.store_eval_only(board.hash, e as i16);
                    e
                }
            }
        };
        let mut static_eval = raw;
        if !in_check && tt_score != TT_EVAL_NONE as i32 {
            let better = match tt_bound {
                Bound::Exact => true,
                Bound::Lower => tt_score > static_eval,
                Bound::Upper => tt_score < static_eval,
                Bound::NoBound => false,
            };
            if better {
                static_eval = tt_score;
            }
        }
        self.eval_stack[ply] = raw;
        let improving = !in_check
            && ply >= 2
            && self.eval_stack[ply - 2] != TT_EVAL_NONE as i32
            && raw > self.eval_stack[ply - 2];

        // Reverse futility.
        if !pv_node && depth < 9 && !in_check && excluded.is_none() && static_eval.abs() < MATE_IN_MAX
        {
            let margin = 150 * (depth - improving as i32);
            if static_eval - margin >= beta {
                return static_eval;
            }
        }

        // Null move. The reduction grows with how far above beta we already
        // are, not with depth.
        if !pv_node
            && !in_check
            && excluded.is_none()
            && static_eval >= beta
            && static_eval >= raw
            && raw >= beta - 20 * depth - 40 * improving as i32 + 100
            && crate::search::has_pieces_pub(board, board.side)
            && !(ply > 0 && self.null_at[ply - 1])
        {
            let r = 4 + ((static_eval - beta) / 200).min(6);
            let nd = (depth - r).max(0);
            let undo = board.make_null_move();
            self.keys.push(board.hash);
            self.null_at[ply] = true;
            let score = -self.negamax_ref(board, nd, -beta, -beta + 1, ply + 1, false, !cut_node);
            self.null_at[ply] = false;
            self.keys.pop();
            board.unmake_null_move(&undo);
            if score >= beta {
                return if is_mate(score) { beta } else { score };
            }
        }

        // Nothing usable stored at a principal variation node: search shallower
        // and come back rather than pay full depth to discover a first move.
        if !root && pv_node && !in_check && tt_move.is_none() {
            depth -= 2;
        }
        if depth <= 0 {
            return self.quiescence_ref(board, alpha, beta, ply);
        }

        let legal = crate::movegen::generate_legal(board, &self.atk);
        let mut picker = crate::refsearch::MoveOrder::new(tt_move, depth, false);
        picker.load(board, &legal);

        let alpha_orig = alpha;
        let mut best_score = -INF;
        let mut best_move = None;
        let mut quiets_searched: Vec<Move> = Vec::new();
        let mut n_moves = 0;

        while let Some(mv) = {
            let hist = &self.ref_hist;
            picker.next_move(board, &self.atk, hist, &slots, ply)
        } {
            if Some(mv) == excluded {
                continue;
            }
            n_moves += 1;
            let mut extension = 0i32;
            let is_quiet = !mv.is_capture() && mv.promotion.is_none();
            let move_score = if mv.is_capture() {
                0
            } else {
                self.ref_hist.quiet_score(board, mv, &slots)
            };

            // Shallow pruning, and none of it without pieces on the board.
            if !root
                && crate::search::has_pieces_pub(board, board.side)
                && best_score > -MATE_IN_MAX
            {
                if mv.is_capture() || mv.promotion.is_some() {
                    if depth < 10 && !crate::see::see_ge(&self.atk, board, &mv, -140 * depth) {
                        continue;
                    }
                } else {
                    if depth < 7 && n_moves > 3 + depth * depth {
                        continue;
                    }
                    if move_score < -600 * depth * depth {
                        continue;
                    }
                    if !in_check
                        && depth < 12
                        && raw != TT_EVAL_NONE as i32
                        && raw + 100 + 150 * depth + move_score / 75 < alpha
                    {
                        continue;
                    }
                    if depth < 10
                        && !crate::see::see_ge(
                            &self.atk,
                            board,
                            &mv,
                            -10 * (depth + depth * depth),
                        )
                    {
                        continue;
                    }
                }
            }

            // Singular extension.
            if !root
                && entry.is_some()
                && depth > 4
                && excluded.is_none()
                && Some(mv) == tt_move
                && tt_depth >= depth - 3
                && matches!(tt_bound, Bound::Lower | Bound::Exact)
                && !is_mate(tt_score)
            {
                let target = tt_score - 2 * depth;
                let sd = (depth - 1) / 2;
                self.excluded[ply] = Some(mv);
                let s = self.negamax_ref(board, sd, target - 1, target, ply, false, cut_node);
                self.excluded[ply] = None;
                if self.stopped {
                    return 0;
                }
                if s < target {
                    extension = 1;
                    if !pv_node && s < target - 40 && (ply as i32) < depth {
                        extension = 2;
                    }
                } else if !pv_node && target >= beta {
                    return target;
                }
            }

            let piece = board.piece_at(mv.from).map(|(pt, _)| pt.idx()).unwrap_or(0);
            let capture_or_promo = mv.is_capture() || mv.promotion.is_some();
            self.played_moves[ply] = Some(mv);
            let undo = board.make_move(&mv);
            self.tt.prefetch(board.hash);
            self.keys.push(board.hash);

            // A move that gives check, while the position is still a contest.
            if board.in_check(board.side, &self.atk)
                && raw != TT_EVAL_NONE as i32
                && raw.abs() > 75
            {
                extension = 1;
            }
            let curr_depth = depth + extension;

            let mut score = -INF;
            let mut full = !(pv_node && n_moves == 1);
            let mut did_lmr = false;
            if depth > 2 && n_moves > 1 + 2 * root as i32 && (!pv_node || !capture_or_promo) {
                did_lmr = true;
                let r = ilog2(n_moves) / 2 + ilog2(depth) / 2
                    - (capture_or_promo || pv_node) as i32
                    - (move_score + 15000) / 30000
                    + 2 * cut_node as i32
                    + 1;
                let nd = (curr_depth - r - 1).clamp(0, curr_depth - 1);
                score = -self.negamax_ref(board, nd, -alpha - 1, -alpha, ply + 1, false, true);
                full = score > alpha && r > 0;
            }
            if full {
                score = -self.negamax_ref(
                    board,
                    curr_depth - 1,
                    -alpha - 1,
                    -alpha,
                    ply + 1,
                    false,
                    !cut_node,
                );
            }
            if pv_node && (n_moves == 1 || (score > alpha && (root || score < beta))) {
                score = -self.negamax_ref(board, curr_depth - 1, -beta, -alpha, ply + 1, true, false);
            }

            self.keys.pop();
            board.unmake_move(&mv, &undo);

            if self.stopped {
                return 0;
            }

            if did_lmr && full {
                let bonus = if score > best_score {
                    crate::refsearch::hist_bonus(depth)
                } else {
                    -crate::refsearch::hist_bonus(depth)
                };
                self.ref_hist.add_bonus(board, mv, piece, &slots, bonus);
            }

            if score > best_score {
                best_score = score;
                if score > alpha {
                    alpha = score;
                    best_move = Some(mv);
                    self.update_pv_pub(ply, mv);
                    if alpha >= beta {
                        break;
                    }
                }
            }
            if is_quiet {
                quiets_searched.push(mv);
            }
        }

        if let Some(bm) = best_move {
            if !bm.is_capture() && bm.promotion.is_none() {
                let piece = board.piece_at(bm.from).map(|(pt, _)| pt.idx()).unwrap_or(0);
                self.ref_hist.bestmove(board, bm, piece, &slots, ply, depth);
                let malus = -crate::refsearch::hist_bonus(depth);
                for q in &quiets_searched {
                    if *q != bm {
                        let p = board.piece_at(q.from).map(|(pt, _)| pt.idx()).unwrap_or(0);
                        self.ref_hist.add_bonus(board, *q, p, &slots, malus);
                    }
                }
            }
        }

        if n_moves == 0 {
            if excluded.is_some() {
                return alpha;
            }
            return if in_check { -crate::search::MATE + ply as i32 } else { 0 };
        }

        let bound = if best_score >= beta {
            Bound::Lower
        } else if pv_node && best_score > alpha_orig {
            Bound::Exact
        } else {
            Bound::Upper
        };
        if excluded.is_none() {
            self.tt.store(
                board.hash,
                depth,
                score_to_tt(best_score, ply),
                bound,
                best_move,
                tt_pv,
                if in_check { TT_EVAL_NONE } else { raw as i16 },
            );
        }
        best_score
    }

    /// Quiescence, transcribed alongside it.
    pub fn quiescence_ref(&mut self, board: &mut Board, mut alpha: i32, beta: i32, ply: usize) -> i32 {
        self.nodes += 1;
        if self.out_of_time() {
            return 0;
        }
        if ply >= MAX_PLY - 1 {
            return crate::search::debug_eval(board, false);
        }
        if self.is_draw(board) {
            return 0;
        }
        let in_check = board.in_check(board.side, &self.atk);
        let a = alpha.max(-crate::search::MATE + ply as i32);
        let b = beta.min(crate::search::MATE - ply as i32 - 1);
        if a >= b {
            return a;
        }
        alpha = a;

        let entry = self.tt.probe(board.hash);
        let mut tt_move = None;
        let mut tt_score = TT_EVAL_NONE as i32;
        let mut tt_bound = Bound::NoBound;
        if let Some(e) = entry {
            tt_move = e.best;
            tt_bound = e.bound;
            if e.has_bound() {
                tt_score = score_from_tt(e.score, ply);
                let usable = match e.bound {
                    Bound::Exact => true,
                    Bound::Lower => tt_score >= beta,
                    Bound::Upper => tt_score <= alpha,
                    Bound::NoBound => false,
                };
                if usable {
                    return tt_score;
                }
            }
            if !in_check && tt_move.is_some_and(|m| !m.is_capture()) {
                tt_move = None;
            }
        }

        let mut raw = TT_EVAL_NONE as i32;
        let mut best_score = -INF;
        if !in_check {
            raw = match entry {
                Some(e) if e.static_eval != TT_EVAL_NONE => e.static_eval as i32,
                _ => crate::search::debug_eval(board, false),
            };
            best_score = raw;
            if tt_score != TT_EVAL_NONE as i32 {
                let better = match tt_bound {
                    Bound::Exact => true,
                    Bound::Lower => tt_score > raw,
                    Bound::Upper => tt_score < raw,
                    Bound::NoBound => false,
                };
                if better {
                    best_score = tt_score;
                }
            }
            alpha = alpha.max(best_score);
            if alpha >= beta {
                return alpha;
            }
        }

        let legal = crate::movegen::generate_legal(board, &self.atk);
        let slots = crate::refsearch::Histories::cont_slots(&self.played_moves, ply);
        let mut picker = crate::refsearch::MoveOrder::new(tt_move, 0, true);
        picker.load(board, &legal);

        let alpha_orig = alpha;
        let mut best_move = None;
        let mut n = 0;
        while let Some(mv) = {
            let hist = &self.ref_hist;
            picker.next_move(board, &self.atk, hist, &slots, ply)
        } {
            n += 1;
            if !in_check && !crate::see::see_ge(&self.atk, board, &mv, 0) {
                continue;
            }
            self.played_moves[ply] = Some(mv);
            let undo = board.make_move(&mv);
            self.tt.prefetch(board.hash);
            self.keys.push(board.hash);
            let score = -self.quiescence_ref(board, -beta, -alpha, ply + 1);
            self.keys.pop();
            board.unmake_move(&mv, &undo);
            if self.stopped {
                return 0;
            }
            if score > best_score {
                best_score = score;
                if score > alpha {
                    alpha = score;
                    best_move = Some(mv);
                    if alpha >= beta {
                        break;
                    }
                }
            }
        }

        if n == 0 && in_check {
            return -crate::search::MATE + ply as i32;
        }

        let bound = if best_score >= beta {
            Bound::Lower
        } else if best_score > alpha_orig {
            Bound::Exact
        } else {
            Bound::Upper
        };
        self.tt.store(
            board.hash,
            0,
            score_to_tt(best_score, ply),
            bound,
            best_move,
            false,
            if in_check { TT_EVAL_NONE } else { raw as i16 },
        );
        best_score
    }
}

/// Floor of the base two logarithm, as the reference's reduction formula uses.
#[inline]
fn ilog2(v: i32) -> i32 {
    if v <= 0 {
        0
    } else {
        31 - (v as u32).leading_zeros() as i32
    }
}
