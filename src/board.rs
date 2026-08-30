use crate::attacks::*;
use crate::bitboard::*;
use crate::moves::*;
use crate::types::*;

pub const CASTLE_WK: u8 = 1;
pub const CASTLE_WQ: u8 = 2;
pub const CASTLE_BK: u8 = 4;
pub const CASTLE_BQ: u8 = 8;

#[derive(Clone)]
pub struct Board {
    pub pieces: [[Bitboard; 6]; 2], // [color][piece_type]
    pub occ_color: [Bitboard; 2],
    pub occ_all: Bitboard,
    pub side: Color,
    /// The network accumulator, carried with the position.
    ///
    /// Here rather than in the searcher because every path that changes a piece
    /// already goes through `add_piece` and `remove_piece`. Anywhere else would
    /// mean finding and patching castling, en passant and promotion separately,
    /// and one missed case is an evaluation that is quietly wrong only in rare
    /// positions. `None` until a network is installed, so nothing pays for it
    /// before there is one.
    pub acc: Option<Box<crate::nnue::Accumulator>>,
    pub castling: u8,
    pub ep_square: Square,
    pub halfmove: u32,
    pub fullmove: u32,
    /// Zobrist key for this position, kept up to date move by move.
    ///
    /// Maintained here rather than recomputed by the search because the
    /// recompute walks all 32 pieces and the search wants a key at every
    /// node. See `zobrist::Zobrist::hash_completo` for why this only became
    /// worth doing once king moves stopped needing a make/unmake.
    pub hash: u64,
    // Mailbox O(1) -- piece_at() fazia uma varredura ate' 12 bitboards
    // (2 cores x 6 tipos) a cada chamada; era uma fatia real do tempo
    // total dentro de make_move/unmake_move (ver perf), alem de ser
    // usado em SEE. Mantido em sincronia por add_piece()/remove_piece().
    pub mailbox: [Option<(PieceType, Color)>; 64],
}

#[derive(Copy, Clone)]
pub struct Undo {
    pub captured: Option<(PieceType, Color)>,
    pub castling: u8,
    pub ep_square: Square,
    pub halfmove: u32,
    /// A whole snapshot rather than deltas: restoring it in `unmake_move` is
    /// correct by construction, with no need to reverse each capture,
    /// promotion and castling individually. It is also what makes the
    /// incremental key cheap -- undoing a move restores one u64 instead of
    /// replaying every XOR backwards, so only `make_move` ever pays.
    pub hash: u64,
}

/// Undo minimo para um null move (passar a vez): so' muda side + ep_square.
#[derive(Copy, Clone)]
pub struct NullUndo {
    pub ep_square: Square,
    pub hash: u64,
}

impl Board {
    pub fn startpos() -> Self {
        Self::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    }

    pub fn from_fen(fen: &str) -> Self {
        let parts: Vec<&str> = fen.split_whitespace().collect();
        let mut pieces = [[0u64; 6]; 2];
        let mut mailbox: [Option<(PieceType, Color)>; 64] = [None; 64];
        let mut rank = 7i32;
        let mut file = 0i32;
        for ch in parts[0].chars() {
            match ch {
                '/' => {
                    rank -= 1;
                    file = 0;
                }
                '1'..='8' => {
                    file += ch.to_digit(10).unwrap() as i32;
                }
                c => {
                    let color = if c.is_ascii_uppercase() { Color::White } else { Color::Black };
                    let kind = match c.to_ascii_lowercase() {
                        'p' => PieceType::Pawn,
                        'n' => PieceType::Knight,
                        'b' => PieceType::Bishop,
                        'r' => PieceType::Rook,
                        'q' => PieceType::Queen,
                        'k' => PieceType::King,
                        _ => panic!("fen piece invalido: {}", c),
                    };
                    let s = sq(file as u8, rank as u8);
                    pieces[color.idx()][kind.idx()] |= bb(s);
                    mailbox[s as usize] = Some((kind, color));
                    file += 1;
                }
            }
        }
        let side = if parts.get(1) == Some(&"b") { Color::Black } else { Color::White };
        let mut castling = 0u8;
        if let Some(c) = parts.get(2) {
            if c.contains('K') {
                castling |= CASTLE_WK;
            }
            if c.contains('Q') {
                castling |= CASTLE_WQ;
            }
            if c.contains('k') {
                castling |= CASTLE_BK;
            }
            if c.contains('q') {
                castling |= CASTLE_BQ;
            }
        }
        let ep_square = match parts.get(3) {
            Some(s) if *s != "-" => parse_sq(s),
            _ => NO_SQUARE,
        };
        let halfmove = parts.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
        let fullmove = parts.get(5).and_then(|s| s.parse().ok()).unwrap_or(1);

        let mut b = Board {
            pieces,
            occ_color: [0, 0],
            occ_all: 0,
            acc: None,
            side,
            castling,
            ep_square,
            halfmove,
            fullmove,
            hash: 0,
            mailbox,
        };
        b.hash = crate::zobrist::tabelas().hash_completo(&b);
        b.recompute_occ();
        // Built once here from the finished position. Every later change goes
        // through add_piece/remove_piece, one piece at a time.
        if let Some(net) = crate::nnue::net() {
            let kings = [b.king_sq(Color::White), b.king_sq(Color::Black)];
            b.acc = Some(Box::new(crate::nnue::Accumulator::fresh(net, &b.pieces, kings)));
        }
        b
    }

    /// Repovoa o mailbox a partir dos bitboards. So' e' preciso quando um
    /// tabuleiro e' montado de fora (ler bitboards de um ficheiro, por
    /// exemplo) em vez de vir do from_fen ou de um make_move.
    pub fn rebuild_mailbox(&mut self) {
        self.mailbox = [None; 64];
        for c in [Color::White, Color::Black] {
            for pt in ALL_PIECES {
                let mut bb = self.pieces[c.idx()][pt.idx()];
                while bb != 0 {
                    let s = bb.trailing_zeros() as usize;
                    bb &= bb - 1;
                    self.mailbox[s] = Some((pt, c));
                }
            }
        }
    }

    pub fn recompute_occ(&mut self) {
        for c in [Color::White, Color::Black] {
            let mut o = 0u64;
            for pt in ALL_PIECES {
                o |= self.pieces[c.idx()][pt.idx()];
            }
            self.occ_color[c.idx()] = o;
        }
        self.occ_all = self.occ_color[0] | self.occ_color[1];
    }

    #[inline]
    #[inline(always)]
    pub fn piece_at(&self, s: Square) -> Option<(PieceType, Color)> {
        self.mailbox[s as usize]
    }

    pub fn king_sq(&self, color: Color) -> Square {
        self.pieces[color.idx()][PieceType::King.idx()].trailing_zeros() as Square
    }

    pub fn is_square_attacked(&self, s: Square, by: Color, atk: &Attacks) -> bool {
        let occ = self.occ_all;
        // pawns: a pawn of `by` attacks `s` if s is in the pawn-attack set
        // of `by`'s color computed FROM s using the opposite color table
        // (symmetry trick: attacker squares = pawn_attacks[opp(by)][s] intersected with by's pawns)
        if atk.pawn[by.opp().idx()][s as usize] & self.pieces[by.idx()][PieceType::Pawn.idx()] != 0 {
            return true;
        }
        if atk.knight[s as usize] & self.pieces[by.idx()][PieceType::Knight.idx()] != 0 {
            return true;
        }
        if atk.king[s as usize] & self.pieces[by.idx()][PieceType::King.idx()] != 0 {
            return true;
        }
        let bishops_queens = self.pieces[by.idx()][PieceType::Bishop.idx()]
            | self.pieces[by.idx()][PieceType::Queen.idx()];
        if bishop_attacks(s, occ) & bishops_queens != 0 {
            return true;
        }
        let rooks_queens = self.pieces[by.idx()][PieceType::Rook.idx()]
            | self.pieces[by.idx()][PieceType::Queen.idx()];
        if rook_attacks(s, occ) & rooks_queens != 0 {
            return true;
        }
        false
    }

    /// Would our king be in check on `to`, having come from `from`?
    ///
    /// Answers what `generate_legal` used to answer with a make/unmake, and
    /// the two differences from a plain `is_square_attacked(to, ...)` are
    /// exactly what made the naive version wrong:
    ///
    /// - **The king is lifted off `from` first.** Standing on the board it
    ///   blocks the very ray it would be fleeing along, so a king running
    ///   directly away from a rook or bishop would look safe on the new
    ///   square when it is still checked.
    /// - **A piece captured on `to` stops attacking.** It is removed from the
    ///   attacker sets, not merely stepped over -- otherwise capturing the
    ///   checker would look illegal because the dead piece still "attacks"
    ///   its own square.
    ///
    /// King moves are the largest group `generate_legal` could not settle
    /// without a make/unmake (up to eight per node, every node, even when not
    /// in check), and make/unmake is expensive here because it drags the
    /// board, the mailbox and the accumulator with it.
    pub fn king_move_leaves_check(&self, from: Square, to: Square, atk: &Attacks) -> bool {
        let by = self.side.opp();
        let occ = (self.occ_all & !bb(from)) | bb(to);
        // The captured piece, if any, is gone -- so drop `to` from every
        // attacker set below rather than only from the occupancy.
        let inimigos = !bb(to);
        if atk.pawn[by.opp().idx()][to as usize]
            & self.pieces[by.idx()][PieceType::Pawn.idx()]
            & inimigos
            != 0
        {
            return true;
        }
        if atk.knight[to as usize] & self.pieces[by.idx()][PieceType::Knight.idx()] & inimigos != 0 {
            return true;
        }
        if atk.king[to as usize] & self.pieces[by.idx()][PieceType::King.idx()] & inimigos != 0 {
            return true;
        }
        let bishops_queens = (self.pieces[by.idx()][PieceType::Bishop.idx()]
            | self.pieces[by.idx()][PieceType::Queen.idx()])
            & inimigos;
        if bishop_attacks(to, occ) & bishops_queens != 0 {
            return true;
        }
        let rooks_queens = (self.pieces[by.idx()][PieceType::Rook.idx()]
            | self.pieces[by.idx()][PieceType::Queen.idx()])
            & inimigos;
        if rook_attacks(to, occ) & rooks_queens != 0 {
            return true;
        }
        false
    }

    pub fn in_check(&self, color: Color, atk: &Attacks) -> bool {
        self.is_square_attacked(self.king_sq(color), color.opp(), atk)
    }

    pub(crate) fn remove_piece(&mut self, pt: PieceType, c: Color, s: Square) {
        self.pieces[c.idx()][pt.idx()] &= !bb(s);
        self.occ_color[c.idx()] &= !bb(s);
        self.occ_all &= !bb(s);
        self.mailbox[s as usize] = None;
        self.hash ^= crate::zobrist::tabelas().piece_sq[c.idx()][pt.idx()][s as usize];
        // This and `add_piece` are the only two places a piece ever appears on
        // or leaves a square -- castling, en passant and promotion all route
        // through them.
        if let (Some(a), Some(net)) = (self.acc.as_mut(), crate::nnue::net()) {
            a.update(net, c, pt, s, false);
        }
    }
    fn add_piece(&mut self, pt: PieceType, c: Color, s: Square) {
        self.pieces[c.idx()][pt.idx()] |= bb(s);
        self.occ_color[c.idx()] |= bb(s);
        self.occ_all |= bb(s);
        self.mailbox[s as usize] = Some((pt, c));
        self.hash ^= crate::zobrist::tabelas().piece_sq[c.idx()][pt.idx()][s as usize];
        if let (Some(a), Some(net)) = (self.acc.as_mut(), crate::nnue::net()) {
            a.update(net, c, pt, s, true);
        }
    }

    /// Aplica um lance PSEUDO-LEGAL (a legalidade -- nao ficar em xeque --
    /// e' verificada por quem gera os lances, chamando in_check depois).
    pub fn make_move(&mut self, mv: &Move) -> Undo {
        let us = self.side;
        let them = us.opp();
        let (moving_pt, _) = self.piece_at(mv.from).expect("make_move: nada em from");
        let captured = if mv.flag == MoveFlag::EnPassant {
            Some((PieceType::Pawn, them))
        } else {
            self.piece_at(mv.to)
        };

        let undo = Undo {
            captured,
            castling: self.castling,
            ep_square: self.ep_square,
            halfmove: self.halfmove,
            hash: self.hash,
        };


        // remove captured piece (normal or en passant)
        match mv.flag {
            MoveFlag::EnPassant => {
                let cap_sq = if us == Color::White { mv.to - 8 } else { mv.to + 8 };
                self.remove_piece(PieceType::Pawn, them, cap_sq);
            }
            _ => {
                if let Some((cpt, cc)) = captured {
                    self.remove_piece(cpt, cc, mv.to);
                }
            }
        }

        // move the piece
        self.remove_piece(moving_pt, us, mv.from);
        let final_pt = mv.promotion.unwrap_or(moving_pt);
        self.add_piece(final_pt, us, mv.to);

        // castling: move the rook too
        match mv.flag {
            MoveFlag::CastleKing => {
                let (rf, rt) = if us == Color::White { (7u8, 5u8) } else { (63u8, 61u8) };
                self.remove_piece(PieceType::Rook, us, rf);
                self.add_piece(PieceType::Rook, us, rt);
            }
            MoveFlag::CastleQueen => {
                let (rf, rt) = if us == Color::White { (0u8, 3u8) } else { (56u8, 59u8) };
                self.remove_piece(PieceType::Rook, us, rf);
                self.add_piece(PieceType::Rook, us, rt);
            }
            _ => {}
        }

        // en passant square update
        self.ep_square = if mv.flag == MoveFlag::DoublePush {
            if us == Color::White { mv.from + 8 } else { mv.from - 8 }
        } else {
            NO_SQUARE
        };

        // castling rights update
        if moving_pt == PieceType::King {
            if us == Color::White {
                self.castling &= !(CASTLE_WK | CASTLE_WQ);
            } else {
                self.castling &= !(CASTLE_BK | CASTLE_BQ);
            }
        }
        for s in [mv.from, mv.to] {
            match s {
                0 => self.castling &= !CASTLE_WQ,
                7 => self.castling &= !CASTLE_WK,
                56 => self.castling &= !CASTLE_BQ,
                63 => self.castling &= !CASTLE_BK,
                _ => {}
            }
        }

        // halfmove clock
        if moving_pt == PieceType::Pawn || captured.is_some() {
            self.halfmove = 0;
        } else {
            self.halfmove += 1;
        }
        if us == Color::Black {
            self.fullmove += 1;
        }

        self.side = them;
        // Everything a Zobrist key depends on that is NOT a piece on a
        // square. The pieces took care of themselves in add_piece/
        // remove_piece; these are the three pieces of state that also
        // belong in the key, and each is XORed out at its old value and in
        // at its new one. Castling rights are indexed by the whole 4-bit
        // mask rather than per-right, so one XOR pair covers any number of
        // rights lost at once.
        let z = crate::zobrist::tabelas();
        self.hash ^= z.side;
        if undo.castling != self.castling {
            self.hash ^= z.castling[(undo.castling & 0xF) as usize];
            self.hash ^= z.castling[(self.castling & 0xF) as usize];
        }
        if undo.ep_square != NO_SQUARE {
            self.hash ^= z.ep_file[file_of(undo.ep_square) as usize];
        }
        if self.ep_square != NO_SQUARE {
            self.hash ^= z.ep_file[file_of(self.ep_square) as usize];
        }
        // Only a king move can invalidate a perspective wholesale, so only a
        // king move needs to ask. Everything else was paying for the question.
        if moving_pt == PieceType::King {
            self.refresh_perspectives();
        }
        undo
    }

    /// Rebuild any perspective the last move invalidated wholesale.
    ///
    /// Empty until a network is wired in, but called from `make_move` AND from
    /// `unmake_move` from the start, deliberately.
    ///
    /// A king that crosses a bucket boundary, or the middle of the board where
    /// the mirror flips, changes the input number of every piece for its own
    /// perspective: the same piece on the same square is a different feature
    /// now. Everything the move did updated the accumulator under the OLD
    /// bucket, so that perspective has to be rebuilt rather than patched.
    ///
    /// The reason both call sites exist before there is anything to call: an
    /// earlier engine of ours had this in `make_move` only. Undoing a king move
    /// that crossed a boundary then left the accumulator describing the king on
    /// the right square under the wrong bucket, and any evaluation asked for in
    /// that window silently read weights for a king that is not there. Having
    /// the call site already here means the network cannot arrive and forget
    /// half of it.
    #[inline]
    fn refresh_perspectives(&mut self) {
        let net = match crate::nnue::net() {
            Some(n) => n,
            None => return,
        };
        // Everything the refresh needs is read BEFORE the accumulator is
        // borrowed, because it lives inside this same struct. Reading the board
        // through a copy taken around a mutable borrow is how an accumulator
        // ends up describing a position that no longer exists.
        let pieces = self.pieces;
        let kings = [self.king_sq(Color::White), self.king_sq(Color::Black)];
        if let Some(a) = self.acc.as_mut() {
            a.refresh(net, &pieces, kings);
        }
    }

    /// Passa a vez ao adversario sem mover peca (para null-move pruning).
    /// So' altera `side` e limpa `ep_square`; tudo o resto fica intacto.
    /// NUNCA chamar em xeque (o rei poderia ser "capturado" na resposta).
    pub fn make_null_move(&mut self) -> NullUndo {
        // A null move touches no piece, but it does advance a ply. Any
        // per-ply record a network keeps has to be written here too, as empty
        // -- leaving the previous move's record in place at this depth makes
        // the accumulator chain apply a move that never happened. Caught once
        // by the verifier at 16986 wrong entries in 1.5 million, all here.
        let undo = NullUndo { ep_square: self.ep_square, hash: self.hash };
        self.side = self.side.opp();
        self.ep_square = NO_SQUARE;
        let z = crate::zobrist::tabelas();
        self.hash ^= z.side;
        if undo.ep_square != NO_SQUARE {
            self.hash ^= z.ep_file[file_of(undo.ep_square) as usize];
        }
        undo
    }

    pub fn unmake_null_move(&mut self, undo: &NullUndo) {
        self.side = self.side.opp();
        self.ep_square = undo.ep_square;
        self.hash = undo.hash;
    }

    pub fn unmake_move(&mut self, mv: &Move, undo: &Undo) {
        let them = self.side; // side that is about to move again = the one who just moved's opponent... wait: after make_move, self.side = opponent of mover. So "us" (who made mv) = self.side.opp()
        let us = them.opp();
        self.side = us;

        let (final_pt, _) = self.piece_at(mv.to).expect("unmake: nada em to");
        let moving_pt = if mv.promotion.is_some() { PieceType::Pawn } else { final_pt };

        self.remove_piece(final_pt, us, mv.to);
        self.add_piece(moving_pt, us, mv.from);

        match mv.flag {
            MoveFlag::EnPassant => {
                let cap_sq = if us == Color::White { mv.to - 8 } else { mv.to + 8 };
                self.add_piece(PieceType::Pawn, us.opp(), cap_sq);
            }
            MoveFlag::CastleKing => {
                let (rf, rt) = if us == Color::White { (7u8, 5u8) } else { (63u8, 61u8) };
                self.remove_piece(PieceType::Rook, us, rt);
                self.add_piece(PieceType::Rook, us, rf);
            }
            MoveFlag::CastleQueen => {
                let (rf, rt) = if us == Color::White { (0u8, 3u8) } else { (56u8, 59u8) };
                self.remove_piece(PieceType::Rook, us, rt);
                self.add_piece(PieceType::Rook, us, rf);
            }
            _ => {
                if let Some((cpt, cc)) = undo.captured {
                    self.add_piece(cpt, cc, mv.to);
                }
            }
        }

        self.castling = undo.castling;
        self.ep_square = undo.ep_square;
        self.halfmove = undo.halfmove;
        if us == Color::Black {
            self.fullmove -= 1;
        }
        // One assignment instead of replaying every XOR backwards -- and it
        // also repairs whatever the add_piece/remove_piece calls above did to
        // the key while restoring the board.
        self.hash = undo.hash;
        // Mirrors the end of `make_move` -- see `refresh_perspectives` for why
        // leaving this out is a bug that only shows up as a quietly wrong
        // evaluation. `moving_pt` is what was on `from` before the move, so a
        // promotion reports a pawn and castling reports the king, which is
        // exactly right in both cases.
        if moving_pt == PieceType::King {
            self.refresh_perspectives();
        }
    }

    pub fn to_fen(&self) -> String {
        let mut s = String::new();
        for rank in (0..8i32).rev() {
            let mut empty = 0;
            for file in 0..8u8 {
                let sqi = sq(file, rank as u8);
                match self.piece_at(sqi) {
                    None => empty += 1,
                    Some((pt, c)) => {
                        if empty > 0 {
                            s.push_str(&empty.to_string());
                            empty = 0;
                        }
                        s.push(pt.to_char(c));
                    }
                }
            }
            if empty > 0 {
                s.push_str(&empty.to_string());
            }
            if rank > 0 {
                s.push('/');
            }
        }
        s.push(' ');
        s.push(if self.side == Color::White { 'w' } else { 'b' });
        s.push(' ');
        let mut cr = String::new();
        if self.castling & CASTLE_WK != 0 {
            cr.push('K');
        }
        if self.castling & CASTLE_WQ != 0 {
            cr.push('Q');
        }
        if self.castling & CASTLE_BK != 0 {
            cr.push('k');
        }
        if self.castling & CASTLE_BQ != 0 {
            cr.push('q');
        }
        s.push_str(if cr.is_empty() { "-" } else { &cr });
        s.push(' ');
        s.push_str(&sq_name(self.ep_square));
        s.push(' ');
        s.push_str(&self.halfmove.to_string());
        s.push(' ');
        s.push_str(&self.fullmove.to_string());
        s
    }
}
