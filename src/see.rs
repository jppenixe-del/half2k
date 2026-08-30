//! Static exchange evaluation.
//!
//! Plays out the whole capture sequence on one square, cheapest attacker first,
//! and reports what the side to move ends up with. It is what lets the search
//! tell a capture that wins material from one that merely looks violent, and it
//! is the difference between a quiescence that settles and one that chases
//! every recapture to the horizon.
//!
//! Nothing here needs search state, so it is a set of free functions.

use crate::attacks::{bishop_attacks, rook_attacks, Attacks};
use crate::board::Board;
use crate::bitboard::*;
use crate::moves::{Move, MoveFlag};
use crate::types::{file_of, rank_of, sq, Color, PieceType, Square};

/// Static Exchange Evaluation: simula a sequencia completa de
/// capturas/recapturas na casa `mv.to`, sempre com o atacante menos
/// valioso de cada lado (a jogada optima para ambos), e devolve o
/// ganho material líquido assumindo optimo jogo de ambos os lados
/// (cada lado escolhe parar ou continuar a troca, o que for melhor
/// para si -- minimax classico sobre a "swap list"). Nao verifica
/// se a recaptura deixaria o proprio rei em xeque (limitacao
/// standard/aceite de SEE simples, presente em praticamente todos
/// os motores). So' chamar em lances de captura (incl. en passant).
pub fn see(a: &Attacks, board: &Board, mv: &Move) -> i32 {
    let to = mv.to;
    let Some((attacker_pt0, attacker_color0)) = board.piece_at(mv.from) else {
        return 0;
    };
    let victim_val0 = if mv.flag == MoveFlag::EnPassant {
        PieceType::Pawn.value()
    } else {
        match board.piece_at(to) {
            Some((pt, _)) => pt.value(),
            // Quiet move: nothing is captured, so the exchange starts
            // at zero -- but the sequence below still runs, because the
            // piece we just moved can be taken on `to`. That makes this
            // a general "does this move lose material?" test rather
            // than a capture-only one (it used to bail out with 0 here,
            // which silently made any SEE-based test on a quiet move a
            // no-op). Every existing caller guards on is_capture(), so
            // their behaviour is unchanged.
            None => 0,
        }
    };

    let mut occ = board.occ_all;
    occ &= !bb(mv.from);
    if mv.flag == MoveFlag::EnPassant {
        let ep_captured = sq(file_of(to), rank_of(mv.from));
        occ &= !bb(ep_captured);
    }

    // Era um Vec com capacidade 1: uma alocacao no heap por CHAMADA, e
    // cada push a realocar (1->2->4->8). SEE e' chamado na ordenacao de
    // lances, na poda e na avaliacao de pecas penduradas -- milhoes de
    // vezes por segundo. A sequencia de trocas tem um tecto de 32, por
    // isso cabe na pilha e nunca precisou do heap.
    let mut gains = [0i32; 34];
    gains[0] = victim_val0;
    let mut n_gains = 1usize;
    let mut attacker_val = attacker_pt0.value();
    let mut side = attacker_color0.opp();

    // Os atacantes eram revarridos DO ZERO a cada troca -- ate' 32 vezes
    // por SEE, cada uma com duas consultas de peao, uma de cavalo, uma de
    // rei, duas magias e oito ORs para montar as mascaras. Mas ao tirar
    // uma peca do tabuleiro so' um deslizante pode passar a atacar a casa
    // (a bateria atras dele); cavalos, peoes e reis nunca aparecem de
    // novo. Entao calcula-se o conjunto uma vez e a seguir so' se
    // reexaminam os deslizantes, com as mascaras montadas ca' fora.
    let diag = board.pieces[Color::White.idx()][PieceType::Bishop.idx()]
        | board.pieces[Color::Black.idx()][PieceType::Bishop.idx()]
        | board.pieces[Color::White.idx()][PieceType::Queen.idx()]
        | board.pieces[Color::Black.idx()][PieceType::Queen.idx()];
    let orth = board.pieces[Color::White.idx()][PieceType::Rook.idx()]
        | board.pieces[Color::Black.idx()][PieceType::Rook.idx()]
        | board.pieces[Color::White.idx()][PieceType::Queen.idx()]
        | board.pieces[Color::Black.idx()][PieceType::Queen.idx()];
    let mut attackers = attackers_to(a, board, to, occ);
    loop {
        let side_attackers = attackers & board.occ_color[side.idx()];
        let Some((lva_sq, lva_pt)) = least_valuable_attacker(board, side_attackers, side) else {
            break;
        };
        gains[n_gains] = attacker_val - gains[n_gains - 1];
        n_gains += 1;
        attacker_val = lva_pt.value();
        occ &= !bb(lva_sq);
        // Tira o atacante usado e push_change o que ele tapava.
        attackers |= (bishop_attacks(to, occ) & diag) | (rook_attacks(to, occ) & orth);
        attackers &= occ;
        side = side.opp();
        if n_gains > 32 {
            break;
        }
    }

    for i in (1..n_gains).rev() {
        gains[i - 1] = (-gains[i]).min(gains[i - 1]);
    }
    gains[0]
}
/// `see(mv) >= limiar`, mas sem calcular o valor exacto.
///
/// O SEE completo constroi a sequencia de trocas toda e so' no fim, na
/// passagem inversa, e' que sabe o resultado. Quando a pergunta e' apenas
/// "isto passa a fasquia?" -- e e' o que a maioria dos sitios pergunta:
/// `>= 0` na quiescencia, `< see_allowance` na poda -- a resposta costuma
/// ficar decidida a' primeira ou segunda troca. Aqui leva-se um valor
/// corrente com o truque negamax (`valor = peca - valor`) e sai-se assim
/// que o sinal deixa de poder mudar.
///
/// Tem de concordar com `see(..) >= limiar` em TODAS as posicoes; se
/// discordar numa que seja, a contagem de nos do bench muda.
pub fn see_ge(a: &Attacks, board: &Board, mv: &Move, limiar: i32) -> bool {
    let to = mv.to;
    let Some((attacker_pt0, attacker_color0)) = board.piece_at(mv.from) else {
        return 0 >= limiar;
    };
    let victim_val0 = if mv.flag == MoveFlag::EnPassant {
        PieceType::Pawn.value()
    } else {
        match board.piece_at(to) {
            Some((pt, _)) => pt.value(),
            None => 0,
        }
    };

    let mut valor = victim_val0 - limiar;
    if valor < 0 {
        return false;
    }
    valor = attacker_pt0.value() - valor;
    if valor <= 0 {
        return true;
    }

    let mut occ = board.occ_all;
    occ &= !bb(mv.from);
    if mv.flag == MoveFlag::EnPassant {
        let ep_captured = sq(file_of(to), rank_of(mv.from));
        occ &= !bb(ep_captured);
    }

    let diag = board.pieces[Color::White.idx()][PieceType::Bishop.idx()]
        | board.pieces[Color::Black.idx()][PieceType::Bishop.idx()]
        | board.pieces[Color::White.idx()][PieceType::Queen.idx()]
        | board.pieces[Color::Black.idx()][PieceType::Queen.idx()];
    let orth = board.pieces[Color::White.idx()][PieceType::Rook.idx()]
        | board.pieces[Color::Black.idx()][PieceType::Rook.idx()]
        | board.pieces[Color::White.idx()][PieceType::Queen.idx()]
        | board.pieces[Color::Black.idx()][PieceType::Queen.idx()];
    let mut attackers = attackers_to(a, board, to, occ);

    let mut side = attacker_color0.opp();
    let mut res = true;
    loop {
        attackers &= occ;
        let side_attackers = attackers & board.occ_color[side.idx()];
        let Some((lva_sq, lva_pt)) = least_valuable_attacker(board, side_attackers, side)
        else {
            break;
        };
        res = !res;
        valor = lva_pt.value() - valor;
        // `res` diz de quem e' a vez de ficar a ganhar: o limite muda de
        // 0 para 1 conforme o lado, tal como na formulacao classica.
        if valor < res as i32 {
            break;
        }
        occ &= !bb(lva_sq);
        attackers |= (bishop_attacks(to, occ) & diag) | (rook_attacks(to, occ) & orth);
        side = side.opp();
    }
    res
}

pub fn attackers_to(a: &Attacks, board: &Board, s: crate::types::Square, occ: crate::bitboard::Bitboard) -> crate::bitboard::Bitboard {
    let a = a;
    let mut att = 0u64;
    att |= a.pawn[Color::Black.idx()][s as usize] & board.pieces[Color::White.idx()][PieceType::Pawn.idx()];
    att |= a.pawn[Color::White.idx()][s as usize] & board.pieces[Color::Black.idx()][PieceType::Pawn.idx()];
    att |= a.knight[s as usize]
        & (board.pieces[Color::White.idx()][PieceType::Knight.idx()] | board.pieces[Color::Black.idx()][PieceType::Knight.idx()]);
    att |= a.king[s as usize]
        & (board.pieces[Color::White.idx()][PieceType::King.idx()] | board.pieces[Color::Black.idx()][PieceType::King.idx()]);
    let diag = board.pieces[Color::White.idx()][PieceType::Bishop.idx()]
        | board.pieces[Color::Black.idx()][PieceType::Bishop.idx()]
        | board.pieces[Color::White.idx()][PieceType::Queen.idx()]
        | board.pieces[Color::Black.idx()][PieceType::Queen.idx()];
    att |= bishop_attacks(s, occ) & diag;
    let orth = board.pieces[Color::White.idx()][PieceType::Rook.idx()]
        | board.pieces[Color::Black.idx()][PieceType::Rook.idx()]
        | board.pieces[Color::White.idx()][PieceType::Queen.idx()]
        | board.pieces[Color::Black.idx()][PieceType::Queen.idx()];
    att |= rook_attacks(s, occ) & orth;
    att & occ
}
pub fn least_valuable_attacker(
    board: &Board,
    attackers: crate::bitboard::Bitboard,
    side: Color,
) -> Option<(crate::types::Square, PieceType)> {
    for pt in [
        PieceType::Pawn,
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
        PieceType::King,
    ] {
        let bbp = attackers & board.pieces[side.idx()][pt.idx()];
        if bbp != 0 {
            return Some((bbp.trailing_zeros() as crate::types::Square, pt));
        }
    }
    None
}
