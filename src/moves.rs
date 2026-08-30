use crate::types::*;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum MoveFlag {
    Quiet,
    DoublePush,
    Capture,
    EnPassant,
    CastleKing,
    CastleQueen,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Move {
    pub from: Square,
    pub to: Square,
    pub promotion: Option<PieceType>,
    pub flag: MoveFlag,
}

impl Move {
    pub fn quiet(from: Square, to: Square) -> Self {
        Move { from, to, promotion: None, flag: MoveFlag::Quiet }
    }
    pub fn capture(from: Square, to: Square) -> Self {
        Move { from, to, promotion: None, flag: MoveFlag::Capture }
    }
    pub fn is_capture(&self) -> bool {
        matches!(self.flag, MoveFlag::Capture | MoveFlag::EnPassant)
    }
    pub fn to_uci(&self) -> String {
        let mut s = format!("{}{}", sq_name(self.from), sq_name(self.to));
        if let Some(p) = self.promotion {
            s.push(match p {
                PieceType::Queen => 'q',
                PieceType::Rook => 'r',
                PieceType::Bishop => 'b',
                PieceType::Knight => 'n',
                _ => '?',
            });
        }
        s
    }
}
