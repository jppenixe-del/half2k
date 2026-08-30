use crate::types::Square;

pub type Bitboard = u64;

#[inline(always)]
pub fn bb(s: Square) -> Bitboard {
    1u64 << s
}

#[inline(always)]
pub fn pop_lsb(b: &mut Bitboard) -> Square {
    let s = b.trailing_zeros() as Square;
    *b &= *b - 1;
    s
}

#[inline(always)]
pub fn count(b: Bitboard) -> u32 {
    b.count_ones()
}

/// a1=index 0 is a dark square; (rank+file)%2==1 is the light-square
/// parity used throughout the codebase (see eval.rs bishop-color checks).
pub const LIGHT_SQUARES: Bitboard = 0x55AA55AA55AA55AA;
pub const DARK_SQUARES: Bitboard = !LIGHT_SQUARES;
pub const FILE_A: Bitboard = 0x0101010101010101;
pub const FILE_H: Bitboard = 0x8080808080808080;
pub const RANK_1: Bitboard = 0x00000000000000FF;
pub const RANK_2: Bitboard = 0x000000000000FF00;
pub const RANK_3: Bitboard = 0x0000000000FF0000;
pub const RANK_4: Bitboard = 0x00000000FF000000;
pub const RANK_5: Bitboard = 0x000000FF00000000;
pub const RANK_6: Bitboard = 0x0000FF0000000000;
pub const RANK_7: Bitboard = 0x00FF000000000000;
pub const RANK_8: Bitboard = 0xFF00000000000000;

#[inline(always)]
pub fn north(b: Bitboard) -> Bitboard {
    b << 8
}
#[inline(always)]
pub fn south(b: Bitboard) -> Bitboard {
    b >> 8
}
#[inline(always)]
pub fn east(b: Bitboard) -> Bitboard {
    (b & !FILE_H) << 1
}
#[inline(always)]
pub fn west(b: Bitboard) -> Bitboard {
    (b & !FILE_A) >> 1
}
