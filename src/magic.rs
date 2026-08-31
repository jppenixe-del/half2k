//! Real magic bitboards for sliding-piece attacks, replacing the naive
//! ray-cast scan that used to live in attacks.rs (profiled at ~17% of
//! total search time -- see the commit that adds this file). Standard,
//! well-known technique (Chess Programming Wiki "Magic Bitboards");
//! magic numbers are found here via random search at startup rather than
//! hardcoded, so there's nothing to transcribe/verify against an
//! external source -- the search is self-verifying (every candidate is
//! checked against a straightforward ray-cast implementation for every
//! occupancy subset before being accepted).
use crate::bitboard::*;
use crate::types::Square;

fn file_of_i(s: Square) -> i32 {
    (s % 8) as i32
}
fn rank_of_i(s: Square) -> i32 {
    (s / 8) as i32
}
fn sq_i(f: i32, r: i32) -> Square {
    (r * 8 + f) as Square
}

/// Ground-truth slow attack generator (same algorithm the old
/// attacks::ray_attacks used) -- only ever called during table
/// construction at startup, never from the search.
fn ray_attacks_slow(s: Square, occ: Bitboard, dirs: &[(i32, i32)]) -> Bitboard {
    let f0 = file_of_i(s);
    let r0 = rank_of_i(s);
    let mut out = 0u64;
    for &(df, dr) in dirs {
        let mut f = f0 + df;
        let mut r = r0 + dr;
        while (0..8).contains(&f) && (0..8).contains(&r) {
            let t = sq_i(f, r);
            out |= bb(t);
            if occ & bb(t) != 0 {
                break;
            }
            f += df;
            r += dr;
        }
    }
    out
}

const ROOK_DIRS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
const BISHOP_DIRS: [(i32, i32); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];

/// Relevant-occupancy mask: the ray in each direction, EXCLUDING the
/// final edge square (a blocker there can't hide anything further, so
/// it never changes the attack set and is dropped to shrink the table).
fn relevant_mask(s: Square, dirs: &[(i32, i32)]) -> Bitboard {
    let f0 = file_of_i(s);
    let r0 = rank_of_i(s);
    let mut out = 0u64;
    for &(df, dr) in dirs {
        let mut f = f0 + df;
        let mut r = r0 + dr;
        while (0..8).contains(&(f + df)) && (0..8).contains(&(r + dr)) {
            out |= bb(sq_i(f, r));
            f += df;
            r += dr;
        }
    }
    out
}

/// n-th subset of `mask` via the standard binary-counter-over-set-bits
/// trick (enumerates all 2^popcount(mask) occupancy subsets).
fn subset(index: usize, mask: Bitboard) -> Bitboard {
    let mut result = 0u64;
    let mut m = mask;
    let mut i = index;
    while m != 0 {
        let bit = m & m.wrapping_neg(); // lowest set bit
        m &= m - 1;
        if i & 1 != 0 {
            result |= bit;
        }
        i >>= 1;
    }
    result
}

struct Xorshift64(u64);
impl Xorshift64 {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// Sparse random candidate -- magics with few set bits are far more
    /// likely to hash well, standard trick.
    fn sparse(&mut self) -> u64 {
        self.next() & self.next() & self.next()
    }
}

struct SquareMagic {
    mask: Bitboard,
    magic: u64,
    shift: u32,
    table: Vec<Bitboard>,
}

impl SquareMagic {
    #[inline(always)]
    fn index(&self, occ: Bitboard) -> usize {
        (((occ & self.mask).wrapping_mul(self.magic)) >> self.shift) as usize
    }
    #[inline(always)]
    fn attacks(&self, occ: Bitboard) -> Bitboard {
        self.table[self.index(occ)]
    }
}

fn find_magic_for_square(s: Square, dirs: &[(i32, i32)], rng: &mut Xorshift64) -> SquareMagic {
    let mask = relevant_mask(s, dirs);
    let bits = mask.count_ones();
    let size = 1usize << bits;
    let shift = 64 - bits;

    // Precompute every (occupancy subset, real attack) pair once -- reused
    // across every magic candidate tried below.
    let mut occs = Vec::with_capacity(size);
    let mut atts = Vec::with_capacity(size);
    for i in 0..size {
        let occ = subset(i, mask);
        occs.push(occ);
        atts.push(ray_attacks_slow(s, occ, dirs));
    }

    // Um so' buffer para todas as tentativas, com marca de epoca em vez de
    // limpeza. O que aqui estava -- `vec![None; size]` DENTRO do ciclo --
    // alocava e zerava 64 KB por cada magic candidato, e sao precisos muitos
    // candidatos ate' um servir: no perfil do bench, esta funcao e o `memset`
    // que ela provoca somavam 9,5% do tempo todo. Uma entrada conta como vazia
    // quando a sua marca nao e' a desta tentativa, portanto nao ha' nada a
    // limpar entre tentativas.
    let mut tabela: Vec<Bitboard> = vec![0; size];
    let mut marca: Vec<u32> = vec![0; size];
    let mut epoca: u32 = 0;

    loop {
        let magic = rng.sparse();
        epoca += 1;
        let mut ok = true;
        for i in 0..size {
            let idx = ((occs[i].wrapping_mul(magic)) >> shift) as usize;
            if marca[idx] != epoca {
                marca[idx] = epoca;
                tabela[idx] = atts[i];
            } else if tabela[idx] != atts[i] {
                ok = false;
                break;
            }
            // colisao construtiva (mesmo ataque) e' aceitavel, como antes
        }
        if !ok {
            continue;
        }
        // As entradas que esta tentativa nunca tocou ficam a zero, que e'
        // exactamente o que o `unwrap_or(0)` produzia.
        let final_table: Vec<Bitboard> =
            (0..size).map(|i| if marca[i] == epoca { tabela[i] } else { 0 }).collect();
        return SquareMagic { mask, magic, shift, table: final_table };
    }
}

pub struct Magics {
    rook: Vec<SquareMagic>,
    bishop: Vec<SquareMagic>,
}


/// Constroi um `SquareMagic` a partir de um magic JA' CONHECIDO (os nossos,
/// embutidos abaixo). Preenche a tabela directamente -- sem o ciclo de
/// tentativa-e-erro do `find_magic_for_square`, que era ~0,4s de arranque por
/// lancamento. Colisoes construtivas (mesmo ataque no mesmo indice) sao OK; uma
/// colisao DESTRUTIVA seria um magic errado (const corrompida) e faz panic.
fn magic_from_known(s: Square, dirs: &[(i32, i32)], magic: u64) -> SquareMagic {
    let mask = relevant_mask(s, dirs);
    let bits = mask.count_ones();
    let size = 1usize << bits;
    let shift = 64 - bits;
    let mut table: Vec<Bitboard> = vec![0; size];
    let mut posto: Vec<bool> = vec![false; size];
    for i in 0..size {
        let occ = subset(i, mask);
        let att = ray_attacks_slow(s, occ, dirs);
        let idx = ((occ.wrapping_mul(magic)) >> shift) as usize;
        if posto[idx] && table[idx] != att {
            panic!("magic embutido invalido para o quadrado {}", s);
        }
        table[idx] = att;
        posto[idx] = true;
    }
    SquareMagic { mask, magic, shift, table }
}

pub(crate) const ROOK_MAGICS: [u64; 64] = [0x9080001184204004,0x00c01008a0004000,0x0500081100c12000,0x4700090084203000,0x9200060068210410,0x0080020014000980,0x00801a0001005080,0x0900018021450002,0x3000800220400480,0x2000400042201000,0x0008808020009000,0x8000801000800800,0x09010008020c1100,0x0602001012000884,0x0441002402000100,0x0140800244802100,0x9220608000c00090,0x0090054000c82000,0x0001010010406001,0x4801848018003000,0x0404050010080100,0x0001010006040008,0x0001840028053002,0x020022000080c104,0x0408208080044004,0x00500040c0002002,0x8d02002200348040,0x0000080080100080,0x8091104500080100,0x0802008080140042,0x0001000500040600,0x0090018a0000c104,0x0314814000800420,0x0401810042002e00,0x0241422001001100,0x0080801000800802,0x0080040180800800,0x4002010802001004,0x8002010804001002,0x200a01004e000c84,0x100281c011208008,0x02c1200050004000,0x0400302001010040,0x0001100009010020,0x0001240008008080,0x002a002010040400,0x2420210002008080,0x088d0401c0920021,0x0001004820820200,0x0100a90080420600,0x0a20002110008880,0x20801000200d0100,0x0200240008018080,0x1002003008040a00,0x8424234a10082c00,0x06a2310488440200,0x7009001242800025,0x2d0d022040008051,0x420b900820010041,0x0100140890010021,0x0900480010050105,0x0401000c00024801,0x0082121008010184,0x800201040024d086,];
pub(crate) const BISHOP_MAGICS: [u64; 64] = [0x0008010804840080,0x01100200b10a0802,0x0010041081a02404,0x3009204200840000,0x4044242001048401,0x0ca2025004000082,0xa900480210100880,0x8228620a06200621,0x0c40109230040180,0x000408480ca08200,0x0000284244002198,0x0210180a00240300,0x81340c0421800804,0x1080911048042400,0x190088c104104010,0x00014840c4042014,0x004000a058020080,0x20900c2511020410,0x1026008408021100,0x8002000403620200,0x0004000081a08040,0x5081800110100100,0x0003000401180200,0x0002004100820301,0x041010800c20524b,0x0461042021180200,0x0128010018005300,0x1002006012008200,0x1440840000802004,0x0001802012021000,0x0041440203028806,0x0000408420420800,0x1401041190602000,0x40008a1010208422,0x6304020800050440,0x00420084000a0210,0x0051010400020020,0x00540800a00a0280,0xc044010404004404,0x0100840082084a00,0x3008020804006080,0x00010910900c2200,0x01104c004c000800,0x0806002011040808,0x0004202040900700,0x312204a102008101,0x0002080104102300,0x0404108411402110,0x0841009010082004,0xc414208208200080,0x4080a20042084000,0x201c000284044001,0x082c00102a021800,0x1000400408188208,0x0210101021484020,0x08021802208a0084,0x6211048884054000,0x4441120101015000,0x0014008100a82404,0x8020082220208800,0x0000e00410020610,0x8008014108190904,0x04100c2038020098,0x40102000a1020022,];

impl Magics {
    pub fn new() -> Self {
        // Magics EMBUTIDOS (deterministicos, os nossos -- ver `dump_magic_numbers`).
        // Constroi as tabelas directamente, sem o ciclo de busca (~0,4s antes).
        // O `find_magic_for_square` fica so' para o `dumpmagics`.
        let mut rook = Vec::with_capacity(64);
        let mut bishop = Vec::with_capacity(64);
        for s in 0..64u8 {
            rook.push(magic_from_known(s, &ROOK_DIRS, ROOK_MAGICS[s as usize]));
            bishop.push(magic_from_known(s, &BISHOP_DIRS, BISHOP_MAGICS[s as usize]));
        }
        Magics { rook, bishop }
    }

    #[inline(always)]
    pub fn rook_attacks(&self, s: Square, occ: Bitboard) -> Bitboard {
        self.rook[s as usize].attacks(occ)
    }
    #[inline(always)]
    pub fn bishop_attacks(&self, s: Square, occ: Bitboard) -> Bitboard {
        self.bishop[s as usize].attacks(occ)
    }
}

/// Despeja os 128 magics encontrados (deterministicos, semente fixa) como
/// arrays const de Rust, para os embutir e saltar a busca por tentativa no
/// arranque -- ~0,4s por lancamento do motor. Sao os NOSSOS magics (a nossa
/// busca encontra-os), so' cacheados.
pub fn dump_magic_numbers() {
    let m = Magics::new();
    print!("pub(crate) const ROOK_MAGICS: [u64; 64] = [");
    for sm in &m.rook { print!("{:#018x},", sm.magic); }
    println!("];");
    print!("pub(crate) const BISHOP_MAGICS: [u64; 64] = [");
    for sm in &m.bishop { print!("{:#018x},", sm.magic); }
    println!("];");
}
