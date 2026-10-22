//! Contract test of the "general modular product" seam: every ModMul strategy, and Shoup under every strategy, agrees
//! with the exact i128 product on 10^6 random (a, b, p) per prime width (plus the edge operands 0, 1, p-1).
use pc_core::modmul::{mul_shoup, shoup, Barrett32, Barrett64, ModMul, Montgomery};
use pc_core::params::nbtheory::is_prime;
use pc_core::word::Word;

const PER_WIDTH: usize = 1_000_000;
const PRIMES_PER_WIDTH: usize = 64;

struct SplitMix(u64);
impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        ((self.next() as u128 * n as u128) >> 64) as u64
    }
}

fn random_prime(rng: &mut SplitMix, bits: u32) -> u64 {
    loop {
        let x = (1u64 << (bits - 1)) | rng.below(1u64 << (bits - 1)) | 1;
        if is_prime(x) {
            return x;
        }
    }
}

fn contract<W: Word, M: ModMul<W>>(bits_list: &[u32], seed: u64) -> usize {
    let mut rng = SplitMix(seed);
    let mut checked = 0;
    for &bits in bits_list {
        assert!(bits <= M::MAX_PRIME_BITS, "{} does not cover {bits} bits", M::NAME);
        let primes: Vec<u64> = (0..PRIMES_PER_WIDTH).map(|_| random_prime(&mut rng, bits)).collect();
        for (k, &p) in primes.iter().enumerate() {
            let c = M::prime_const(W::from_u64(p)).unwrap();
            assert_eq!(M::modulus(&c).to_u64(), p);
            let n = PER_WIDTH / PRIMES_PER_WIDTH;
            for t in 0..n {
                let (a, b) = match t {
                    0 => (0, p - 1),
                    1 => (1, p - 1),
                    2 => (p - 1, p - 1),
                    _ => (rng.below(p), rng.below(p)),
                };
                let want = (a as i128 * b as i128 % p as i128) as u64;
                let ra = M::to_repr(W::from_u64(a), &c);
                let rb = M::to_repr(W::from_u64(b), &c);
                assert_eq!(M::from_repr(ra, &c).to_u64(), a, "{} repr roundtrip p={p}", M::NAME);
                let got = M::from_repr(M::mul(ra, rb, &c), &c).to_u64();
                assert_eq!(got, want, "{} {bits}-bit p={p} a={a} b={b}", M::NAME);
                // Shoup by the standard-form constant b, applied in this strategy's representation.
                let s = shoup(W::from_u64(b), W::from_u64(p));
                let got = M::from_repr(mul_shoup(ra, &s, W::from_u64(p)), &c).to_u64();
                assert_eq!(got, want, "shoup under {} {bits}-bit p={p} a={a} b={b}", M::NAME);
                checked += 1;
            }
            // Shoup on unreduced operands (any a < 2^BITS) is part of its contract.
            if k == 0 {
                let s = shoup(W::from_u64(p - 1), W::from_u64(p));
                let top = if W::BITS == 64 { u64::MAX } else { (1u64 << W::BITS) - 1 };
                for a in [top, top - 1, p, 2 * p.min(top / 2)] {
                    let want = (a as u128 * (p - 1) as u128 % p as u128) as u64;
                    assert_eq!(mul_shoup(W::from_u64(a), &s, W::from_u64(p)).to_u64(), want);
                }
            }
        }
    }
    checked
}

#[test]
fn barrett64_u32() {
    let n = contract::<u32, Barrett64>(&[20, 24, 27, 28, 29, 30, 31], 1);
    println!("barrett64: {n} products");
}

#[test]
fn barrett32_u32() {
    let n = contract::<u32, Barrett32>(&[20, 24, 27, 28, 29, 30], 2);
    println!("barrett32: {n} products");
}

#[test]
fn montgomery_u32() {
    let n = contract::<u32, Montgomery>(&[20, 24, 27, 28, 29, 30, 31], 3);
    println!("montgomery u32: {n} products");
}

#[test]
fn montgomery_u64() {
    let n = contract::<u64, Montgomery>(&[36, 45, 50, 53, 59, 60], 4);
    println!("montgomery u64: {n} products");
}

#[test]
fn montgomery_u16() {
    let n = contract::<u16, Montgomery>(&[10, 13, 15], 5);
    println!("montgomery u16: {n} products");
}

#[test]
fn layouts() {
    use pc_core::modmul::{Barrett32Const, Barrett64Const, MontConst, ShoupConst};
    assert_eq!(std::mem::size_of::<Barrett64Const>(), 16);
    assert_eq!(std::mem::size_of::<Barrett32Const>(), 16);
    assert_eq!(std::mem::size_of::<MontConst<u32>>(), 12);
    assert_eq!(std::mem::size_of::<ShoupConst<u32>>(), 8);
    assert_eq!(std::mem::size_of::<ShoupConst<u64>>(), 16);
}
