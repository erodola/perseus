//! Host reference math for the selftest: exact u128 modular arithmetic, NTT, ChaCha12 expander, FFT encode/decode.
use pc_kernels::N;

pub const LOGN: u32 = 16;

pub fn mulmod(a: u64, b: u64, p: u64) -> u64 {
    ((a as u128 * b as u128) % p as u128) as u64
}
pub fn addmod(a: u64, b: u64, p: u64) -> u64 {
    let s = a as u128 + b as u128;
    (if s >= p as u128 { s - p as u128 } else { s }) as u64
}
pub fn submod(a: u64, b: u64, p: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        a + (p - b)
    }
}
pub fn powmod(mut a: u64, mut e: u64, p: u64) -> u64 {
    let mut r = 1u64 % p;
    a %= p;
    while e > 0 {
        if e & 1 == 1 {
            r = mulmod(r, a, p);
        }
        a = mulmod(a, a, p);
        e >>= 1;
    }
    r
}
pub fn invmod(a: u64, p: u64) -> u64 {
    powmod(a, p - 2, p)
}
/// Shoup companion floor(w 2^bits / p)
pub fn shoup(w: u64, p: u64, word_bits: u32) -> u64 {
    (((w as u128) << word_bits) / p as u128) as u64
}
pub fn brev(x: usize, bits: u32) -> usize {
    x.reverse_bits() >> (usize::BITS - bits)
}

/// Deterministic Miller-Rabin for n < 2^64.
pub fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for p in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if n % p == 0 {
            return n == p;
        }
    }
    let (mut d, mut s) = (n - 1, 0);
    while d % 2 == 0 {
        d /= 2;
        s += 1;
    }
    'w: for a in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        let mut x = powmod(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 1..s {
            x = mulmod(x, x, n);
            if x == n - 1 {
                continue 'w;
            }
        }
        return false;
    }
    true
}

/// `count` primes p = 1 mod 2N just below 2^bits (NTT-friendly), largest first.
pub fn ntt_primes(bits: u32, count: usize) -> Vec<u64> {
    let step = 2 * N as u64;
    let mut c = (1u64 << bits) + 1;
    let mut v = Vec::new();
    while v.len() < count {
        c -= step;
        if is_prime(c) {
            v.push(c);
        }
    }
    v
}
/// `count` primes just below 2^bits (no NTT structure).
pub fn plain_primes(bits: u32, count: usize) -> Vec<u64> {
    let mut c = (1u64 << bits) - 1;
    let mut v = Vec::new();
    while v.len() < count {
        if is_prime(c) {
            v.push(c);
        }
        c -= 2;
    }
    v
}

pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    pub fn below(&mut self, p: u64) -> u64 {
        ((self.next() as u128 * p as u128) >> 64) as u64
    }
    pub fn f64(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn vec(&mut self, n: usize, p: u64) -> Vec<u64> {
        (0..n).map(|_| self.below(p)).collect()
    }
}

/// A primitive 2N-th root of unity mod p.
pub fn psi_of(p: u64, rng: &mut Rng) -> u64 {
    loop {
        let g = 2 + rng.below(p - 3);
        let x = powmod(g, (p - 1) / (2 * N as u64), p);
        if powmod(x, N as u64, p) == p - 1 {
            return x;
        }
    }
}

/// Twiddle tables in the kernels' layout: w\[i\] = psi^brev(i), iw\[i\] = psi^-brev(i).
pub fn twiddles(p: u64, psi: u64) -> (Vec<u64>, Vec<u64>) {
    let ipsi = invmod(psi, p);
    let mut pw = vec![1u64; N];
    let mut ipw = vec![1u64; N];
    for i in 1..N {
        pw[i] = mulmod(pw[i - 1], psi, p);
        ipw[i] = mulmod(ipw[i - 1], ipsi, p);
    }
    let w = (0..N).map(|i| pw[brev(i, LOGN)]).collect();
    let iw = (0..N).map(|i| ipw[brev(i, LOGN)]).collect();
    (w, iw)
}

/// OpenFHE ForwardTransformToBitReverse with root psi.
pub fn ntt_ref(a: &mut [u64], w: &[u64], p: u64) {
    let n = a.len();
    let mut t = n;
    let mut m = 1;
    while m < n {
        t /= 2;
        for i in 0..m {
            let s = w[m + i];
            let j1 = 2 * i * t;
            for j in j1..j1 + t {
                let u = a[j];
                let v = mulmod(a[j + t], s, p);
                a[j] = addmod(u, v, p);
                a[j + t] = submod(u, v, p);
            }
        }
        m *= 2;
    }
}

pub fn automorph_slot(k: u32, slot: u32) -> u32 {
    let j = slot.reverse_bits() >> (32 - LOGN);
    let jt = (j << 1) + 1;
    let r = (jt.wrapping_mul(k) & ((1u32 << (LOGN + 1)) - 1)) >> 1;
    r.reverse_bits() >> (32 - LOGN)
}

/// a(X^k) in Z_p\[X\]/(X^N + 1), coefficient form
pub fn poly_automorph(a: &[u64], k: u64, p: u64) -> Vec<u64> {
    let n = a.len() as u64;
    let mut out = vec![0u64; a.len()];
    for (i, &x) in a.iter().enumerate() {
        let e = (i as u64 * k) % (2 * n);
        if e < n {
            out[e as usize] = x;
        } else {
            out[(e - n) as usize] = (p - x) % p;
        }
    }
    out
}

// ------------------------------------------------------------------------------------------------ ChaCha12 expander
fn qr(x: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    x[a] = x[a].wrapping_add(x[b]);
    x[d] = (x[d] ^ x[a]).rotate_left(16);
    x[c] = x[c].wrapping_add(x[d]);
    x[b] = (x[b] ^ x[c]).rotate_left(12);
    x[a] = x[a].wrapping_add(x[b]);
    x[d] = (x[d] ^ x[a]).rotate_left(8);
    x[c] = x[c].wrapping_add(x[d]);
    x[b] = (x[b] ^ x[c]).rotate_left(7);
}
pub fn chacha12(key: &[u32; 8], ctr: u32, w13: u32, w14: u32, w15: u32) -> [u32; 16] {
    let s = [
        0x61707865u32, 0x3320646e, 0x79622d32, 0x6b206574, key[0], key[1], key[2], key[3], key[4], key[5], key[6], key[7],
        ctr, w13, w14, w15,
    ];
    let mut x = s;
    for _ in 0..6 {
        qr(&mut x, 0, 4, 8, 12);
        qr(&mut x, 1, 5, 9, 13);
        qr(&mut x, 2, 6, 10, 14);
        qr(&mut x, 3, 7, 11, 15);
        qr(&mut x, 0, 5, 10, 15);
        qr(&mut x, 1, 6, 11, 12);
        qr(&mut x, 2, 7, 8, 13);
        qr(&mut x, 3, 4, 9, 14);
    }
    let mut o = [0u32; 16];
    for i in 0..16 {
        o[i] = x[i].wrapping_add(s[i]);
    }
    o
}
pub const TAG_KSKA: u32 = 0x4b534b41;
pub const TAG_ENCA: u32 = 0x454e4341;

/// The expander rule of chacha.cuh for words of `wb` bytes. u64 KSKA = the frozen KSKB lane: the pc-core twin
/// (`pc_core::keys::kskb_expand`, = FIDESlib expand_coeff64) is the reference, not a re-derivation.
pub fn expand_slot(seed: &[u32; 8], slot: usize, w13: u32, p: u64, tag: u32, wb: usize) -> u64 {
    if wb == 8 && tag == TAG_KSKA {
        return pc_core::keys::kskb_expand(&pc_core::keys::KskSeed(*seed), w13, p, slot as u32, N);
    }
    if wb == 4 && tag == TAG_KSKA {
        return pc_core::keys::kska_expand(&pc_core::keys::KskSeed(*seed), w13, p as u32, slot as u32, N) as u64;
    }
    let (w13, tag) = if wb == 8 { (w13 ^ (((p >> 32) as u32) << 4), tag + 1) } else { (w13, tag) };
    let per = 64 / wb;
    let nblk = (N / per) as u32;
    let b0 = (slot / per) as u32;
    let j = slot % per;
    let maxw: u64 = if wb == 8 { u64::MAX } else { (1u64 << (8 * wb)) - 1 };
    let mp = (maxw / p) * p;
    let pw = p as u32;
    let mut v = 0u64;
    for t in 0..16u32 {
        let blk = chacha12(seed, b0.wrapping_add(t * nblk), w13, pw, tag);
        v = match wb {
            4 => blk[j] as u64,
            8 => blk[2 * j] as u64 | ((blk[2 * j + 1] as u64) << 32),
            _ => ((blk[j >> 1] >> (16 * (j & 1))) & 0xFFFF) as u64,
        };
        if v < mp {
            break;
        }
    }
    v % p
}

// ------------------------------------------------------------------------------------------------ FFT encode/decode
#[derive(Clone, Copy, Debug, Default)]
pub struct C64 {
    pub re: f64,
    pub im: f64,
}
impl C64 {
    pub fn mul(self, o: C64) -> C64 {
        C64 { re: self.re * o.re - self.im * o.im, im: self.re * o.im + self.im * o.re }
    }
    pub fn cis(t: f64) -> C64 {
        C64 { re: t.cos(), im: t.sin() }
    }
}
/// In-place radix-2 FFT, sum_j a_j exp(sign 2 pi i jk / n), no normalization.
pub fn fft(a: &mut [C64], sign: f64) {
    let n = a.len();
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = brev(i, bits);
        if i < j {
            a.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let tw: Vec<C64> = (0..len / 2).map(|k| C64::cis(sign * 2.0 * std::f64::consts::PI * k as f64 / len as f64)).collect();
        for s in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let u = a[s + k];
                let v = a[s + k + len / 2].mul(tw[k]);
                a[s + k] = C64 { re: u.re + v.re, im: u.im + v.im };
                a[s + k + len / 2] = C64 { re: u.re - v.re, im: u.im - v.im };
            }
        }
        len *= 2;
    }
}
fn five_pow(j: usize) -> usize {
    let m = 2 * N;
    let (mut t, mut b, mut e) = (1usize, 5usize, j);
    while e > 0 {
        if e & 1 == 1 {
            t = t * b % m;
        }
        b = b * b % m;
        e >>= 1;
    }
    t
}
/// Real coefficients c_k (before rounding), as the device encode.
pub fn encode_coeffs(v: &[C64], scale: f64) -> Vec<f64> {
    let mut w = vec![C64::default(); N];
    for (j, &x) in v.iter().enumerate() {
        let t = five_pow(j);
        w[(t - 1) / 2] = x;
        w[(2 * N - t - 1) / 2] = C64 { re: x.re, im: -x.im };
    }
    fft(&mut w, -1.0);
    (0..N)
        .map(|k| {
            let z = C64::cis(-std::f64::consts::PI * k as f64 / N as f64);
            w[k].mul(z).re / N as f64 * scale
        })
        .collect()
}
/// OpenFHE split + per-limb residues (exact), as the device scatter.
pub fn encode_limbs(c: &[f64], primes: &[u64], deg_factor: &[u64]) -> Vec<Vec<u64>> {
    let cmax = c.iter().fold(0.0f64, |a, &x| a.max(x.abs()));
    let split = if cmax > 0.0 {
        let e = ((cmax.to_bits() >> 52) & 0x7FF) as i32 - 1022; // frexp exponent for a normal double
        let f_half = cmax.to_bits() & ((1u64 << 52) - 1) == 0;
        let logc = if f_half { e - 1 } else { e };
        if logc > 61 {
            logc - 61
        } else {
            0
        }
    } else {
        0
    };
    primes
        .iter()
        .zip(deg_factor)
        .map(|(&p, &df)| {
            let mut f = df;
            for _ in 0..split {
                f = addmod(f, f, p);
            }
            c.iter()
                .map(|&x| {
                    let m = (x / (1u64 << split) as f64).round() as i64;
                    let mut r = m.unsigned_abs() % p;
                    if m < 0 && r != 0 {
                        r = p - r;
                    }
                    mulmod(r, f, p)
                })
                .collect()
        })
        .collect()
}
/// Slots of the integer coefficients m_k (already centered) at `scale`.
pub fn decode(m: &[f64], scale: f64) -> Vec<C64> {
    let mut u: Vec<C64> = (0..N)
        .map(|k| {
            let z = C64::cis(std::f64::consts::PI * k as f64 / N as f64);
            C64 { re: m[k] / scale * z.re, im: m[k] / scale * z.im }
        })
        .collect();
    fft(&mut u, 1.0);
    (0..N / 2).map(|j| u[(five_pow(j) - 1) / 2]).collect()
}
