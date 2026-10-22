//! PCMM key material: the plaintext x ciphertext matmul at the StC-first split (ePrint 2024/1284 Alg. 2; Sylph,
//! arXiv 2601.18511), port of perseus `impl/pcmm.py` (`find_prime`, `_tables`, `_ks_key`, `make_keys`) over the
//! ChaCha20 DRBG (`Prng::from_os` for real keys; only the distribution matters).
//!
//! Keys: `fwd` (ring 2^16 `s` -> `s'(X^16)`, `s'` ternary of degree 2^12), `pack` (r small-ring switches, one per MLWE
//! component s_l -> s') and `back` (s'(X^16) -> s); each has one digit per q prime and one 31-bit special prime P.
//! Layout (u32): `[digit j < 2][prime i < 3 (q0, q1, P)][a, b][n]`, pack keys back to back; `PcmmKeys` holds the
//! coefficient form, `eval_*` the NTT form the device reads. Details in docs/ARCHITECTURE.md "pc-core pcmm_keys".
use crate::error::{Error, Result};
use crate::keys::{sample_ternary, DiscreteGaussian, Prng};
use crate::modmul::{invmod, mulmod, powmod};
use crate::params::nbtheory::is_prime;

/// Ring dimension of the CKKS ciphertext.
pub const N: usize = 1 << 16;
/// Ring-switch factor 2^16 -> 2^12.
pub const S_SWITCH: usize = 16;
/// Small ring dimension.
pub const N1: usize = N / S_SWITCH;
/// Key-switch digits (the q primes q0, q1).
pub const L: usize = 2;
/// Primes of a key (q0, q1, P).
pub const B: usize = 3;
/// Error width of the PCMM keys (pcmm.py SIGMA).
pub const SIGMA: f64 = 3.2;
/// Bits of the special prime.
pub const P_BITS: u32 = 31;

/// MLWE shape for `t` token lanes: pieces of degree np = 2t, rank r = N1 / np, nch = 16 r channels per ciphertext.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Shape {
    pub t: usize,
}

impl Shape {
    pub fn new(t: usize) -> Result<Shape> {
        if t == 0 || !t.is_power_of_two() || 2 * t > N1 {
            return Err(Error::Keys(format!("pcmm shape: t = {t} token lanes (need a power of two, 2t <= {N1})")));
        }
        Ok(Shape { t })
    }
    pub fn np(&self) -> usize {
        2 * self.t
    }
    pub fn r(&self) -> usize {
        N1 / self.np()
    }
    pub fn nch(&self) -> usize {
        S_SWITCH * self.r()
    }
    /// Channel held by GEMM row `sub * r + comp` (folded into W by the weight encoder; pcmm.py `ch_of_row`).
    pub fn ch_of_row(&self) -> Vec<usize> {
        let (r, nch) = (self.r(), self.nch());
        let bits = nch.trailing_zeros();
        (0..nch).map(|idx| brev(idx / r + S_SWITCH * (idx % r), bits)).collect()
    }
}

/// Bit reversal of the low `bits` bits.
pub fn brev(x: usize, bits: u32) -> usize {
    if bits == 0 {
        return 0;
    }
    x.reverse_bits() >> (usize::BITS - bits)
}

/// Largest prime below 2^bits that is 1 mod m and not in `avoid` (pcmm.py `find_prime`).
pub fn find_prime(bits: u32, m: u64, avoid: &[u64]) -> Result<u64> {
    let mut p = ((1u64 << bits) - 1) / m * m + 1;
    while p > 1u64 << (bits - 1) {
        if !avoid.contains(&p) && is_prime(p) {
            return Ok(p);
        }
        p -= m;
    }
    Err(Error::Keys(format!("pcmm: no {bits}-bit prime = 1 mod {m} outside {avoid:?}")))
}

/// Smallest generator of Z_p^* (pcmm.py `_primitive_root`).
pub fn primitive_root(p: u64) -> u64 {
    let mut x = p - 1;
    let mut fs = Vec::new();
    let mut f = 2u64;
    while f * f <= x {
        if x % f == 0 {
            fs.push(f);
            while x % f == 0 {
                x /= f;
            }
        }
        f += 1;
    }
    if x > 1 {
        fs.push(x);
    }
    let mut g = 2u64;
    while !fs.iter().all(|&f| powmod(g, (p - 1) / f, p) != 1) {
        g += 1;
    }
    g
}

/// Negacyclic NTT tables of one prime at one ring degree, in the device's order (FIDESlib Pcmm.cu nttSmem):
/// `psi[k] = psi^brev(k)`, `ipsi[k] = psi^-brev(k)`, `ninv = n^-1`.
#[derive(Clone, Debug)]
pub struct NttTab {
    pub p: u64,
    pub n: usize,
    pub psi: Vec<u32>,
    pub ipsi: Vec<u32>,
    pub ninv: u32,
}

impl NttTab {
    pub fn new(p: u64, n: usize) -> Result<NttTab> {
        if p >= 1 << 32 || (p - 1) % (2 * n as u64) != 0 || !n.is_power_of_two() {
            return Err(Error::Keys(format!("pcmm ntt: prime {p} does not support the negacyclic NTT of degree {n} in 32 bits")));
        }
        let psi = powmod(primitive_root(p), (p - 1) / (2 * n as u64), p);
        let ipsi = invmod(psi, p);
        let bits = n.trailing_zeros();
        let (mut pw, mut ipw) = (vec![0u32; n], vec![0u32; n]);
        let (mut a, mut b) = (1u64, 1u64);
        for k in 0..n {
            let r = brev(k, bits);
            pw[r] = a as u32;
            ipw[r] = b as u32;
            a = mulmod(a, psi, p);
            b = mulmod(b, ipsi, p);
        }
        Ok(NttTab { p, n, psi: pw, ipsi: ipw, ninv: invmod(n as u64 % p, p) as u32 })
    }

    /// Forward transform in place (Cooley-Tukey, natural -> bit-reversed).
    pub fn ntt(&self, a: &mut [u32]) {
        let (n, p) = (self.n, self.p);
        let (mut m, mut t) = (1, n / 2);
        while m < n {
            for i in 0..m {
                let w = self.psi[m + i] as u64;
                let j0 = 2 * i * t;
                for j in j0..j0 + t {
                    let u = a[j] as u64;
                    let v = a[j + t] as u64 * w % p;
                    a[j] = ((u + v) % p) as u32;
                    a[j + t] = ((u + p - v) % p) as u32;
                }
            }
            m *= 2;
            t /= 2;
        }
    }

    /// Inverse transform in place (Gentleman-Sande, bit-reversed -> natural, times n^-1).
    pub fn intt(&self, a: &mut [u32]) {
        let (n, p) = (self.n, self.p);
        let (mut m, mut t) = (n / 2, 1);
        while m >= 1 {
            for i in 0..m {
                let w = self.ipsi[m + i] as u64;
                let j0 = 2 * i * t;
                for j in j0..j0 + t {
                    let (u, v) = (a[j] as u64, a[j + t] as u64);
                    a[j] = ((u + v) % p) as u32;
                    a[j + t] = ((u + p - v) % p * w % p) as u32;
                }
            }
            m /= 2;
            t *= 2;
        }
        let ni = self.ninv as u64;
        for x in a.iter_mut() {
            *x = (*x as u64 * ni % p) as u32;
        }
    }

    /// Negacyclic product a b mod (X^n + 1, p) (coefficient form in and out).
    pub fn polymul(&self, a: &[u32], b: &[u32]) -> Vec<u32> {
        let (mut fa, mut fb) = (a.to_vec(), b.to_vec());
        self.ntt(&mut fa);
        self.ntt(&mut fb);
        for (x, y) in fa.iter_mut().zip(&fb) {
            *x = (*x as u64 * *y as u64 % self.p) as u32;
        }
        self.intt(&mut fa);
        fa
    }
}

/// Signed integers reduced mod p.
pub fn signed_mod(v: &[i64], p: u64) -> Vec<u32> {
    v.iter().map(|&x| x.rem_euclid(p as i64) as u32).collect()
}

/// `[digit j][prime i][a, b][n]` coefficient form: key_j = (a, -a s_to + e + P (Q / q_j) s_from) over q0, q1, P
/// (pcmm.py `_ks_key`); one error polynomial per digit (shared by the primes), `a` uniform per (digit, prime).
pub fn ks_key(s_from: &[i64], s_to: &[i64], tabs: &[NttTab; B], prng: &mut Prng, dg: &DiscreteGaussian) -> Vec<u32> {
    let n = s_from.len();
    let (q0, q1, pp) = (tabs[0].p, tabs[1].p, tabs[2].p);
    let qs = [q0, q1];
    let mut out = vec![0u32; L * B * 2 * n];
    for j in 0..L {
        // g = P * (Q / q_j) = P * q_{1-j}
        let g = pp as u128 * qs[1 - j] as u128;
        let e = dg.sample_vec(prng, n);
        for (i, tab) in tabs.iter().enumerate() {
            let p = tab.p;
            let a: Vec<u32> = (0..n).map(|_| prng.uniform_below(p) as u32).collect();
            let as_to = tab.polymul(&a, &signed_mod(s_to, p));
            let gm = (g % p as u128) as u64;
            let base = ((j * B + i) * 2) * n;
            for k in 0..n {
                let sf = s_from[k].rem_euclid(p as i64) as u64;
                let b = (p - as_to[k] as u64 + e[k].rem_euclid(p as i64) as u64 + sf * gm % p) % p;
                out[base + k] = a[k];
                out[base + n + k] = b as u32;
            }
        }
    }
    out
}

/// The PCMM key set of one token-lane shape, coefficient form (pcmm.py `make_keys`).
#[derive(Clone, Debug)]
pub struct PcmmKeys {
    pub shape: Shape,
    /// q0, q1, P.
    pub primes: [u64; B],
    /// The chain's moduli at the split (limbs 0..3), as the device checks them.
    pub split: [u64; 4],
    /// s -> s'(X^16) at N: `[2][3][2][N]`.
    pub fwd: Vec<u32>,
    /// s'(X^16) -> s at N.
    pub back: Vec<u32>,
    /// r keys s_l -> s' at N1: `[r][2][3][2][N1]`.
    pub pack: Vec<u32>,
}

impl PcmmKeys {
    /// Keys for `t` lanes from the ring-2^16 secret's coefficients (`sk`, ternary) and the chain's four lowest
    /// moduli, sampled from `prng` (pass `Prng::from_os()` for real keys).
    pub fn generate(sk: &[i8], split: &[u64], t: usize, prng: &mut Prng) -> Result<PcmmKeys> {
        let shape = Shape::new(t)?;
        if sk.len() != N {
            return Err(Error::Keys(format!("pcmm keys: secret of {} coefficients, need {N}", sk.len())));
        }
        if sk.iter().any(|&c| !(-1..=1).contains(&c)) {
            return Err(Error::Keys("pcmm keys: the secret is not ternary".into()));
        }
        if split.len() < 4 {
            return Err(Error::Keys(format!("pcmm keys: {} split moduli, need the chain's 4 lowest", split.len())));
        }
        let split4 = [split[0], split[1], split[2], split[3]];
        if split4.iter().any(|&q| q >= 1 << 31) {
            return Err(Error::Keys(format!("pcmm keys: split moduli {split4:?} must be below 2^31 (balanced int8 digits)")));
        }
        let p = find_prime(P_BITS, 2 * N as u64, &split4)?;
        let primes = [split4[0], split4[1], p];
        let t16 = [NttTab::new(primes[0], N)?, NttTab::new(primes[1], N)?, NttTab::new(primes[2], N)?];
        let t12 = [NttTab::new(primes[0], N1)?, NttTab::new(primes[1], N1)?, NttTab::new(primes[2], N1)?];
        let dg = DiscreteGaussian::new(SIGMA)?;
        let s: Vec<i64> = sk.iter().map(|&c| c as i64).collect();
        let s1: Vec<i64> = sample_ternary(prng, N1).into_iter().map(|c| c as i64).collect();
        let mut emb = vec![0i64; N];
        for (k, &c) in s1.iter().enumerate() {
            emb[k * S_SWITCH] = c;
        }
        let (np, r) = (shape.np(), shape.r());
        let mut pack = Vec::with_capacity(r * L * B * 2 * N1);
        for l in 0..r {
            // component l of s': s_l[w] = s'[w r + l], placed at the piece stride r
            let mut e = vec![0i64; N1];
            for w in 0..np {
                e[w * r] = s1[w * r + l];
            }
            pack.extend(ks_key(&e, &s1, &t12, prng, &dg));
        }
        let fwd = ks_key(&s, &emb, &t16, prng, &dg);
        let back = ks_key(&emb, &s, &t16, prng, &dg);
        Ok(PcmmKeys { shape, primes, split: split4, fwd, back, pack })
    }

    /// NTT tables of the three primes at `n` (N or N1).
    pub fn tables(&self, n: usize) -> Result<[NttTab; B]> {
        Ok([NttTab::new(self.primes[0], n)?, NttTab::new(self.primes[1], n)?, NttTab::new(self.primes[2], n)?])
    }

    /// A key array in the device's NTT form (each `[a|b]` polynomial transformed with its prime's table).
    pub fn eval_form(key: &[u32], tabs: &[NttTab; B]) -> Vec<u32> {
        let n = tabs[0].n;
        let mut out = key.to_vec();
        for (idx, poly) in out.chunks_mut(n).enumerate() {
            // [.. ][prime][ab][n]: the prime of polynomial idx is (idx / 2) % 3
            tabs[(idx / 2) % B].ntt(poly);
        }
        out
    }

    pub fn bytes(&self) -> usize {
        4 * (self.fwd.len() + self.back.len() + self.pack.len())
    }
}

// ── weights ─────────────────────────────────────────────────────────────────────────────────────────────────────

/// Balanced base-256 digits of an integer matrix: `n_planes` int8 planes with w = sum_u planes\[u\] 256^u
/// (pcmm.py `digit_planes`), plane-major.
pub fn digit_planes(w: &[i64], n_planes: usize) -> Result<Vec<i8>> {
    let mut v = w.to_vec();
    let mut out = vec![0i8; n_planes * w.len()];
    for u in 0..n_planes {
        for (k, x) in v.iter_mut().enumerate() {
            let r = (*x + 128).rem_euclid(256) - 128;
            out[u * w.len() + k] = r as i8;
            *x = (*x - r) / 256;
        }
    }
    if v.iter().any(|&x| x != 0) {
        return Err(Error::Keys(format!("digit_planes: the matrix needs more than {n_planes} planes")));
    }
    Ok(out)
}

/// Inverse of `digit_planes` (pcmm.py `from_planes`).
pub fn from_planes(planes: &[i8], n_planes: usize) -> Vec<i64> {
    let len = planes.len() / n_planes.max(1);
    let mut acc = vec![0i64; len];
    for u in (0..n_planes).rev() {
        for (k, a) in acc.iter_mut().enumerate() {
            *a = *a * 256 + planes[u * len + k] as i64;
        }
    }
    acc
}

/// Largest |integer weight| a `wbits`-bit encoding uses (pcmm.py `PcmmLinear.encode`: `top`).
pub fn weight_top(wbits: u32) -> i64 {
    let nb = wbits.div_ceil(8);
    let reach = 127 * (256i128.pow(nb) - 1) / 255;
    ((1i128 << (wbits - 1)) - 1).min(reach) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prng() -> Prng {
        Prng::from_seed([7; 8], 3)
    }

    #[test]
    fn shape_and_perm() {
        let s = Shape::new(64).unwrap();
        assert_eq!((s.np(), s.r(), s.nch()), (128, 32, 512));
        let mut p = s.ch_of_row();
        p.sort();
        assert_eq!(p, (0..512).collect::<Vec<_>>());
        assert!(Shape::new(3).is_err() && Shape::new(4096).is_err());
    }

    #[test]
    fn ntt_roundtrip_and_negacyclic() {
        let p = find_prime(31, 2 * N as u64, &[]).unwrap();
        assert_eq!(p % (2 * N as u64), 1);
        let tab = NttTab::new(p, 16).unwrap();
        let a: Vec<u32> = (0..16).map(|i| (i * 7 + 3) as u32).collect();
        let b: Vec<u32> = (0..16).map(|i| (i * i + 1) as u32).collect();
        let mut x = a.clone();
        tab.ntt(&mut x);
        tab.intt(&mut x);
        assert_eq!(x, a);
        // schoolbook negacyclic
        let mut want = vec![0i128; 16];
        for i in 0..16 {
            for j in 0..16 {
                let v = a[i] as i128 * b[j] as i128;
                if i + j < 16 {
                    want[i + j] += v;
                } else {
                    want[i + j - 16] -= v;
                }
            }
        }
        let want: Vec<u32> = want.iter().map(|v| v.rem_euclid(p as i128) as u32).collect();
        assert_eq!(tab.polymul(&a, &b), want);
    }

    #[test]
    fn digits() {
        let w = vec![0i64, 1, -1, 127, -128, 128, 32767, -32768, 8_355_711, -8_388_608];
        let d = digit_planes(&w, 3).unwrap();
        assert_eq!(from_planes(&d, 3), w);
        assert!(digit_planes(&[1 << 24], 3).is_err());
        assert_eq!(weight_top(16), 32639);
        assert_eq!(weight_top(24), 8_355_711);
    }

    /// The key relation b + a s_to = e + P (Q/q_j) s_from (mod p), |e| small, on a reduced-size check of `ks_key`.
    #[test]
    fn key_relation() {
        let mut g = prng();
        let p = find_prime(31, 2 * N as u64, &[]).unwrap();
        let q0 = find_prime(30, 2 * N as u64, &[p]).unwrap();
        let q1 = find_prime(30, 2 * N as u64, &[p, q0]).unwrap();
        let n = 64;
        let tabs = [NttTab::new(q0, n).unwrap(), NttTab::new(q1, n).unwrap(), NttTab::new(p, n).unwrap()];
        let s_from: Vec<i64> = sample_ternary(&mut g, n).into_iter().map(|c| c as i64).collect();
        let s_to: Vec<i64> = sample_ternary(&mut g, n).into_iter().map(|c| c as i64).collect();
        let dg = DiscreteGaussian::new(SIGMA).unwrap();
        let k = ks_key(&s_from, &s_to, &tabs, &mut g, &dg);
        for j in 0..L {
            let gq = p as u128 * [q1, q0][j] as u128;
            for (i, tab) in tabs.iter().enumerate() {
                let pi = tab.p;
                let base = ((j * B + i) * 2) * n;
                let a = &k[base..base + n];
                let b = &k[base + n..base + 2 * n];
                let as_to = tab.polymul(a, &signed_mod(&s_to, pi));
                for c in 0..n {
                    let lhs = (b[c] as u64 + as_to[c] as u64) % pi;
                    let m = (s_from[c].rem_euclid(pi as i64) as u64 * (gq % pi as u128) as u64) % pi;
                    let e = (lhs + pi - m) % pi;
                    let ec = if e > pi / 2 { e as i64 - pi as i64 } else { e as i64 };
                    assert!(ec.abs() <= 60, "error {ec}");
                }
            }
        }
    }
}
