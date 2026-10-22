//! CCMM key material: the ciphertext x ciphertext batch matmul (Cheon-Kang-Lee, ePrint 2025/1957), port of perseus
//! `impl/ccmm.py::make_keys` over the ChaCha20 DRBG (`Prng::from_os` for real keys; only the distribution matters).
//!
//! The CCMM runs on the four bottom moduli q0..q3 (the `stc_high` output) plus one 31-bit special prime P; every key
//! is hybrid (one digit per q prime). Keys: `fwd` (s -> s'(X^8), s' ternary of degree 2^13), `back` (the switch back),
//! `autos[t]` (t < 128, sigma_h(s') -> s' with h = 2 KY t + 1; t = 0 still switches, as FIDESlib Ccmm.cu does) and
//! `rlk` (s'^2 -> s'). Layout (u32): `[digit j < 4][prime i < 5 (q0..q3, P)][a, b][n]` in coefficient form, `autos`
//! back to back; the device reads the NTT form (`eval_form`). Details in docs/ARCHITECTURE.md "pc-core ccmm_keys".
use crate::error::{Error, Result};
use crate::keys::{sample_ternary, DiscreteGaussian, Prng};
use crate::pcmm_keys::{find_prime, signed_mod, NttTab};
use rayon::prelude::*;

/// Ring dimension of the CKKS ciphertext.
pub const NB: usize = 1 << 16;
/// Small ring dimension (one CKL column ciphertext).
pub const NS: usize = 1 << 13;
/// Ring-switch stride 2^16 -> 2^13.
pub const SW: usize = NB / NS;
/// Matrix size d.
pub const D: usize = 128;
/// Batch ring degree k (R_k = Z\[Y\] / (Y^64 + 1), Y = X^d): 32 complex batch slots.
pub const KY: usize = NS / D;
/// Ciphertexts per matrix (128 columns, 8 per ring-2^16 ciphertext).
pub const NBIG: usize = D / SW;
/// Key-switch digits (the q primes).
pub const NQ: usize = 4;
/// Primes of a key (q0..q3, P).
pub const NP: usize = 5;
/// Error width (ccmm.py SIGMA, shared with pcmm.py).
pub const SIGMA: f64 = 3.2;
/// Bits of the special prime.
pub const P_BITS: u32 = 31;

/// Words of one key at ring degree `n`.
pub const fn key_words(n: usize) -> usize {
    NQ * NP * 2 * n
}

/// sigma_h on a signed coefficient vector of degree n: coefficient u -> position u h mod 2n with the negacyclic sign
/// (ccmm.py `_auto_poly`).
pub fn auto_poly(s: &[i64], h: usize) -> Vec<i64> {
    let n = s.len();
    let mut out = vec![0i64; n];
    for (u, &v) in s.iter().enumerate() {
        let t = u * h % (2 * n);
        if t < n {
            out[t] = v;
        } else {
            out[t - n] = -v;
        }
    }
    out
}

/// s^2 mod (X^n + 1) exactly (ccmm.py `_negacyclic_square`; |coefficients| <= n for a ternary s).
pub fn negacyclic_square(s: &[i64]) -> Vec<i64> {
    let n = s.len();
    let nz: Vec<(usize, i64)> = s.iter().enumerate().filter(|(_, &v)| v != 0).map(|(i, &v)| (i, v)).collect();
    let mut out = vec![0i64; n];
    for &(i, a) in &nz {
        for &(j, b) in &nz {
            let k = i + j;
            if k < n {
                out[k] += a * b;
            } else {
                out[k - n] -= a * b;
            }
        }
    }
    out
}

/// The automorphism exponent of column t of the CMT.
pub fn auto_h(t: usize) -> usize {
    2 * KY * t + 1
}

/// a^-1 mod m for gcd(a, m) = 1, any modulus (extended Euclid; `modmul::invmod` is Fermat, primes only).
pub fn inv_mod(a: u64, m: u64) -> Option<u64> {
    let (mut t, mut nt, mut r, mut nr) = (0i128, 1i128, m as i128, (a % m) as i128);
    while nr != 0 {
        let q = r / nr;
        (t, nt) = (nt, t - q * nt);
        (r, nr) = (nr, r - q * nr);
    }
    if r != 1 {
        return None;
    }
    Some(t.rem_euclid(m as i128) as u64)
}

/// Source column of CMT output column t (FIDESlib ScrConst::tstar): (h_t^-1 mod 2 NS - 1) / (2 KY).
pub fn tstar(t: usize) -> usize {
    // h_t is odd, so it is a unit mod the power of two 2 NS
    let hinv = inv_mod(auto_h(t) as u64, 2 * NS as u64).unwrap_or(1);
    ((hinv - 1) / (2 * KY as u64)) as usize
}

/// `[digit j][prime i][a, b][n]` coefficient form: key_j = (a, -a s_to + e + P (Q / q_j) s_from) over q0..q3, P
/// (pcmm.py `_ks_key` with four digits); one error polynomial per digit shared by the primes, `a` uniform per (digit,
/// prime). The randomness is drawn sequentially from `prng`; the products run in parallel.
pub fn ks_key(s_from: &[i64], s_to: &[i64], tabs: &[NttTab; NP], prng: &mut Prng, dg: &DiscreteGaussian) -> Vec<u32> {
    let n = s_from.len();
    let q: Vec<u64> = tabs[..NQ].iter().map(|t| t.p).collect();
    let mut draws: Vec<(usize, usize, Vec<u32>, Vec<i64>)> = Vec::with_capacity(NQ * NP);
    for j in 0..NQ {
        let e = dg.sample_vec(prng, n);
        for (i, tab) in tabs.iter().enumerate() {
            let a: Vec<u32> = (0..n).map(|_| prng.uniform_below(tab.p) as u32).collect();
            draws.push((j, i, a, e.clone()));
        }
    }
    let mut out = vec![0u32; key_words(n)];
    out.par_chunks_mut(2 * n).zip(draws.par_iter()).for_each(|(dst, (j, i, a, e))| {
        let tab = &tabs[*i];
        let p = tab.p;
        // g = P (Q / q_j) mod p
        let mut g = tabs[NQ].p % p;
        for (jj, &qq) in q.iter().enumerate() {
            if jj != *j {
                g = (g as u128 * (qq % p) as u128 % p as u128) as u64;
            }
        }
        let as_to = tab.polymul(a, &signed_mod(s_to, p));
        for k in 0..n {
            let sf = s_from[k].rem_euclid(p as i64) as u64;
            let b = (p - as_to[k] as u64 + e[k].rem_euclid(p as i64) as u64 + (sf as u128 * g as u128 % p as u128) as u64) % p;
            dst[k] = a[k];
            dst[n + k] = b as u32;
        }
    });
    out
}

/// The CCMM key set (coefficient form), ccmm.py `make_keys`.
#[derive(Clone, Debug)]
pub struct CcmmKeys {
    /// q0..q3 (the chain's four bottom moduli), P.
    pub primes: [u64; NP],
    /// s -> s'(X^8) at NB.
    pub fwd: Vec<u32>,
    /// s'(X^8) -> s at NB.
    pub back: Vec<u32>,
    /// 128 keys sigma_h(s') -> s' at NS, h = 2 KY t + 1.
    pub autos: Vec<u32>,
    /// s'^2 -> s' at NS.
    pub rlk: Vec<u32>,
}

impl CcmmKeys {
    /// The special prime for the chain's four bottom moduli (31 bits, 1 mod 2 NB, not one of them).
    pub fn special_prime(moduli: &[u64]) -> Result<u64> {
        find_prime(P_BITS, 2 * NB as u64, &moduli[..NQ.min(moduli.len())])
    }

    /// Keys from the ring-2^16 secret's coefficients (`sk`, ternary) and the chain's moduli (the four lowest are
    /// used), sampled from `prng`. Returns the keys and the small-ring secret s' (tests decrypt with it).
    pub fn generate(sk: &[i8], moduli: &[u64], prng: &mut Prng) -> Result<(CcmmKeys, Vec<i8>)> {
        if sk.len() != NB {
            return Err(Error::Keys(format!("ccmm keys: secret of {} coefficients, need {NB}", sk.len())));
        }
        if sk.iter().any(|&c| !(-1..=1).contains(&c)) {
            return Err(Error::Keys("ccmm keys: the secret is not ternary".into()));
        }
        if moduli.len() < NQ {
            return Err(Error::Keys(format!("ccmm keys: {} moduli, need the chain's {NQ} lowest", moduli.len())));
        }
        let q4 = [moduli[0], moduli[1], moduli[2], moduli[3]];
        if q4.iter().any(|&q| q >= 1 << 31) {
            return Err(Error::Keys(format!("ccmm keys: moduli {q4:?} must be below 2^31 (balanced int8 digits)")));
        }
        let p = Self::special_prime(&q4)?;
        let primes = [q4[0], q4[1], q4[2], q4[3], p];
        let t16 = tables(&primes, NB)?;
        let t13 = tables(&primes, NS)?;
        let dg = DiscreteGaussian::new(SIGMA)?;
        let s: Vec<i64> = sk.iter().map(|&c| c as i64).collect();
        let s1i = sample_ternary(prng, NS);
        let s1: Vec<i64> = s1i.iter().map(|&c| c as i64).collect();
        let mut emb = vec![0i64; NB];
        for (k, &c) in s1.iter().enumerate() {
            emb[k * SW] = c;
        }
        let fwd = ks_key(&s, &emb, &t16, prng, &dg);
        let back = ks_key(&emb, &s, &t16, prng, &dg);
        let mut autos = Vec::with_capacity(D * key_words(NS));
        for t in 0..D {
            autos.extend(ks_key(&auto_poly(&s1, auto_h(t)), &s1, &t13, prng, &dg));
        }
        let rlk = ks_key(&negacyclic_square(&s1), &s1, &t13, prng, &dg);
        Ok((CcmmKeys { primes, fwd, back, autos, rlk }, s1i))
    }

    /// Keys given as arrays (perseus `bts.ccmm_keys` dict: `primes`, `fwd`, `back`, `autos`, `rlk`; its twiddle
    /// tables are recomputed from the primes, the device's own NTT order); sizes and residues checked.
    pub fn from_arrays(primes: &[u64], fwd: Vec<u32>, back: Vec<u32>, autos: Vec<u32>, rlk: Vec<u32>) -> Result<CcmmKeys> {
        if primes.len() != NP {
            return Err(Error::Keys(format!("ccmm keys: {} primes, need {NP} (q0..q3, P)", primes.len())));
        }
        let primes = [primes[0], primes[1], primes[2], primes[3], primes[4]];
        for (what, v, want, n) in [("fwd", &fwd, key_words(NB), NB), ("back", &back, key_words(NB), NB), ("autos", &autos, D * key_words(NS), NS), ("rlk", &rlk, key_words(NS), NS)] {
            if v.len() != want {
                return Err(Error::Keys(format!("ccmm keys: {what} has {} words, need {want}", v.len())));
            }
            if let Some((idx, x)) = v.iter().enumerate().find(|(idx, &x)| x as u64 >= primes[(idx / (2 * n)) % NP]) {
                return Err(Error::Keys(format!("ccmm keys: {what}[{idx}] = {x} is not reduced")));
            }
        }
        tables(&primes, NB)?;
        tables(&primes, NS)?;
        tables(&primes, KY)?;
        Ok(CcmmKeys { primes, fwd, back, autos, rlk })
    }

    /// NTT tables of the five primes at `n` (NB, NS or KY).
    pub fn tables(&self, n: usize) -> Result<[NttTab; NP]> {
        tables(&self.primes, n)
    }

    /// A key array in the device's NTT form (each `[a|b]` polynomial transformed with its prime's table).
    pub fn eval_form(key: &[u32], tabs: &[NttTab; NP]) -> Vec<u32> {
        let n = tabs[0].n;
        let mut out = key.to_vec();
        out.par_chunks_mut(n).enumerate().for_each(|(idx, poly)| tabs[(idx / 2) % NP].ntt(poly));
        out
    }

    pub fn bytes(&self) -> usize {
        4 * (self.fwd.len() + self.back.len() + self.autos.len() + self.rlk.len())
    }
}

/// NTT tables of `primes` at degree `n`.
pub fn tables(primes: &[u64; NP], n: usize) -> Result<[NttTab; NP]> {
    Ok([NttTab::new(primes[0], n)?, NttTab::new(primes[1], n)?, NttTab::new(primes[2], n)?, NttTab::new(primes[3], n)?, NttTab::new(primes[4], n)?])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_and_square() {
        let s = vec![1i64, -1, 0, 1];
        // h = 3: u -> 3u mod 8: 0 -> 0, 1 -> 3, 2 -> 6 (= -X^2), 3 -> 9 = 1 (mod 8)
        assert_eq!(auto_poly(&s, 3), vec![1, 1, 0, -1]);
        // (1 - X + X^3)^2 mod X^4 + 1 = 1 - 2X + X^2 + 2X^3 - 2X^4 + X^6 -> (1 + 2) - 2X + (1 - 1) X^2 + 2X^3
        assert_eq!(negacyclic_square(&s), vec![3, -2, 0, 2]);
        assert_eq!(auto_h(0), 1);
        assert_eq!(auto_h(127), 2 * 64 * 127 + 1);
        for t in 0..D {
            let h = auto_h(t) as u64;
            let hi = inv_mod(h, 2 * NS as u64).unwrap();
            assert_eq!(h * hi % (2 * NS as u64), 1);
            assert_eq!(hi % (2 * KY as u64), 1, "h^-1 = 1 mod 2 KY");
            assert!(tstar(t) < D);
        }
        let mut seen: Vec<usize> = (0..D).map(tstar).collect();
        seen.sort();
        assert_eq!(seen, (0..D).collect::<Vec<_>>());
        assert_eq!(inv_mod(3, 7), Some(5));
        assert_eq!(inv_mod(2, 4), None);
    }

    /// The key relation b + a s_to = e + P (Q / q_j) s_from (mod every prime), |e| small, at a reduced degree.
    #[test]
    fn key_relation() {
        let mut g = Prng::from_seed([5; 8], 1);
        let n = 64;
        let p = find_prime(31, 2 * NB as u64, &[]).unwrap();
        let mut q = Vec::new();
        for _ in 0..NQ {
            let mut avoid = q.clone();
            avoid.push(p);
            q.push(find_prime(28, 2 * NB as u64, &avoid).unwrap());
        }
        let primes = [q[0], q[1], q[2], q[3], p];
        let tabs = tables(&primes, n).unwrap();
        let s_from: Vec<i64> = sample_ternary(&mut g, n).into_iter().map(|c| c as i64).collect();
        let s_to: Vec<i64> = sample_ternary(&mut g, n).into_iter().map(|c| c as i64).collect();
        let dg = DiscreteGaussian::new(SIGMA).unwrap();
        let k = ks_key(&s_from, &s_to, &tabs, &mut g, &dg);
        for j in 0..NQ {
            for (i, tab) in tabs.iter().enumerate() {
                let pi = tab.p;
                let mut gq = p as u128 % pi as u128;
                for (jj, &qq) in q.iter().enumerate() {
                    if jj != j {
                        gq = gq * qq as u128 % pi as u128;
                    }
                }
                let base = ((j * NP + i) * 2) * n;
                let a = &k[base..base + n];
                let b = &k[base + n..base + 2 * n];
                let as_to = tab.polymul(a, &signed_mod(&s_to, pi));
                for c in 0..n {
                    let lhs = (b[c] as u64 + as_to[c] as u64) % pi;
                    let m = (s_from[c].rem_euclid(pi as i64) as u128 * gq % pi as u128) as u64;
                    let e = (lhs + pi - m) % pi;
                    let ec = if e > pi / 2 { e as i64 - pi as i64 } else { e as i64 };
                    assert!(ec.abs() <= 60, "error {ec}");
                }
            }
        }
        // from_arrays accepts what generate-shaped arrays look like and refuses unreduced words
        assert!(CcmmKeys::from_arrays(&primes, vec![0; 3], vec![], vec![], vec![]).is_err());
    }
}
