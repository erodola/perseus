//! Automorphisms of Z\[X\]/(X^N + 1): Galois elements, eval-form permutation tables, coefficient-form maps.
//!
//! THE convention (one, used by every crate):
//! - A rotation of the slot vector LEFT by r (out\[j\] = in[j + r], OpenFHE `EvalRotate(ct, r)`) is the automorphism
//!   sigma_g: X -> X^g with g = `rotation_galois(r)` = 5^r mod 2N (r taken mod N/2; negative r = 5^-r). Conjugation
//!   is g = 2N - 1. g is also the key-map index of OpenFHE's rotation key (`FindAutomorphismIndex2nComplex`).
//! - In evaluation form (OpenFHE/FIDESlib bit-reversed NTT order, index i holds a(psi^(2 brev(i) + 1))), sigma_g is
//!   the GATHER out\[i\] = in[t_g\[i\]] with t_g = `eval_gather_table(g)` (OpenFHE `PrecomputeAutoMap`).
//! - FIDESlib's kernels apply the same map as a SCATTER out[s(i)] = in\[i\] with s = `fides_scatter_index(k, .)`,
//!   k = g^-1 = 5^(2N - r) mod 2N (`fides_kernel_exponent`): scatter by g^-1 == gather by g.
//! - OpenFHE's key for index g switches from s to sigma_{g^-1}(s) (EvalAutomorphismKeyGen permutes the secret by
//!   g^-1); the evaluator key-switches first and applies sigma_g afterwards ("dot then sigma").
use crate::error::{Error, Result};

fn check_n(n: usize) -> Result<u32> {
    if n < 8 || !n.is_power_of_two() || n > (1 << 30) {
        return Err(Error::Params(format!("galois: ring dimension n={n} must be a power of two in [8, 2^30]")));
    }
    Ok(n.trailing_zeros())
}

fn powmod(mut b: u64, mut e: u64, m: u64) -> u64 {
    let mut r = 1u64 % m;
    b %= m;
    while e > 0 {
        if e & 1 == 1 {
            r = (r as u128 * b as u128 % m as u128) as u64;
        }
        b = (b as u128 * b as u128 % m as u128) as u64;
        e >>= 1;
    }
    r
}

/// Galois element of a left slot rotation by `r` (any sign): 5^(r mod N/2) mod 2N.
pub fn rotation_galois(r: i64, n: usize) -> Result<u64> {
    check_n(n)?;
    let h = (n / 2) as i64;
    let rr = r.rem_euclid(h) as u64;
    Ok(powmod(5, rr, 2 * n as u64))
}

/// Galois element of the conjugation automorphism: 2N - 1.
pub fn conj_galois(n: usize) -> Result<u64> {
    check_n(n)?;
    Ok(2 * n as u64 - 1)
}

/// Inverse of an odd g in (Z / 2N)^*.
pub fn galois_inverse(g: u64, n: usize) -> Result<u64> {
    check_n(n)?;
    let m = 2 * n as u64;
    if g % 2 == 0 {
        return Err(Error::Params(format!("galois element g={g} is even (not a unit mod 2N={m})")));
    }
    // the unit group mod 2^k has exponent 2^(k-2): g^-1 = g^(2^(k-2) - 1)
    Ok(powmod(g % m, (m >> 2) - 1, m))
}

/// The rotation step r (in [0, N/2)) with 5^r = g mod 2N, or None if g is not a power of 5 (e.g. conjugation).
pub fn galois_to_rotation(g: u64, n: usize) -> Result<Option<u64>> {
    check_n(n)?;
    let m = 2 * n as u64;
    let mut p = 1u64;
    for r in 0..(n / 2) as u64 {
        if p == g % m {
            return Ok(Some(r));
        }
        p = p * 5 % m;
    }
    Ok(None)
}

/// FIDESlib's automorph kernel exponent for a left rotation by r: 5^(2N - r) mod 2N = g^-1.
pub fn fides_kernel_exponent(r: i64, n: usize) -> Result<u64> {
    galois_inverse(rotation_galois(r, n)?, n)
}

#[inline]
fn brev(x: u32, bits: u32) -> u32 {
    x.reverse_bits() >> (32 - bits)
}

/// OpenFHE `PrecomputeAutoMap(n, g)`: eval-form gather table, sigma_g(a)\[i\] = a[t\[i\]].
pub fn eval_gather_table(g: u64, n: usize) -> Result<Vec<u32>> {
    let logn = check_n(n)?;
    if g % 2 == 0 {
        return Err(Error::Params(format!("galois element g={g} is even")));
    }
    let m = 2 * n as u64;
    let mut t = vec![0u32; n];
    for j in 0..n as u32 {
        let jt = 2 * j as u64 + 1;
        let idx = ((jt * (g % m)) % m) >> 1;
        t[brev(j, logn) as usize] = brev(idx as u32, logn);
    }
    Ok(t)
}

/// FIDESlib `automorph_slot(logN, k, slot)`: the scatter destination of eval index `slot` under kernel exponent k.
#[inline]
pub fn fides_scatter_index(k: u64, slot: u32, logn: u32) -> u32 {
    let j = brev(slot, logn) as u64;
    let jt = 2 * j + 1;
    let r = ((jt * k) & ((1u64 << (logn + 1)) - 1)) >> 1;
    brev(r as u32, logn)
}

/// sigma_g on an eval-form row (gather).
pub fn apply_eval<T: Copy>(row: &[T], g: u64) -> Result<Vec<T>> {
    let t = eval_gather_table(g, row.len())?;
    Ok(t.iter().map(|&i| row[i as usize]).collect())
}

/// sigma_g on a coefficient-form polynomial over Z (signed coefficients): X^i -> X^(i g mod 2N), with the sign flip
/// of X^N = -1.
pub fn apply_coeff_signed(c: &[i64], g: u64) -> Result<Vec<i64>> {
    let n = c.len();
    check_n(n)?;
    let m = 2 * n as u64;
    let mut out = vec![0i64; n];
    for (i, &v) in c.iter().enumerate() {
        let e = (i as u64 * (g % m)) % m;
        if e < n as u64 {
            out[e as usize] = v;
        } else {
            out[(e - n as u64) as usize] = -v;
        }
    }
    Ok(out)
}

/// sigma_g on a coefficient-form polynomial mod q (values in [0, q)).
pub fn apply_coeff_mod(c: &[u32], g: u64, q: u32) -> Result<Vec<u32>> {
    let n = c.len();
    check_n(n)?;
    let m = 2 * n as u64;
    let mut out = vec![0u32; n];
    for (i, &v) in c.iter().enumerate() {
        let e = (i as u64 * (g % m)) % m;
        if e < n as u64 {
            out[e as usize] = v;
        } else {
            out[(e - n as u64) as usize] = if v == 0 { 0 } else { q - v };
        }
    }
    Ok(out)
}
