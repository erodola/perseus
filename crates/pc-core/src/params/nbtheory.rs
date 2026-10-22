//! Number theory for parameter generation, ported from OpenFHE `core/include/math/nbtheory-impl.h`
//! (`FirstPrime`, `LastPrime`, `NextPrime`, `PreviousPrime`, `RootOfUnity`, `MillerRabinPrimalityTest`) and the
//! big-integer bit lengths the HE-standard check and the special-prime count need. Values are u64; the word-size
//! bound OpenFHE's NativeInteger imposes is checked by the callers against `ChainSpec::word`.
use crate::error::{Error, Result};
use crate::modmul::{mulmod, powmod};

/// Deterministic Miller-Rabin for every u64 (bases 2..37 are a proven witness set below 3.3e24). OpenFHE draws 100
/// random witnesses; on the moduli sizes used here both are exact, so the prime sequences coincide.
pub fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for sp in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if n % sp == 0 {
            return n == sp;
        }
    }
    let mut d = n - 1;
    let mut s = 0;
    while d % 2 == 0 {
        d /= 2;
        s += 1;
    }
    'witness: for a in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        let mut x = powmod(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 1..s {
            x = mulmod(x, x, n);
            if x == n - 1 {
                continue 'witness;
            }
        }
        return false;
    }
    true
}

fn overflow(func: &str, what: &str) -> Error {
    Error::Params(format!("{func}: {what}"))
}

/// Smallest prime `q >= 2^n_bits` with `q == 1 mod m` (OpenFHE FirstPrime).
pub fn first_prime(n_bits: u32, m: u64) -> Result<u64> {
    if n_bits >= 63 {
        return Err(overflow("FirstPrime", &format!("requested bit length {n_bits} exceeds 62")));
    }
    let q = 1u64 << n_bits;
    let r = q % m;
    let mut q_new = q + 1 - r;
    if r > 0 {
        q_new += m;
    }
    while !is_prime(q_new) {
        q_new = q_new.checked_add(m).ok_or_else(|| overflow("FirstPrime", "overflow growing candidate"))?;
    }
    Ok(q_new)
}

/// Largest prime `q < 2^n_bits` (in the residue class, see OpenFHE LastPrime) with `q == 1 mod m` and exactly
/// `n_bits` bits.
pub fn last_prime(n_bits: u32, m: u64) -> Result<u64> {
    if n_bits >= 63 {
        return Err(overflow("LastPrime", &format!("requested bit length {n_bits} exceeds 62")));
    }
    let q = 1u64 << n_bits;
    let r = q % m;
    let mut q_new = q + 1 - r;
    if r < 2 {
        q_new = q_new.checked_sub(m).ok_or_else(|| overflow("LastPrime", "underflow shrinking candidate"))?;
    }
    while !is_prime(q_new) {
        q_new = q_new.checked_sub(m).ok_or_else(|| overflow("LastPrime", "underflow shrinking candidate"))?;
    }
    if bit_length(q_new) != n_bits {
        return Err(Error::Params(format!(
            "LastPrime: requested {n_bits} bits, but returned {}. Please adjust parameters.",
            bit_length(q_new)
        )));
    }
    Ok(q_new)
}

/// Next prime above `q` in the class `q mod m` (OpenFHE NextPrime: starts at `q + m`).
pub fn next_prime(q: u64, m: u64) -> Result<u64> {
    let mut q_new = q.checked_add(m).ok_or_else(|| overflow("NextPrime", "overflow growing candidate"))?;
    while !is_prime(q_new) {
        q_new = q_new.checked_add(m).ok_or_else(|| overflow("NextPrime", "overflow growing candidate"))?;
    }
    Ok(q_new)
}

/// Previous prime below `q` in the class `q mod m` (OpenFHE PreviousPrime: starts at `q - m`).
pub fn previous_prime(q: u64, m: u64) -> Result<u64> {
    let mut q_new = q.checked_sub(m).ok_or_else(|| overflow("PreviousPrime", "underflow shrinking candidate"))?;
    while !is_prime(q_new) {
        q_new = q_new.checked_sub(m).ok_or_else(|| overflow("PreviousPrime", "underflow shrinking candidate"))?;
    }
    Ok(q_new)
}

/// The minimal primitive `m`-th root of unity mod prime `p`, `m` a power of two (OpenFHE RootOfUnity returns the
/// minimum over all primitive roots so that contexts agree; the generator it starts from is random, the minimum is
/// not). The primitive roots are the odd powers of any one of them.
pub fn root_of_unity(m: u64, p: u64) -> Result<u64> {
    if !m.is_power_of_two() || m < 2 {
        return Err(Error::Params(format!("RootOfUnity: cyclotomic order {m} is not a power of two")));
    }
    if (p - 1) % m != 0 {
        return Err(Error::Params(format!(
            "RootOfUnity: primeModulus = {p} and m = {m} do not satisfy (q-1)/m integer"
        )));
    }
    let e = (p - 1) / m;
    let mut w = 0;
    for g in 2..p {
        let c = powmod(g, e, p);
        if powmod(c, m / 2, p) == p - 1 {
            w = c;
            break;
        }
    }
    if w == 0 {
        return Err(Error::Params(format!("RootOfUnity: no primitive {m}-th root mod {p}")));
    }
    let w2 = mulmod(w, w, p);
    let mut x = w;
    let mut min = w;
    let mut k = 1;
    while k + 2 < m {
        x = mulmod(x, w2, p);
        k += 2;
        if x < min && x != 1 {
            min = x;
        }
    }
    Ok(min)
}

pub fn bit_length(x: u64) -> u32 {
    64 - x.leading_zeros()
}

/// Bit length of the product of `ps` (BigInteger::GetMSB / GetLengthForBase(2) of a modulus product).
pub fn product_bits(ps: &[u64]) -> u32 {
    let mut limbs: Vec<u64> = vec![1];
    for &p in ps {
        let mut carry: u128 = 0;
        for l in limbs.iter_mut() {
            let t = *l as u128 * p as u128 + carry;
            *l = t as u64;
            carry = t >> 64;
        }
        if carry > 0 {
            limbs.push(carry as u64);
        }
    }
    while limbs.len() > 1 && *limbs.last().unwrap_or(&1) == 0 {
        limbs.pop();
    }
    let top = *limbs.last().unwrap_or(&0);
    (limbs.len() as u32 - 1) * 64 + bit_length(top)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn primes_and_roots() {
        assert!(is_prime(268042241));
        assert!(!is_prime(268042241 * 3));
        assert!(!is_prime(3215031751)); // strong pseudoprime to bases 2,3,5,7
        let m = 1u64 << 17;
        let q = previous_prime(first_prime(28, m).unwrap(), m).unwrap();
        assert_eq!(q, 268042241);
        assert_eq!(root_of_unity(m, 268042241).unwrap(), 5274);
        assert_eq!(product_bits(&[1u64 << 63, 4]), 66);
        assert_eq!(product_bits(&[3, 5]), 4);
    }
}
