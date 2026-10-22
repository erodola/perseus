//! Q-chain prime search, ported operation for operation from the patched OpenFHE
//! `pke/lib/scheme/ckksrns/ckksrns-parametergeneration.cpp` (`CompositePrimeModuliGen`, `SinglePrimeModuliGen`,
//! and patch 0002's `MixedPrimeModuliGen`). The double arithmetic (scaling-factor recursion, `log2` guards, `llround`)
//! is kept in the same order so the chosen primes are identical; `std::pow(x, 2)` is `x * x` (both compilers fold it).
//! Only the FLEXIBLEAUTO / COMPOSITESCALINGMANUAL paths the stack uses are ported (no FLEXIBLEAUTOEXT extra prime).
use super::nbtheory::{first_prime, last_prime, next_prime, previous_prime};
use crate::error::{Error, Result};
use std::collections::HashSet;

fn l2(x: u64) -> f64 {
    (x as f64).log2()
}

fn llround(x: f64) -> Result<u64> {
    let r = x.round();
    if !(0.0..1.8e19).contains(&r) {
        return Err(Error::Params(format!("prime search: scaling factor {x:e} does not fit a native integer")));
    }
    Ok(r as u64)
}

const COMPOSITE_ERR: &str =
    "COMPOSITE SCALING prime sampling error. Consider increasing the scaling factor or the register word size.";

fn ce<T>(r: Result<T>) -> Result<T> {
    r.map_err(|_| Error::Params(COMPOSITE_ERR.to_string()))
}

/// CompositePrimeModuliGen: `moduli.len() == num_primes` (already multiplied by the composite degree).
pub fn composite_prime_moduli_gen(
    moduli: &mut [u64],
    composite_degree: u32,
    num_primes: u32,
    first_mod_size: u32,
    dcrt_bits: u32,
    cycl_order: u64,
    register_word_size: u32,
) -> Result<()> {
    if first_mod_size <= dcrt_bits {
        return Err(Error::Params("firstModSize must be > scalingModSize.".into()));
    }
    let d = composite_degree as usize;
    let np = num_primes as usize;
    let rws = register_word_size as f64;
    let mut record: HashSet<u64> = HashSet::new();

    let mut rem_bits = dcrt_bits;
    for dd in 1..=d {
        let q_bit_size = (rem_bits as f64 / (d - dd + 1) as f64).ceil() as u32;
        let mut q = first_prime(q_bit_size, cycl_order)?;
        q = previous_prime(q, cycl_order)?;
        while l2(q) > rws || l2(q) > q_bit_size as f64 || record.contains(&q) {
            q = previous_prime(q, cycl_order)?;
        }
        moduli[np - dd] = q;
        record.insert(q);
        rem_bits -= l2(q).ceil() as u32;
    }

    if np > 1 {
        let n_prev = d.div_ceil(2);
        let mut q_prev = vec![0u64; n_prev];
        let mut q_next = vec![0u64; d - n_prev];

        let mut sf = moduli[np - 1] as f64;
        for dd in 2..=d {
            sf *= moduli[np - dd] as f64;
        }

        let mut flag = true;
        let mut i = np - d;
        while i >= 2 * d {
            sf = sf * sf;
            for dd in 0..d {
                sf /= moduli[i + dd] as f64;
            }
            let sf_sqrt = sf.powf(1.0 / d as f64);
            let sf_int = llround(sf_sqrt)?;
            let sf_rem = sf_int % cycl_order;

            let mut prime_product = 1.0f64;
            let mut cur: HashSet<u64> = HashSet::new();

            for step in 0..q_prev.len() {
                q_prev[step] = (sf_int - sf_rem + 1)
                    .checked_sub(cycl_order)
                    .ok_or_else(|| Error::Params(COMPOSITE_ERR.to_string()))?;
                loop {
                    q_prev[step] = ce(previous_prime(q_prev[step], cycl_order))?;
                    let v = q_prev[step];
                    if !(l2(v) > rws || record.contains(&v) || cur.contains(&v)) {
                        break;
                    }
                }
                cur.insert(q_prev[step]);
                prime_product *= q_prev[step] as f64;
            }

            let mut fits_register = true;
            for step in 0..q_next.len() {
                q_next[step] = sf_int - sf_rem + 1 + cycl_order;
                loop {
                    q_next[step] = if fits_register {
                        ce(next_prime(q_next[step], cycl_order))?
                    } else {
                        ce(previous_prime(q_next[step], cycl_order))?
                    };
                    if l2(q_next[step]) > rws {
                        fits_register = false;
                    }
                    let v = q_next[step];
                    if !(l2(v) > rws || record.contains(&v) || cur.contains(&v)) {
                        break;
                    }
                }
                cur.insert(q_next[step]);
                prime_product *= q_next[step] as f64;
            }

            let last_next = q_next.len().wrapping_sub(1);
            let last_prev = q_prev.len() - 1;
            if !flag {
                let mut q_prev_next = q_next[last_next];
                while prime_product > sf {
                    loop {
                        cur.remove(&q_prev_next);
                        q_prev_next = ce(previous_prime(q_prev_next, cycl_order))?;
                        let v = q_prev_next;
                        if !(l2(v) > rws || record.contains(&v) || cur.contains(&v)) {
                            break;
                        }
                    }
                    cur.insert(q_prev_next);
                    prime_product /= q_next[last_next] as f64;
                    q_next[last_next] = q_prev_next;
                    prime_product *= q_prev_next as f64;
                }
                flag = true;
            } else {
                let mut q_next_prev = q_prev[last_prev];
                fits_register = true;
                while prime_product < sf {
                    loop {
                        cur.remove(&q_next_prev);
                        q_next_prev = if fits_register {
                            ce(next_prime(q_next_prev, cycl_order))?
                        } else {
                            ce(previous_prime(q_next_prev, cycl_order))?
                        };
                        if l2(q_next_prev) > rws {
                            fits_register = false;
                        }
                        let v = q_next_prev;
                        if !(l2(v) > rws || record.contains(&v) || cur.contains(&v)) {
                            break;
                        }
                    }
                    cur.insert(q_next_prev);
                    prime_product /= q_prev[last_prev] as f64;
                    q_prev[last_prev] = q_next_prev;
                    prime_product *= q_next_prev as f64;
                }
                flag = false;
            }

            let m = q_prev.len();
            for dd in 1..=m {
                moduli[i - dd] = q_prev[dd - 1];
            }
            for dd in m + 1..=d {
                moduli[i - dd] = q_next[dd - (m + 1)];
            }
            for dd in 1..=d {
                record.insert(moduli[i - dd]);
            }
            i -= d;
        }
    }

    let mut rem_bits = first_mod_size;
    for dd in 1..=d {
        let q_bit_size = (rem_bits as f64 / (d - dd + 1) as f64).ceil() as u32;
        let mut next = ce(first_prime(q_bit_size, cycl_order))?;
        next = ce(previous_prime(next, cycl_order))?;
        while l2(next) > q_bit_size as f64 || l2(next) > rws || record.contains(&next) {
            next = ce(previous_prime(next, cycl_order))?;
        }
        moduli[dd - 1] = next;
        record.insert(next);
        rem_bits -= q_bit_size;
    }
    Ok(())
}

/// SinglePrimeModuliGen, FLEXIBLEAUTO branch (`flexible == true`) or the alternating FIXED* branch.
pub fn single_prime_moduli_gen(
    moduli: &mut [u64],
    flexible: bool,
    num_primes: u32,
    first_mod_size: u32,
    dcrt_bits: u32,
    cycl_order: u64,
) -> Result<()> {
    let np = num_primes as usize;
    let q = first_prime(dcrt_bits, cycl_order)?;
    moduli[np - 1] = q;
    let mut max_prime = q;
    let mut min_prime = q;
    if np > 1 {
        if !flexible {
            let mut q_prev = q;
            let mut q_next = q;
            let mut cnt = 0usize;
            for i in (1..=np - 2).rev() {
                if cnt % 2 == 0 {
                    q_prev = previous_prime(q_prev, cycl_order)?;
                    moduli[i] = q_prev;
                } else {
                    q_next = next_prime(q_next, cycl_order)?;
                    moduli[i] = q_next;
                }
                if moduli[i] > max_prime {
                    max_prime = moduli[i];
                } else if moduli[i] < min_prime {
                    min_prime = moduli[i];
                }
                cnt += 1;
            }
        } else {
            let mut sf = moduli[np - 1] as f64;
            let mut cnt = 0usize;
            for i in (1..=np.saturating_sub(2)).rev() {
                sf = sf * sf / moduli[i + 1] as f64;
                pick_flexible(moduli, i, np, sf, cnt, cycl_order)?;
                if moduli[i] > max_prime {
                    max_prime = moduli[i];
                } else if moduli[i] < min_prime {
                    min_prime = moduli[i];
                }
                cnt += 1;
            }
        }
    }
    set_first_modulus(moduli, first_mod_size, first_mod_size == dcrt_bits, max_prime, cycl_order)
}

/// One FLEXIBLEAUTO step: the prime nearest below (even `cnt`) or above (odd) the running scaling factor, distinct
/// from every prime above it.
fn pick_flexible(moduli: &mut [u64], i: usize, np: usize, sf: f64, cnt: usize, cycl_order: u64) -> Result<()> {
    let sf_int = llround(sf)?;
    let sf_rem = sf_int % cycl_order;
    let mut has_same = true;
    if cnt % 2 == 0 {
        let mut q_prev = (sf_int + 1)
            .checked_sub(cycl_order + sf_rem)
            .ok_or_else(|| Error::Params(format!("prime search: scaling factor {sf:e} too small")))?;
        while has_same {
            has_same = false;
            q_prev = previous_prime(q_prev, cycl_order)?;
            if moduli[i + 1..np].contains(&q_prev) {
                has_same = true;
            }
        }
        moduli[i] = q_prev;
    } else {
        let mut q_next = sf_int + cycl_order - sf_rem + 1;
        while has_same {
            has_same = false;
            q_next = next_prime(q_next, cycl_order)?;
            if moduli[i + 1..np].contains(&q_next) {
                has_same = true;
            }
        }
        moduli[i] = q_next;
    }
    Ok(())
}

fn set_first_modulus(moduli: &mut [u64], first_mod_size: u32, same_size: bool, max_prime: u64, m: u64) -> Result<()> {
    if same_size {
        moduli[0] = next_prime(max_prime, m)?;
    } else {
        moduli[0] = last_prime(first_mod_size, m)?;
        let q0 = moduli[0];
        if moduli[1..].contains(&q0) {
            moduli[0] = next_prime(max_prime, m)?;
        }
    }
    Ok(())
}

/// MixedPrimeModuliGen (patch 0002): `sizes_per_level[i-1]` is the bit size of `moduli[i]`, bottom-up; the
/// scaling-factor target resets to `2^size` at every segment boundary.
pub fn mixed_prime_moduli_gen(
    moduli: &mut [u64],
    num_primes: u32,
    first_mod_size: u32,
    sizes_per_level: &[u32],
    cycl_order: u64,
) -> Result<()> {
    let np = num_primes as usize;
    for i in 1..np.saturating_sub(1) {
        let (a, b) = (sizes_per_level[i - 1], sizes_per_level[i]);
        if a != b && a.abs_diff(b) < 2 {
            return Err(Error::Params("scalingModSizePerLevel: adjacent segments must differ by >= 2 bits.".into()));
        }
    }
    let top = sizes_per_level[np - 2];
    let q = first_prime(top, cycl_order)?;
    moduli[np - 1] = q;
    let mut max_prime = q;
    let mut min_prime = q;
    let mut sf = moduli[np - 1] as f64;
    let mut cnt = 0usize;
    for i in (1..=np.saturating_sub(2)).rev() {
        let here = sizes_per_level[i - 1];
        let above = sizes_per_level[i];
        if here == above {
            sf = sf * sf / moduli[i + 1] as f64;
        } else {
            sf = 2f64.powf(here as f64);
        }
        pick_flexible(moduli, i, np, sf, cnt, cycl_order)?;
        if moduli[i] > max_prime {
            max_prime = moduli[i];
        } else if moduli[i] < min_prime {
            min_prime = moduli[i];
        }
        cnt += 1;
    }
    set_first_modulus(moduli, first_mod_size, first_mod_size == sizes_per_level[0], max_prime, cycl_order)
}
