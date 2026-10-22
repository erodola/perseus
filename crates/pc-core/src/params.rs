//! CKKS parameter generation: `ChainSpec -> Chain` (the Q chain, the special primes P, their NTT roots and the hybrid
//! key-switching digit partition). Ported from the patched OpenFHE `ParamsGenCKKSRNSInternal` and the HYBRID part of
//! `CryptoParametersRNS::PrecomputeCRTTables`, so the n32 preset reproduces OpenFHE's chain prime for prime.
//!
//! Level conventions: Q prime ids `0..q.len()`, special prime ids `q.len()..q.len()+p.len()` (FIDESlib's id space);
//! one CKKS level is `level_unit = composite_degree` primes; "limb top" `l` means the primes `0..=l` are live.
pub mod check;
pub mod nbtheory;
pub mod primegen;

use crate::error::{Error, Result};
use crate::word::WordId;
use nbtheory::{first_prime, previous_prime, product_bits, root_of_unity};

/// HE-standard security level (OpenFHE `SecurityLevel`, classic estimates).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Security {
    NotSet,
    Classic128,
    Classic192,
    Classic256,
}

/// Secret-key distribution. Both are ternary for the HE-standard table: under sparse encapsulation the main key is
/// still uniform ternary (OpenFHE patch 0014), the sparse `h`-weight key only lives in the encapsulation step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Secret {
    UniformTernary,
    SparseEncapsulated { h: u32 },
}

/// Rescaling technique, as `ParamsGenCKKSRNSInternal` picks the prime-search routine from it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scaling {
    /// One prime per level, scaling factors tracked per level (OpenFHE FLEXIBLEAUTO).
    FlexibleAuto,
    /// `composite_degree > 1` primes per level (OpenFHE COMPOSITESCALINGMANUAL).
    CompositeManual,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChainSpec {
    pub logn: u32,
    /// Primes per CKKS level (1 = classic chain).
    pub composite_degree: u32,
    /// Bits of one level's modulus (OpenFHE scalingModSize).
    pub scaling_bits: u32,
    /// Bits of the bottom level's modulus q0 (OpenFHE firstModSize).
    pub first_mod_bits: u32,
    /// Number of levels including q0 (OpenFHE multiplicative depth + 1); the chain has `num_levels *
    /// composite_degree` primes.
    pub num_levels: u32,
    /// Hybrid key-switching digits (OpenFHE numLargeDigits).
    pub dnum: u32,
    /// Bit size of the special primes (OpenFHE auxBits: 28 under NATIVEINT 32, 60 under 64).
    pub special_bits: u32,
    pub security: Security,
    pub secret: Secret,
    /// Storage word; doubles as OpenFHE's registerWordSize (the composite prime-size guard).
    pub word: WordId,
    /// Mixed-size chain (patch 0002, FLEXIBLEAUTO only): bit size per level above q0, bottom-up, `num_levels - 1`
    /// entries.
    pub sizes_per_level: Option<Vec<u32>>,
}

impl ChainSpec {
    /// The shipped n32 chain: logN 16, composite 2, Delta 2^54, q0 56 bits, 27 levels, dnum 6, 28-bit specials.
    pub fn n32() -> Self {
        ChainSpec {
            logn: 16,
            composite_degree: 2,
            scaling_bits: 54,
            first_mod_bits: 56,
            num_levels: 27,
            dnum: 6,
            special_bits: 28,
            security: Security::Classic128,
            secret: Secret::UniformTernary,
            word: WordId::U32,
            sizes_per_level: None,
        }
    }
    /// The n64 chain (functional only): logN 16, 53-bit primes, 60-bit q0, depth 27, dnum 7, 60-bit specials.
    pub fn n64() -> Self {
        ChainSpec {
            logn: 16,
            composite_degree: 1,
            scaling_bits: 53,
            first_mod_bits: 60,
            num_levels: 28,
            dnum: 7,
            special_bits: 60,
            security: Security::Classic128,
            secret: Secret::UniformTernary,
            word: WordId::U64,
            sizes_per_level: None,
        }
    }
    pub fn preset(name: &str) -> Result<Self> {
        match name {
            "n32" => Ok(Self::n32()),
            "n64" => Ok(Self::n64()),
            _ => Err(Error::Params(format!("unknown chain preset '{name}' (known: n32, n64)"))),
        }
    }
    pub fn scaling(&self) -> Scaling {
        if self.composite_degree > 1 {
            Scaling::CompositeManual
        } else {
            Scaling::FlexibleAuto
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Chain {
    pub spec: ChainSpec,
    pub n: usize,
    /// Q primes, bottom (q0 pair) first.
    pub q: Vec<u64>,
    /// Special primes P.
    pub p: Vec<u64>,
    /// Minimal primitive 2N-th root of unity (psi) per Q prime, the NTT root OpenFHE and FIDESlib use.
    pub q_roots: Vec<u64>,
    pub p_roots: Vec<u64>,
    /// Hybrid digit partition: Q prime ids per digit (OpenFHE m_paramsPartQ; `ceil(|Q|/dnum)` primes each).
    pub digits: Vec<Vec<usize>>,
    /// Primes per CKKS level.
    pub level_unit: usize,
    pub word_bits: u32,
    pub scaling: Scaling,
    /// Bit length of Q*P (what the HE-standard check is run on).
    pub log_qp: u32,
}

/// OpenFHE AUXMODSIZE by native word.
fn aux_mod_size(word: WordId) -> u32 {
    match word {
        WordId::U64 => 60,
        WordId::U32 | WordId::U16 => 28,
    }
}

/// HE standard, ternary secret, classic estimates (OpenFHE `stdlatticeparms.cpp`): (ring dim, max log QP).
fn he_std_ternary(level: Security) -> &'static [(u32, u32)] {
    match level {
        Security::Classic128 => {
            &[(1024, 27), (2048, 54), (4096, 109), (8192, 218), (16384, 438), (32768, 881), (65536, 1747), (131072, 3523)]
        }
        Security::Classic192 => {
            &[(1024, 19), (2048, 37), (4096, 75), (8192, 152), (16384, 305), (32768, 611), (65536, 1199), (131072, 2411)]
        }
        Security::Classic256 => {
            &[(1024, 14), (2048, 29), (4096, 58), (8192, 118), (16384, 237), (32768, 476), (65536, 929), (131072, 1866)]
        }
        Security::NotSet => &[],
    }
}

/// StdLatticeParm::FindRingDim: the smallest standard ring dimension whose bound covers `log_q`.
pub fn find_ring_dim(level: Security, log_q: u32) -> u32 {
    let mut prev = 0;
    let mut n = 0;
    for &(dim, max_log_q) in he_std_ternary(level) {
        if log_q <= max_log_q && log_q > prev {
            return dim;
        }
        prev = max_log_q;
        n = dim;
    }
    2 * n
}

/// The max log QP the HE standard allows at ring dimension `n` (None when unconstrained).
pub fn max_log_qp(level: Security, n: u32) -> Option<u32> {
    he_std_ternary(level).iter().find(|e| e.0 == n).map(|e| e.1)
}

/// CryptoParametersRNS::EstimateLogP -> (log P estimate, number of special primes).
fn estimate_log_p(
    num_part_q: u32,
    first_mod: f64,
    dcrt_bits: f64,
    num_primes: u32,
    aux_bits: u32,
    add_one: bool,
) -> Result<(u32, u32)> {
    let size_q = num_primes as usize;
    let num_part_q = (num_part_q as usize).min(size_q);
    let per = (size_q as f64 / num_part_q as f64).ceil() as usize;
    if size_q <= per * (num_part_q - 1) {
        return Err(Error::Params(format!(
            "HYBRID key switching parameters: Can't appropriately distribute {size_q} towers into {num_part_q} digits. \
             Please select different number of digits (dnum)."
        )));
    }
    let mut qi = vec![dcrt_bits; size_q];
    qi[0] = first_mod;
    let mut max_bits = 0u32;
    for j in 0..num_part_q {
        let start = j * per;
        let end = ((j + 1) * per - 1).min(size_q - 1);
        let bits = qi[start..=end].iter().fold(0.0, |a, &b| a + b) as u32;
        max_bits = max_bits.max(bits);
    }
    if add_one && max_bits != aux_bits {
        max_bits += 1;
    }
    let size_p = (max_bits as f64 / aux_bits as f64).ceil() as u32;
    Ok((size_p * aux_bits, size_p))
}

impl Chain {
    /// ParamsGenCKKSRNSInternal (+ the HYBRID prime/partition part of PrecomputeCRTTables).
    pub fn generate(spec: &ChainSpec) -> Result<Chain> {
        let n = 1u64 << spec.logn;
        let cycl_order = 2 * n;
        let scaling = spec.scaling();
        let d = spec.composite_degree;
        let rws = spec.word.bits();
        let max_prime_bits = match spec.word {
            WordId::U16 => 15,
            WordId::U32 => 31,
            WordId::U64 => 60,
        };
        if d < 1 {
            return Err(Error::Params("composite_degree must be >= 1".into()));
        }
        if spec.num_levels < 1 || spec.dnum < 1 {
            return Err(Error::Params(format!(
                "num_levels ({}) and dnum ({}) must be >= 1",
                spec.num_levels, spec.dnum
            )));
        }
        if scaling == Scaling::CompositeManual && d > 2 && spec.scaling_bits < 60 {
            return Err(Error::Params(format!(
                "composite_degree {d} with scaling_bits {} < 60: prime moduli size is too small",
                spec.scaling_bits
            )));
        }
        if let Some(sz) = &spec.sizes_per_level {
            if scaling != Scaling::FlexibleAuto {
                return Err(Error::Params("sizes_per_level requires composite_degree 1 (FLEXIBLEAUTO)".into()));
            }
            if sz.len() != spec.num_levels as usize - 1 {
                return Err(Error::Params(format!(
                    "sizes_per_level must have num_levels - 1 = {} entries, got {}",
                    spec.num_levels - 1,
                    sz.len()
                )));
            }
            if sz.iter().any(|&s| !(20..=59).contains(&s)) {
                return Err(Error::Params("sizes_per_level entries must be in [20, 59]".into()));
            }
        }
        let aux_bits = spec.special_bits;
        if aux_bits == 0 || aux_bits > max_prime_bits {
            return Err(Error::Params(format!(
                "special_bits {aux_bits} out of range for a {}-bit word (max {max_prime_bits})",
                rws
            )));
        }
        // OpenFHE derives auxBits; the spec carries it explicitly, so refuse a value OpenFHE would not have used.
        let openfhe_aux = if scaling == Scaling::CompositeManual && rws <= aux_mod_size(spec.word) {
            rws - 1
        } else {
            aux_mod_size(spec.word)
        };
        if aux_bits != openfhe_aux {
            return Err(Error::Params(format!(
                "special_bits {aux_bits} differs from OpenFHE's auxBits {openfhe_aux} for this word/scaling"
            )));
        }

        // HE-standard estimate (before the search).
        if spec.security != Security::NotSet {
            let mut q_bound = spec.first_mod_bits + (spec.num_levels - 1) * spec.scaling_bits;
            let mut sf_for_p = spec.scaling_bits;
            if let Some(sz) = &spec.sizes_per_level {
                q_bound = spec.first_mod_bits;
                for &s in sz {
                    q_bound += s;
                    sf_for_p = sf_for_p.max(s);
                }
            }
            if q_bound != aux_bits {
                q_bound += 1;
            }
            q_bound += estimate_log_p(
                spec.dnum,
                spec.first_mod_bits as f64,
                sf_for_p as f64,
                spec.num_levels,
                aux_bits,
                true,
            )?
            .0;
            let he_n = find_ring_dim(spec.security, q_bound);
            if he_n as u64 > n {
                return Err(Error::Params(format!(
                    "ring dimension {n} does not comply with the HE standard ({he_n} needed for estimated log QP {q_bound})"
                )));
            }
        }

        let num_primes = spec.num_levels * d;
        let mut q = vec![0u64; num_primes as usize];
        match (scaling, &spec.sizes_per_level) {
            (Scaling::CompositeManual, _) => primegen::composite_prime_moduli_gen(
                &mut q,
                d,
                num_primes,
                spec.first_mod_bits,
                spec.scaling_bits,
                cycl_order,
                rws,
            )?,
            (Scaling::FlexibleAuto, Some(sz)) => {
                primegen::mixed_prime_moduli_gen(&mut q, num_primes, spec.first_mod_bits, sz, cycl_order)?
            }
            (Scaling::FlexibleAuto, None) => primegen::single_prime_moduli_gen(
                &mut q,
                true,
                num_primes,
                spec.first_mod_bits,
                spec.scaling_bits,
                cycl_order,
            )?,
        }
        if let Some(&big) = q.iter().find(|&&x| nbtheory::bit_length(x) > max_prime_bits) {
            return Err(Error::Params(format!(
                "prime {big} ({} bits) does not fit the {rws}-bit word (max {max_prime_bits} bits)",
                nbtheory::bit_length(big)
            )));
        }

        // Hybrid digit partition and special primes (PrecomputeCRTTables, HYBRID).
        let size_q = q.len();
        let dnum = spec.dnum as usize;
        let a = (size_q as f64 / dnum as f64).ceil() as usize;
        if size_q <= a * (dnum - 1) {
            return Err(Error::Params(format!(
                "HYBRID key switching parameters: Can't appropriately distribute {size_q} towers into {dnum} digits. \
                 Please select different number of digits (dnum)."
            )));
        }
        let digits: Vec<Vec<usize>> = (0..dnum).map(|j| (j * a..((j + 1) * a).min(size_q)).collect()).collect();
        let max_bits = digits.iter().map(|dg| product_bits(&dg.iter().map(|&i| q[i]).collect::<Vec<_>>())).max();
        let max_bits = max_bits.unwrap_or(0);
        let size_p = (max_bits as f64 / aux_bits as f64).ceil() as usize;
        let mut p = Vec::with_capacity(size_p);
        let mut p_prev = first_prime(aux_bits, cycl_order)?;
        for _ in 0..size_p {
            let mut cand;
            loop {
                cand = previous_prime(p_prev, cycl_order)?;
                p_prev = cand;
                if !q.contains(&cand) {
                    break;
                }
            }
            p.push(cand);
        }

        let q_roots = q.iter().map(|&x| root_of_unity(cycl_order, x)).collect::<Result<Vec<_>>>()?;
        let p_roots = p.iter().map(|&x| root_of_unity(cycl_order, x)).collect::<Result<Vec<_>>>()?;

        let log_qp = product_bits(&q.iter().chain(p.iter()).copied().collect::<Vec<_>>());
        if spec.security != Security::NotSet {
            let n_actual = find_ring_dim(spec.security, log_qp);
            if n < n_actual as u64 {
                return Err(Error::Params(format!(
                    "the ring dimension [{n}] does not meet security requirements (log QP {log_qp} needs {n_actual})"
                )));
            }
        }

        Ok(Chain {
            spec: spec.clone(),
            n: n as usize,
            q,
            p,
            q_roots,
            p_roots,
            digits,
            level_unit: d as usize,
            word_bits: rws,
            scaling,
            log_qp,
        })
    }

    /// Top limb index of the full chain (FIDESlib `L`).
    pub fn l(&self) -> usize {
        self.q.len() - 1
    }
    pub fn dnum(&self) -> usize {
        self.digits.len()
    }
    /// All primes by id: Q then P.
    pub fn primes(&self) -> Vec<u64> {
        self.q.iter().chain(self.p.iter()).copied().collect()
    }
    pub fn roots(&self) -> Vec<u64> {
        self.q_roots.iter().chain(self.p_roots.iter()).copied().collect()
    }
    pub fn inv_roots(&self) -> Vec<u64> {
        self.primes().iter().zip(self.roots()).map(|(&p, r)| crate::modmul::invmod(r, p)).collect()
    }
    pub fn special_ids(&self) -> Vec<usize> {
        (self.q.len()..self.q.len() + self.p.len()).collect()
    }
    /// The ModUp target basis of digit `d` at the top level, in FIDESlib's order (`digitMeta`): the special primes,
    /// then every Q prime outside the digit in id order.
    pub fn digit_complement(&self, d: usize) -> Vec<usize> {
        let mut v = self.special_ids();
        v.extend((0..self.q.len()).filter(|i| !self.digits[d].contains(i)));
        v
    }
    /// The digit a Q prime belongs to.
    pub fn digit_of(&self, id: usize) -> Option<usize> {
        self.digits.iter().position(|dg| dg.contains(&id))
    }
    /// FIDESlib `prime_bits`: bit width of each prime id.
    pub fn prime_bits(&self) -> Vec<u32> {
        self.primes().iter().map(|&p| nbtheory::bit_length(p)).collect()
    }
    /// FIDESlib `Constants::type`: bit i set when prime id i needs a 64-bit limb (p > 2^31, `pickLimbType`).
    pub fn limb_type_mask(&self) -> u64 {
        self.primes().iter().enumerate().filter(|(_, &p)| p > (1u64 << 31)).fold(0u64, |m, (i, _)| m | 1u64.checked_shl(i as u32).unwrap_or(0))
    }
    /// Is limb top `l` on the composite level grid (a ciphertext can sit there)?
    pub fn on_grid(&self, l: usize) -> bool {
        (self.l() - l) % self.level_unit == 0
    }
}
