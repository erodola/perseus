//! Chebyshev-series utilities of the EvalMod polynomial: OpenFHE `Degree`, `LongDivisionChebyshev`,
//! `ComputeDegreesPS`, `GetDepthByDegree` (ckksrns-utils), and the Paterson-Stockmeyer coefficient plan that
//! `EvalChebyshevSeriesPS` derives before its recursion (k, m, the T_{k 2^(m-1)} division and the c/s split) —
//! the same quantities OpenFHE's dumped `params.json::cheb_ps` holds and the EvalMod builder consumes.
//! Coefficient convention: OpenFHE's (c0 NOT halved in storage; the evaluator adds c0 / 2).
use crate::error::{Error, Result};

/// OpenFHE `Degree(coefficients)` (delta 0): index of the last non-zero coefficient, 0 when all are zero.
pub fn degree(c: &[f64]) -> usize {
    let mut i = c.len();
    while i > 0 {
        i -= 1;
        if c[i] != 0.0 {
            return i;
        }
    }
    0
}

/// OpenFHE `IsNotEqualOne` (delta 2^-44).
fn not_one(v: f64) -> bool {
    (v - 1.0).abs() > f64::from_bits(0x3d30_0000_0000_0000) // 0x1p-44
}

/// Quotient and remainder of a Chebyshev-basis division (OpenFHE `longDiv<double>`).
#[derive(Clone, Debug, PartialEq)]
pub struct LongDiv {
    pub q: Vec<f64>,
    pub r: Vec<f64>,
}

/// OpenFHE `LongDivisionChebyshev(f, g)`: f = q * g + r in the Chebyshev basis, operation for operation (same
/// rounding). Errors (as OpenFHE throws) when a leading coefficient is zero.
pub fn long_division(f: &[f64], g: &[f64]) -> Result<LongDiv> {
    if f.is_empty() || g.is_empty() {
        return Err(Error::BtsSetup("long_division: empty polynomial".into()));
    }
    let mut n = degree(f);
    if n != f.len() - 1 {
        return Err(Error::BtsSetup("long_division: the dominant coefficient of the dividend is zero".into()));
    }
    let k = degree(g);
    if k != g.len() - 1 {
        return Err(Error::BtsSetup("long_division: the dominant coefficient of the divisor is zero".into()));
    }
    if n < k {
        return Ok(LongDiv { q: vec![0.0], r: f.to_vec() });
    }
    let mut q = vec![0.0f64; n - k + 1];
    let mut r = f.to_vec();
    let mut d: Vec<f64> = Vec::with_capacity(g.len() + n);
    let gk = g[k];
    let mut stalls = 0;

    while n > k {
        d.clear();
        d.resize(n + 1, 0.0);
        q[n - k] = 2.0 * *r.last().unwrap_or(&0.0);
        if not_one(gk) {
            q[n - k] /= *g.last().unwrap_or(&1.0);
        }
        let nk = n - k;
        if k == nk {
            d[0] = 2.0 * g[nk];
            for i in 1..2 * k + 1 {
                d[i] = g[(nk as i64 - i as i64).unsigned_abs() as usize];
            }
        } else if k > nk {
            d[0] = 2.0 * g[nk];
            for i in 1..k - nk + 1 {
                d[i] = g[(nk as i64 - i as i64).unsigned_abs() as usize] + g[nk + i];
            }
            for i in k - nk + 1..n + 1 {
                d[i] = g[(i as i64 - n as i64 + k as i64).unsigned_abs() as usize];
            }
        } else {
            d[nk] = g[0];
            for i in n - 2 * k..n + 1 {
                if i != nk {
                    d[i] = g[(i as i64 - n as i64 + k as i64).unsigned_abs() as usize];
                }
            }
        }
        let rl = *r.last().unwrap_or(&0.0);
        if not_one(rl) {
            for x in d.iter_mut() {
                *x *= rl;
            }
        }
        let gl = *g.last().unwrap_or(&1.0);
        if not_one(gl) {
            for x in d.iter_mut() {
                *x /= gl;
            }
        }
        // f -= d over r's length (std::transform over r)
        for (x, y) in r.iter_mut().zip(d.iter()) {
            *x -= y;
        }
        let prev = n;
        if r.len() > 1 {
            n = degree(&r);
            r.truncate(n + 1);
        }
        // When g's leading coefficient is not a power of two, (r_n g_k) / g_k may not cancel r_n exactly: OpenFHE
        // then repeats the step on the residue (overwriting q[n - k]) until it does. Same here, but bounded.
        if n >= prev {
            stalls += 1;
            if stalls > 64 {
                return Err(Error::BtsSetup(format!(
                    "long_division: degree {n} does not drop (divisor leading coefficient {gk} never cancels)"
                )));
            }
        }
    }

    if n == k {
        let mut d = g.to_vec();
        q[0] = *r.last().unwrap_or(&0.0);
        let gl = *g.last().unwrap_or(&1.0);
        if not_one(gl) {
            q[0] /= gl;
        }
        let rl = *r.last().unwrap_or(&0.0);
        if not_one(rl) {
            for x in d.iter_mut() {
                *x *= rl;
            }
        }
        if not_one(gl) {
            for x in d.iter_mut() {
                *x /= gl;
            }
        }
        for (x, y) in r.iter_mut().zip(d.iter()) {
            *x -= y;
        }
        if r.len() > 1 {
            n = degree(&r);
            r.truncate(n + 1);
        }
    }
    q[0] *= 2.0; // [c0] in the first spot, not [c0/2]
    Ok(LongDiv { q, r })
}

/// OpenFHE `ComputeDegreesPS(n)` -> (k, m): n < k (2^m - 1), table for n <= 2204, heuristic above.
pub fn compute_degrees_ps(n: u32) -> Result<(u32, u32)> {
    if n == 0 {
        return Err(Error::BtsSetup("compute_degrees_ps: degree 0 (nothing to evaluate)".into()));
    }
    const RANGE: [(u32, u32); 16] = [
        (2, 1),
        (11, 2),
        (13, 3),
        (17, 2),
        (55, 3),
        (59, 4),
        (76, 3),
        (239, 4),
        (247, 5),
        (284, 4),
        (991, 5),
        (1007, 6),
        (1083, 5),
        (2015, 6),
        (2031, 7),
        (2204, 6),
    ];
    if n <= 2204 {
        let m = RANGE.iter().find(|r| n <= r.0).map(|r| r.1).unwrap_or(6);
        let k = n / ((1u32 << m) - 1) + 1;
        return Ok((k, m));
    }
    // heuristic for larger degrees (verbatim, including its floating-point tests)
    let mut best: Option<(u32, u32, u32)> = None;
    let target = ((n / 2) as f64).sqrt().log2().floor();
    for k in 1..=n {
        let mmax = (((n / k) as f64).log2() + 1.0).ceil() as u32 + 1;
        for m in 1..=mmax {
            if (n as u64) < k as u64 * ((1u64 << m) - 1) && ((k as f64).log2().floor() - target).abs() <= 1.0 {
                let mults = k + 2 * m + (1u32 << (m - 1)) - 4;
                if best.map_or(true, |b| mults < b.2) {
                    best = Some((k, m, mults));
                }
            }
        }
    }
    best.map(|b| (b.0, b.1)).ok_or_else(|| Error::BtsSetup(format!("compute_degrees_ps: no (k, m) for n={n}")))
}

/// OpenFHE `GetDepthByDegree(d)`: multiplicative depth of a degree-d Chebyshev evaluation, d in [5, 261631].
pub fn depth_by_degree(d: usize) -> Result<u32> {
    const RANGE: [(usize, u32); 17] = [
        (4, 3),
        (5, 4),
        (13, 5),
        (27, 6),
        (59, 7),
        (119, 8),
        (247, 9),
        (495, 10),
        (1007, 11),
        (2031, 12),
        (4031, 13),
        (8127, 14),
        (16255, 15),
        (32639, 16),
        (65279, 17),
        (130815, 18),
        (261631, 19),
    ];
    if !(5..=261631).contains(&d) {
        return Err(Error::BtsSetup(format!("depth_by_degree: degree {d} not in [5, 261631]")));
    }
    Ok(RANGE.iter().find(|r| d <= r.0).map(|r| r.1).unwrap_or(19))
}

/// The Paterson-Stockmeyer coefficient plan of `EvalChebyshevSeriesPS` (top level, before any recursion):
/// f2 = f padded to 2 k2m2k + k + 1 with a leading 1 (when padded), divqr = f2 / T_{k2m2k + k}, r2 = divqr.r -
/// T_{k2m2k}, divcs = r2 / divqr.q, s2 = divcs.r padded to k2m2k + 1 with a leading 1; k2m2k = k 2^(m-1) - k.
/// Field-for-field the d3 dump's `cheb_ps` (fx dump tool `cheb_ps`), which the lazy-PS EvalMod builder reads
/// (it uses k, m, divqr.q, divcs.q, s2).
#[derive(Clone, Debug, PartialEq)]
pub struct PsPlan {
    pub n: usize,
    pub k: usize,
    pub m: usize,
    pub f2: Vec<f64>,
    pub divqr: LongDiv,
    pub r2: Vec<f64>,
    pub divcs: LongDiv,
    pub s2: Vec<f64>,
}

impl PsPlan {
    pub fn new(coefficients: &[f64]) -> Result<Self> {
        let n = degree(coefficients);
        let mut f2c = coefficients.to_vec();
        f2c.truncate(n + 1);
        let (k, m) = compute_degrees_ps(n as u32)?;
        let (k, m) = (k as usize, m as usize);
        let k2m2k = k * (1 << (m - 1)) - k;
        let mut f2 = f2c.clone();
        f2.resize(2 * k2m2k + k + 1, 0.0);
        if f2.len() > f2c.len() {
            if let Some(x) = f2.last_mut() {
                *x = 1.0;
            }
        }
        let mut tkm = vec![0.0f64; k2m2k + k + 1];
        if let Some(x) = tkm.last_mut() {
            *x = 1.0;
        }
        let divqr = long_division(&f2, &tkm)?;
        let mut r2 = divqr.r.clone();
        if k2m2k as i64 - degree(&divqr.r) as i64 <= 0 {
            r2[k2m2k] -= 1.0;
            let d = degree(&r2);
            r2.truncate(d + 1);
        } else {
            r2.resize(k2m2k + 1, 0.0);
            if let Some(x) = r2.last_mut() {
                *x = -1.0;
            }
        }
        let divcs = long_division(&r2, &divqr.q)?;
        let mut s2 = divcs.r.clone();
        s2.resize(k2m2k + 1, 0.0);
        if let Some(x) = s2.last_mut() {
            *x = 1.0;
        }
        Ok(PsPlan { n, k, m, f2: f2c, divqr, r2, divcs, s2 })
    }

    /// Whether the single-level lazy-PS shape applies (no InnerEvalChebyshevPS recursion).
    pub fn is_single_level(&self) -> bool {
        self.m == 2 && degree(&self.divqr.q) <= self.k && degree(&self.s2) <= self.k
    }
}
