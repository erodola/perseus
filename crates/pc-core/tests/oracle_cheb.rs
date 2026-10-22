//! cheb: `long_division` / `PsPlan` reproduce every division the d3 dump records (bit for bit), and the frozen
//! EvalMod table equals the dump's.
mod oracle_common;
use oracle_common::*;
use pc_core::bts_setup::{evalmod_coeffs, table_checksum, EVALMOD_DA_ITS, EVALMOD_K, SPARSE_B_CHEB_K24};
use pc_core::cheb::*;

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn ps_plan_reproduces_the_dump() {
    let p = params();
    let cheb = f64v(&p["cheb"]);
    let d = &p["cheb_ps"];
    let ps = PsPlan::new(&cheb).unwrap();
    assert_eq!(ps.n as u64, d["n"].as_u64().unwrap());
    assert_eq!(ps.k as u64, d["k"].as_u64().unwrap());
    assert_eq!(ps.m as u64, d["m"].as_u64().unwrap());
    let fields: [(&str, &Vec<f64>); 7] = [
        ("f2", &ps.f2),
        ("divqr_q", &ps.divqr.q),
        ("divqr_r", &ps.divqr.r),
        ("r2", &ps.r2),
        ("divcs_q", &ps.divcs.q),
        ("divcs_r", &ps.divcs.r),
        ("s2", &ps.s2),
    ];
    let mut n = 0;
    for (k, v) in fields {
        assert_eq!(bits(v), bits(&f64v(&d[k])), "cheb_ps.{k} differs");
        n += v.len();
    }
    assert_eq!(degree(&ps.divqr.q) as u64, d["deg_divqr_q"].as_u64().unwrap());
    assert_eq!(degree(&ps.divcs.q) as u64, d["deg_divcs_q"].as_u64().unwrap());
    assert_eq!(degree(&ps.s2) as u64, d["deg_s2"].as_u64().unwrap());
    assert!(ps.is_single_level());
    eprintln!("[oracle cheb] 2 long divisions (divqr, divcs) + r2/s2: {n} coefficients bit-identical to d3 cheb_ps");
}

#[test]
fn evalmod_table_is_the_dumped_one() {
    let p = params();
    let cheb = f64v(&p["cheb"]);
    assert_eq!(bits(evalmod_coeffs()), bits(&cheb));
    assert_eq!(EVALMOD_DA_ITS as u64, p["double_angle_its"].as_u64().unwrap());
    assert_eq!(EVALMOD_K, p["boot_k"].as_f64().unwrap());
    eprintln!(
        "[oracle cheb] EvalMod table: deg {}, K {EVALMOD_K}, {EVALMOD_DA_ITS} double angles, fnv1a64 {:#018x}; SparseB K24 deg {} fnv1a64 {:#018x}",
        evalmod_coeffs().len() - 1,
        table_checksum(evalmod_coeffs()),
        SPARSE_B_CHEB_K24.len() - 1,
        table_checksum(&SPARSE_B_CHEB_K24)
    );
}

/// Chebyshev series value (OpenFHE convention: c0 is the full coefficient; T-basis sum c0/2 + sum c_i T_i).
fn cheb_eval(c: &[f64], x: f64, half_c0: bool) -> f64 {
    let (mut t0, mut t1) = (1.0, x);
    let mut acc = if half_c0 { c[0] / 2.0 } else { c[0] };
    for (i, &ci) in c.iter().enumerate().skip(1) {
        if i > 1 {
            let t2 = 2.0 * x * t1 - t0;
            t0 = t1;
            t1 = t2;
        }
        acc += ci * t1;
    }
    acc
}

#[test]
fn long_division_identity_and_tables() {
    // f = q g + r as functions (Chebyshev basis, c0 halved convention for every series), random shapes
    let mut pr = pc_core::keys::Prng::from_seed([3; 8], 0);
    // divisor leading coefficients are powers of two, as in every use (T_k: 1, PS quotients: 2)
    for (nf, ng, lead) in [(14usize, 10usize, 1.0), (20, 7, 2.0), (31, 15, 0.5), (9, 9, 2.0), (12, 3, 4.0)] {
        let f: Vec<f64> = (0..=nf).map(|_| 2.0 * pr.uniform_f64() - 1.0).collect();
        let mut g: Vec<f64> = vec![0.0; ng + 1];
        g[ng] = lead;
        g[0] = 0.3;
        let ld = long_division(&f, &g).unwrap();
        for x in [-0.9, -0.3, 0.1, 0.77] {
            let lhs = cheb_eval(&f, x, true);
            let rhs = cheb_eval(&ld.q, x, true) * cheb_eval(&g, x, true) + cheb_eval(&ld.r, x, true);
            assert!((lhs - rhs).abs() < 1e-9, "nf={nf} ng={ng} x={x}: {lhs} vs {rhs}");
        }
    }
    assert!(long_division(&[1.0, 0.0], &[1.0, 1.0]).is_err());
    assert_eq!(compute_degrees_ps(14).unwrap(), (5, 2));
    assert_eq!(compute_degrees_ps(20).unwrap(), (3, 3));
    assert_eq!(compute_degrees_ps(119).unwrap(), (8, 4));
    assert!(compute_degrees_ps(3000).is_ok());
    assert_eq!(depth_by_degree(14).unwrap(), 6);
    assert_eq!(depth_by_degree(20).unwrap(), 6);
    assert!(depth_by_degree(4).is_err());
}
