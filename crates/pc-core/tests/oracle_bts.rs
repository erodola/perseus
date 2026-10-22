//! bts_setup gate: the CtS factor values (OpenFHE collapsed FFT, FIDESlib layout) vs the decoded d3 stage
//! plaintexts (`cts<f>.pt<j>.q.bin`), the stage metadata (rotIn / rotOut / steps) vs params.json `boot`, and the
//! rotation sets.
mod oracle_common;
use oracle_common::*;
use pc_core::bts_setup::*;
use rayon::prelude::*;
use serde_json::Value;

const N: usize = 1 << 16;
const SLOTS: u32 = 1 << 15;

/// scaleEnc = pre / k of `EvalBootstrapSetup`: pre = 1 (compositeDegree > 1) and k = 1 because the d3 context's
/// OpenFHE SecretKeyDist is UNIFORM_TERNARY (patch 0014: uniform main key; the sparse encapsulation lives in
/// FIDESlib), whose bootK = 16 rides the exact post-raise constant cts0_const = pre / (K N) instead.
const SCALE_ENC: f64 = 1.0;

fn factors() -> BtsFactors {
    // scale_dec is not gated here (the StC plaintexts are not in the dump): 1.0
    cts_stc_factors(N, SLOTS, [4, 3], [0, 0], SCALE_ENC, 1.0).unwrap()
}

fn ptidx(idx: usize, mask: u32) -> usize {
    if mask >> 24 == 0x7F {
        idx >> (mask & 0xFF)
    } else {
        idx & mask as usize
    }
}

/// Decode one dumped plaintext: compact eval rows of the bottom 3 primes -> INTT -> exact CRT -> / noise factor ->
/// canonical embedding (N/2 slots).
fn decode_pt(p: &Value, tabs: &[NttTables], stage: usize, j: usize) -> Vec<C64> {
    let meta = &p["boot"]["CtS"][stage]["pts"][j];
    let ids = u64v(&meta["c0"]["ids"]);
    let sizes = u64v(&meta["c0"]["q_sizes"]);
    let nf = meta["noise_factor"].as_f64().unwrap();
    let mask = meta["compact_mask"].as_u64().unwrap() as u32;
    let path = d3().join(format!("cts{stage}.pt{j}.q.bin"));
    let nl = 3;
    let mut rows = Vec::new();
    let mut off = 0usize;
    for l in 0..nl {
        assert_eq!(ids[l] as usize, l);
        let st = read_u32_at(&path, off, sizes[l] as usize);
        off += sizes[l] as usize;
        let mut full: Vec<u32> = (0..N).map(|i| st[ptidx(i, mask)]).collect();
        tabs[l].inv(&mut full);
        rows.push(full);
    }
    let primes: Vec<u32> = tabs[..nl].iter().map(|t| t.p as u32).collect();
    let m: Vec<f64> = (0..N).map(|i| crt_centered(&[rows[0][i], rows[1][i], rows[2][i]], &primes) / nf).collect();
    decode_slots(&m)
}

#[test]
fn stage_metadata_matches_the_dump() {
    let p = params();
    let f = factors();
    assert_eq!(
        (f.enc.layers_collapse, f.enc.rem_collapse, f.enc.num_rotations, f.enc.b, f.enc.g, f.enc.num_rotations_rem, f.enc.b_rem, f.enc.g_rem),
        (4, 3, 31, 2, 16, 15, 2, 8)
    );
    assert_eq!((f.dec.layers_collapse, f.dec.rem_collapse, f.dec.num_rotations, f.dec.b, f.dec.g), (5, 0, 63, 4, 16));
    let lay = f.fides_layout();
    for (kind, stages) in [("CtS", &lay.cts), ("StC", &lay.stc)] {
        let ds = p["boot"][kind].as_array().unwrap();
        assert_eq!(ds.len(), stages.len());
        for (i, (st, d)) in stages.iter().zip(ds).enumerate() {
            assert_eq!(st.num_diags as u64, d["slots"].as_u64().unwrap(), "{kind}{i} diagonals");
            assert_eq!(st.b_step as u64, d["bStep"].as_u64().unwrap(), "{kind}{i} bStep");
            assert_eq!(st.g_step as u64, d["gStep"].as_u64().unwrap(), "{kind}{i} gStep");
            assert_eq!(st.rot_in, i64v(&d["rotIn"]), "{kind}{i} rotIn");
            assert_eq!(st.rot_out, i64v(&d["rotOut"]), "{kind}{i} rotOut");
            if kind == "CtS" {
                assert_eq!(st.num_diags, d["pts"].as_array().unwrap().len());
            }
        }
    }
    // the dump's CtS rotation list = the nonzero rotIn / rotOut of the CtS stages in the FIDESlib layout
    let mut mine: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
    for st in &lay.cts {
        mine.extend(st.rot_in.iter().chain(st.rot_out.iter()).copied().filter(|&x| x != 0));
    }
    let dumped: std::collections::BTreeSet<i64> = i64v(&p["cts_rotations"]).into_iter().collect();
    assert_eq!(mine, dumped, "cts_rotations");
    eprintln!("[oracle bts] stage metadata: 4 CtS + 3 StC stages (steps, rotIn, rotOut) == d3 boot; acc offset {}", lay.acc_offset);
}

#[test]
fn rotation_sets() {
    let f = factors();
    let all = f.bootstrap_rotation_indices();
    assert!(all.iter().all(|&r| r > 0 && r < SLOTS));
    // OpenFHE's own centered BSGS set: CtS stage 3 (stride 2048) needs -15..0 * 2048 = 2048 * {1..16} mod 2^15
    for k in 1..16 {
        assert!(all.contains(&(k * 2048)), "missing {}", k * 2048);
    }
    eprintln!("[oracle bts] OpenFHE FindBootstrapRotationIndices: {} rotations", all.len());
}

#[test]
fn cts_factor_values_match_decoded_d3_plaintexts() {
    let p = params();
    let q = u64v(&p["q"]);
    let roots = u64v(&p["const_root"]);
    let tabs: Vec<NttTables> = (0..3).map(|i| NttTables::new(q[i], roots[i], N)).collect();
    let f = factors();
    let lay = f.fides_layout();
    let mut worst = 0.0f64;
    let mut count = 0;
    for (si, st) in lay.cts.iter().enumerate() {
        let res: Vec<(usize, f64, f64)> = (0..st.num_diags)
            .into_par_iter()
            .map(|j| {
                let got = decode_pt(&p, &tabs, si, j);
                let want = &st.pts[j];
                let mx = want.iter().map(|x| x.abs()).fold(0.0, f64::max);
                let d = got.iter().zip(want).map(|(a, b)| (*a - *b).abs()).fold(0.0, f64::max);
                (j, d / mx, mx)
            })
            .collect();
        let sw = res.iter().map(|r| r.1).fold(0.0, f64::max);
        let nf = p["boot"]["CtS"][si]["pts"][0]["noise_factor"].as_f64().unwrap();
        eprintln!(
            "[oracle bts] CtS stage {si} (OpenFHE s={}, stride {}, pt rotation {}, nf {nf:e}): {} diagonals, max rel diff {sw:.3e}",
            f.cts.len() - 1 - si,
            f.cts[f.cts.len() - 1 - si].stride,
            st.pt_rotation,
            res.len()
        );
        for r in &res {
            assert!(r.1 <= 1e-9, "CtS stage {si} diagonal {}: rel diff {:e} (max |want| {})", r.0, r.1, r.2);
        }
        worst = worst.max(sw);
        count += res.len();
    }
    eprintln!("[oracle bts] CtS gate: {count} diagonals in 4 stages, worst rel diff {worst:.3e} (<= 1e-9)");
}

/// The stages compose to the DFT (small ring, every collapse shape): StC applied s = 0.. is x -> U0 brev(x) and CtS
/// applied s = budget-1 .. 0 is x -> brev(U0^H x), unnormalized (OpenFHE's U0[i][j] = ksi^(j 5^i); the collapsed FFT works
/// on bit-reversed coefficients).
#[test]
fn stages_compose_to_the_dft() {
    let n = 1 << 8;
    let slots = (n / 2) as u32;
    let m = 4 * slots;
    let rg = rot_group(slots, m);
    let pows = ksi_pows(m);
    let lb = slots.trailing_zeros();
    let br = |i: usize| i.reverse_bits() >> (usize::BITS - lb);
    let mut pr = pc_core::keys::Prng::from_seed([5; 8], 0);
    let x: Vec<C64> = (0..slots).map(|_| C64::new(pr.uniform_f64() - 0.5, pr.uniform_f64() - 0.5)).collect();
    let xb: Vec<C64> = (0..slots as usize).map(|i| x[br(i)]).collect();
    let u0 = |v: &[C64]| -> Vec<C64> {
        (0..slots as usize)
            .map(|i| {
                let mut a = C64::ZERO;
                for (j, vj) in v.iter().enumerate() {
                    a += pows[((j as u32).wrapping_mul(rg[i]) & (m - 1)) as usize] * *vj;
                }
                a
            })
            .collect()
    };
    let u0h = |v: &[C64]| -> Vec<C64> {
        (0..slots as usize)
            .map(|j| {
                let mut a = C64::ZERO;
                for (i, vi) in v.iter().enumerate() {
                    a += pows[((j as u32).wrapping_mul(rg[i]) & (m - 1)) as usize].conj() * *vi;
                }
                a
            })
            .collect()
    };
    let want_stc = u0(&xb);
    let h = u0h(&x);
    let want_cts: Vec<C64> = (0..slots as usize).map(|i| h[br(i)]).collect();
    let maxd = |a: &[C64], b: &[C64]| a.iter().zip(b).map(|(x, y)| (*x - *y).abs()).fold(0.0, f64::max);
    for budget in [[2u32, 2u32], [3, 3], [4, 2], [7, 7], [1, 3], [3, 1]] {
        let f = cts_stc_factors(n, slots, budget, [0, 0], 1.0, 1.0).unwrap();
        let mut y = x.clone();
        for st in f.stc.iter() {
            y = st.apply(&y);
        }
        let d = maxd(&y, &want_stc);
        assert!(d < 1e-9, "budget {budget:?}: StC != U0 brev ({d:e})");
        let mut z = x.clone();
        for st in f.cts.iter().rev() {
            z = st.apply(&z);
        }
        let d = maxd(&z, &want_cts);
        assert!(d < 1e-9, "budget {budget:?}: CtS != brev U0^H ({d:e})");
    }
}

fn rotl(v: &[C64], r: i64) -> Vec<C64> {
    rotate_left(v, r)
}

/// OpenFHE BSGS evaluation of a stage (pre-rotated diagonals, rot_in / rot_out) == the stage's reference semantics,
/// and the FIDESlib layout (accumulated plaintext pre-rotation, rotIn from 0, offset paid by the last StC rotOut)
/// computes the same CtS -> StC chain as OpenFHE's.
#[test]
fn bsgs_and_fides_layout_semantics() {
    let n = 1 << 9;
    let slots = (n / 2) as u32;
    let m4 = slots;
    let mut pr = pc_core::keys::Prng::from_seed([11; 8], 0);
    let x: Vec<C64> = (0..slots).map(|_| C64::new(pr.uniform_f64() - 0.5, pr.uniform_f64() - 0.5)).collect();
    let maxd = |a: &[C64], b: &[C64]| a.iter().zip(b).map(|(x, y)| (*x - *y).abs()).fold(0.0, f64::max);
    for budget in [[4u32, 3u32], [3, 3], [2, 4]] {
        let f = cts_stc_factors(n, slots, budget, [0, 0], 0.5, 2.0).unwrap();
        let mut ref_chain = x.clone();
        for st in f.cts.iter().rev().chain(f.stc.iter()) {
            let rin = st.rot_in(m4);
            let rout = st.rot_out(m4);
            let mut y = vec![C64::ZERO; x.len()];
            for i in 0..st.b as usize {
                let mut inner = vec![C64::ZERO; x.len()];
                for j in 0..st.g as usize {
                    let idx = st.g as usize * i + j;
                    if idx >= st.num_diags() {
                        continue;
                    }
                    let a = st.bsgs_diag(idx);
                    let xr = rotl(&ref_chain, rin[j] as i64);
                    for k in 0..x.len() {
                        inner[k] += a[k] * xr[k];
                    }
                }
                let o = rotl(&inner, rout[i] as i64);
                for k in 0..x.len() {
                    y[k] += o[k];
                }
            }
            let want = st.apply(&ref_chain);
            assert!(maxd(&y, &want) < 1e-9, "budget {budget:?} stage s={}: BSGS != reference", st.s);
            ref_chain = want;
        }
        let lay = f.fides_layout();
        let mut fx = x.clone();
        for st in lay.cts.iter().chain(lay.stc.iter()) {
            let mut y = vec![C64::ZERO; x.len()];
            for i in 0..st.g_step as usize {
                let mut inner = vec![C64::ZERO; x.len()];
                for j in 0..st.b_step as usize {
                    let idx = st.b_step as usize * i + j;
                    if idx >= st.num_diags {
                        continue;
                    }
                    let xr = rotl(&fx, st.rot_in[j]);
                    for k in 0..x.len() {
                        inner[k] += st.pts[idx][k] * xr[k];
                    }
                }
                let o = rotl(&inner, st.rot_out[i]);
                for k in 0..x.len() {
                    y[k] += o[k];
                }
            }
            fx = y;
        }
        let d = maxd(&fx, &ref_chain);
        assert!(d < 1e-9, "budget {budget:?}: FIDESlib layout chain != OpenFHE chain ({d:e})");
    }
}

/// SparseB geometry of the shipped sparse routes, the block structure of its diagonals, sparse-slot CtS/StC shapes,
/// and the StC-first stage factors. (No oracle dump holds SparseB / sparse-route plaintexts: values are gated later by
/// the bootstrap routes' bit-exactness checks.)
#[test]
fn sparse_routes_and_stc_first() {
    let n = N as u32;
    assert_eq!(sparse_b_geometry(n, 512, 0), Some(SparseBGeom { n: 1024, r: 64, s: 32, b_step: 16, g_step: 2 }));
    assert_eq!(sparse_b_geometry(n, 256, 0), Some(SparseBGeom { n: 512, r: 128, s: 8, b_step: 8, g_step: 1 }));
    assert_eq!(sparse_b_geometry(n, 1, 0), None);
    assert_eq!(sparse_b_geometry(n, SLOTS, 0), None);
    assert_eq!(sparse_b_geometry(n, 512, 32).map(|g| (g.b_step, g.g_step)), Some((32, 1)));
    // small ring: block structure and the BSGS pre-rotation
    let (rn, sl) = (1u32 << 10, 64u32);
    let g = sparse_b_geometry(rn, sl, 0).unwrap();
    let (pd, qd) = sparse_b_diagonals(rn, sl, &g, 0.75, 1.5);
    assert_eq!((pd.len(), qd.len()), (g.s as usize, g.s as usize));
    let (nn, half) = (g.n as usize, g.n as usize / 2);
    for k in 0..g.s as usize {
        let shift = (k / g.b_step as usize) * g.b_step as usize;
        let p = rotate_left(&pd[k], shift as i64);
        let q = rotate_left(&qd[k], shift as i64);
        for a in 0..g.r as usize / 2 {
            for j in 0..half {
                assert_eq!(p[a * nn + j], p[a * nn + half + j]);
                assert_eq!(q[a * nn + half + j], C64::ZERO);
                assert!((p[a * nn + j].abs() - 0.75).abs() < 1e-12 && (q[a * nn + j].abs() - 1.5).abs() < 1e-12);
            }
        }
    }
    // sparse-slot multi-stage factors: diagonal vectors are [coeff | coeff_i] (2 slots long)
    let f = cts_stc_factors(N, 512, [3, 3], [0, 0], 1.0, 1.0).unwrap();
    assert!(f.cts.iter().chain(f.stc.iter()).all(|s| s.len() == 1024));
    assert!(f.bootstrap_rotation_indices().contains(&512));
    // StC-first
    assert_eq!(stc_first_stage_factors(3, 4, 1).unwrap(), vec![Some(1.0), Some(1.0), Some(1.0 / 16.0)]);
    assert_eq!(stc_first_stage_factors(3, 4, 2).unwrap(), vec![Some(1.0), Some(1.0), None]);
    assert!(stc_first_stage_factors(0, 4, 1).is_err());
    assert_eq!(stc_first_mode2_factor(6, 4.0, 2.0, 3.0, 1.5), 4.0 / 64.0);
}
