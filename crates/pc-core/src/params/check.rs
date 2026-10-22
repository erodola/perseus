//! Field-by-field comparison of a generated chain + scale tables against a `params.json` oracle dump (written
//! by the oracle's `fxdump.cu::ctx_params` from FIDESlib's ContextData). Integers compare by value, doubles bit for
//! bit. Shared by `tests/scale_tables.rs` and the `pc-params --check` gate.
use super::Chain;
use crate::scale::ScaleTables;
use serde_json::{json, Value};

/// Dump fields that are not chain/table data, with what they are (they are reported, not compared).
pub const OUT_OF_SCOPE: &[(&str, &str)] = &[
    ("batch", "runtime slot batch of the dumping context, not a chain property"),
    ("boot_k", "pc-core bootstrap constants"),
    ("cheb", "pc-core EvalMod Chebyshev coefficients"),
    ("double_angle_its", "pc-core EvalMod"),
    ("slots", "pc-core bootstrap route"),
    ("cheb_ps", "pc-core Paterson-Stockmeyer plan"),
    ("drop", "pc-core raise drop of the dumped run"),
    ("boot", "pc-core bootstrap precomputation"),
    ("state", "dumped run state"),
    ("state_fields", "dumped run state"),
    ("state_after_raise", "dumped run state"),
    ("stash_raise", "dumped run stash names"),
    ("stash_em", "dumped run stash names"),
    ("errs", "dumped run errors"),
    ("key_id", "pc-core key material"),
    ("cts_rotations", "pc-core rotation sets"),
    ("gpu_rot_keys", "pc-core rotation sets"),
    ("cpu_auto_keys", "pc-core rotation sets"),
    ("keys", "pc-core key material"),
];

#[derive(Debug, Default)]
pub struct Report {
    /// (field, entries compared, entries differing, first difference)
    pub fields: Vec<(String, usize, usize, Option<String>)>,
    /// Dump fields neither compared nor listed as out of scope.
    pub unknown: Vec<String>,
    /// Compared fields absent from the dump.
    pub missing: Vec<String>,
    pub out_of_scope: Vec<(String, String)>,
}

impl Report {
    pub fn differences(&self) -> usize {
        self.fields.iter().map(|f| f.2).sum::<usize>() + self.unknown.len() + self.missing.len()
    }
}

/// Flatten a JSON value to its leaves (numbers as JSON values), in order.
fn leaves(v: &Value, out: &mut Vec<Value>) {
    match v {
        Value::Array(a) => a.iter().for_each(|x| leaves(x, out)),
        other => out.push(other.clone()),
    }
}

fn same(a: &Value, b: &Value) -> bool {
    match (a.as_f64(), b.as_f64()) {
        (Some(x), Some(y)) if a.is_f64() || b.is_f64() => x.to_bits() == y.to_bits(),
        _ => a == b,
    }
}

/// What this crate computes for every compared field, in the dump's layout.
pub fn expected_fields(chain: &Chain, st: &ScaleTables) -> Vec<(&'static str, Value)> {
    let nq = chain.q.len();
    let np = nq + chain.p.len();
    let l = chain.l();
    let d = chain.level_unit;
    let pad = |v: &[u64]| -> Vec<u64> { (0..np).map(|i| v.get(i).copied().unwrap_or(0)).collect() };
    let qlql: Vec<Vec<u64>> = (0..np)
        .map(|j| (0..np).map(|i| if j < nq && i < j { st.qlql[j][i] } else { 0 }).collect())
        .collect();
    let mut mdpre = vec![0u64; np];
    for (k, &v) in st.p_hat_inv_mod_p.iter().enumerate() {
        mdpre[nq + k] = v;
    }
    let sf_at: Vec<f64> = (0..=l).map(|i| if (l - i) % d == 0 { st.sfr[i] } else { -1.0 }).collect();
    let mrf_at: Vec<f64> = (0..=l).map(|i| if i > 0 { st.mrf[i] } else { -1.0 }).collect();
    vec![
        ("N", json!(chain.n)),
        ("logN", json!(chain.spec.logn)),
        ("L", json!(l)),
        ("K", json!(chain.p.len())),
        ("dnum", json!(chain.dnum())),
        ("composite_degree", json!(d)),
        // FIDESlib RESCALE_TECHNIQUE: COMPOSITESCALING* and FLEXIBLEAUTO both map to FLEXIBLEAUTO (= 3)
        ("rescale_technique", json!(3)),
        ("q", json!(chain.q)),
        ("p", json!(chain.p)),
        ("const_primes", json!(chain.primes())),
        ("const_root", json!(chain.roots())),
        ("const_inv_root", json!(chain.inv_roots())),
        ("const_bits", json!(chain.prime_bits())),
        ("const_type", json!(chain.limb_type_mask())),
        ("P_inv", json!(pad(&st.p_inv_mod_q))),
        ("P_mod", json!(pad(&st.p_mod_q))),
        ("q_inv", json!(st.q_inv)),
        ("QlQlInvModqlDivqlModq", json!(qlql)),
        ("ModDown_pre_scale", json!(mdpre)),
        ("digits", json!(chain.digits)),
        ("digit_to", json!((0..chain.dnum()).map(|k| chain.digit_complement(k)).collect::<Vec<_>>())),
        ("special_ids", json!(chain.special_ids())),
        ("sf_at_limb", json!(sf_at)),
        ("mod_reduce_factor_at", json!(mrf_at)),
        ("ScalingFactorReal", json!(st.sfr)),
        ("ScalingFactorRealBig", json!(st.sfr_big)),
        ("ModReduceFactor", json!(st.mrf)),
    ]
}

pub fn compare(dump: &Value, chain: &Chain, st: &ScaleTables) -> Report {
    let mut r = Report::default();
    let exp = expected_fields(chain, st);
    for (name, want) in &exp {
        let Some(got) = dump.get(*name) else {
            r.missing.push(name.to_string());
            continue;
        };
        let (mut a, mut b) = (Vec::new(), Vec::new());
        leaves(want, &mut a);
        leaves(got, &mut b);
        let mut bad = 0;
        let mut first = None;
        if a.len() != b.len() {
            bad = a.len().max(b.len());
            first = Some(format!("length mine {} dump {}", a.len(), b.len()));
        } else {
            for (i, (x, y)) in a.iter().zip(&b).enumerate() {
                if !same(x, y) {
                    bad += 1;
                    if first.is_none() {
                        first = Some(format!("leaf {i}: mine {x} dump {y}"));
                    }
                }
            }
        }
        r.fields.push((name.to_string(), a.len(), bad, first));
    }
    if let Some(obj) = dump.as_object() {
        for k in obj.keys() {
            if exp.iter().any(|(n, _)| n == k) {
                continue;
            }
            match OUT_OF_SCOPE.iter().find(|(n, _)| n == k) {
                Some((_, why)) => r.out_of_scope.push((k.clone(), why.to_string())),
                None => r.unknown.push(k.clone()),
            }
        }
    }
    r
}
