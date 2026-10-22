//! Scale/CRT tables: the n32 tables equal every table in the d3 params.json, ElemForEvalMult reproduces the reference
//! run's memo, and the HYBRID tables the dump does not carry satisfy their CRT identities on both presets.
use pc_core::modmul::mulmod;
use pc_core::params::{check, Chain, ChainSpec};
use pc_core::scale::ScaleTables;

const D3: &str = "/data02/users/mithrillo/projects/perseus-oracle/fx/dumps/d3";

fn load(name: &str) -> Option<serde_json::Value> {
    let path = format!("{D3}/{name}");
    match std::fs::read_to_string(&path) {
        Ok(s) => Some(serde_json::from_str(&s).expect("dump json")),
        Err(_) => {
            println!("SKIP: {path} not readable");
            None
        }
    }
}

#[test]
fn n32_tables_equal_d3() {
    let c = Chain::generate(&ChainSpec::n32()).unwrap();
    let st = ScaleTables::precompute(&c).unwrap();
    let Some(j) = load("params.json") else { return };
    let r = check::compare(&j, &c, &st);
    for (f, n, bad, first) in &r.fields {
        println!("{f:24} {n:6} {bad:3} {}", first.as_deref().unwrap_or(""));
    }
    assert!(r.unknown.is_empty(), "unclassified dump fields: {:?}", r.unknown);
    assert!(r.missing.is_empty(), "fields missing from the dump: {:?}", r.missing);
    assert_eq!(r.differences(), 0);
    // the comparison is bit-exact: one ulp on one scaling factor and one table entry are both caught
    let mut st2 = st.clone();
    st2.sfr[c.l() - 2] = f64::from_bits(st2.sfr[c.l() - 2].to_bits() + 1);
    st2.q_inv[3][5] ^= 1;
    assert_eq!(check::compare(&j, &c, &st2).differences(), 3); // ScalingFactorReal, sf_at_limb, q_inv
}

#[test]
fn n32_elem_memo() {
    let c = Chain::generate(&ChainSpec::n32()).unwrap();
    let st = ScaleTables::precompute(&c).unwrap();
    let Some(memo) = load("elem_memo.json") else { return };
    let mut n = 0;
    for e in memo.as_array().unwrap() {
        let level = e["level"].as_i64().unwrap() as i32;
        let level_in = e["level_in"].as_i64().unwrap() as i32;
        let op = f64::from_bits(e["operand_bits"].as_u64().unwrap());
        let want: Vec<u64> = e["res"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect();
        let got = st.elem_for_eval_mult(level, op, level_in).unwrap();
        assert_eq!(got, want, "level {level} level_in {level_in} operand {op:e}");
        n += 1;
    }
    println!("[memo] {n} entries, 0 mismatching");
    assert!(n > 0);
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn hybrid_identities(c: &Chain) {
    let st = ScaleTables::precompute(c).unwrap();
    let (q, p) = (&c.q, &c.p);
    let all = c.primes();
    for i in 0..q.len() {
        assert_eq!(mulmod(st.p_mod_q[i], st.p_inv_mod_q[i], q[i]), 1);
        for j in 0..i {
            assert_eq!(mulmod(st.ql_inv_mod_q[i][j], q[i] % q[j], q[j]), 1);
            assert_eq!((st.qlql[i][j] + st.ql_inv_mod_q[i][j]) % q[j], 0);
        }
    }
    let mut rng = Rng(0x1234_5678_9abc_def1);
    // ModDown basis conversion P -> Q: sum_j [x_j PHatInv_j]_pj PHat_j == x + u P (mod q_i), 0 <= u < K.
    for _ in 0..64 {
        let x = rng.next() >> 4;
        let y: Vec<u64> = p.iter().enumerate().map(|(j, &pj)| mulmod(x % pj, st.p_hat_inv_mod_p[j], pj)).collect();
        for i in 0..q.len() {
            let s = (0..p.len()).fold(0u64, |a, j| (a + mulmod(y[j], st.p_hat_mod_q[j][i], q[i])) % q[i]);
            let diff = (s + q[i] - x % q[i]) % q[i];
            assert!((0..p.len() as u64).any(|u| mulmod(u, st.p_mod_q[i], q[i]) == diff), "ModDown CRT at q{i}");
        }
    }
    // ModUp of digit k at limb top lt: same identity over the complement basis.
    for lt in (0..q.len()).step_by(5) {
        for (k, ids) in st.compl_part_ids[lt].iter().enumerate() {
            let live: Vec<usize> = c.digits[k].iter().copied().filter(|&i| i <= lt).collect();
            let t = live.len() - 1;
            assert_eq!(ids.len(), lt + 1 - live.len() + p.len());
            let x = rng.next() % q[live[0]];
            let y: Vec<u64> =
                live.iter().enumerate().map(|(i, &id)| mulmod(x % q[id], st.part_ql_hat_inv_mod_q[k][t][i], q[id])).collect();
            for (jj, &tid) in ids.iter().enumerate() {
                let m = all[tid];
                let s = (0..live.len()).fold(0u64, |a, i| (a + mulmod(y[i], st.part_ql_hat_mod_p[lt][k][i][jj], m)) % m);
                let qk = live.iter().fold(1u64, |a, &id| mulmod(a, q[id] % m, m));
                let diff = (s + m - x % m) % m;
                assert!((0..live.len() as u64).any(|u| mulmod(u, qk, m) == diff), "ModUp CRT lt={lt} k={k}");
            }
        }
    }
    for l in 0..q.len() {
        for i in 0..=l {
            let hat = (0..=l).filter(|&k| k != i).fold(1u64, |a, k| mulmod(a, q[k] % q[i], q[i]));
            assert_eq!(mulmod(hat, st.ql_hat_inv_mod_q[l][i], q[i]), 1);
        }
    }
    // scale bookkeeping: on-grid factors are near the scaling-bit size, the top one is the top level's modulus
    let top: f64 = (0..c.level_unit).map(|j| q[q.len() - 1 - j] as f64).product();
    assert_eq!(st.sfr[c.l()], top);
    for lt in (0..=c.l()).filter(|&l| c.on_grid(l) && l >= c.level_unit) {
        let lg = st.sf(lt as i32).unwrap().log2();
        assert!((lg - c.spec.scaling_bits as f64).abs() < 1.5, "sf at {lt} = 2^{lg}");
    }
}

#[test]
fn hybrid_tables_n32() {
    hybrid_identities(&Chain::generate(&ChainSpec::n32()).unwrap());
}

#[test]
fn hybrid_tables_n64() {
    hybrid_identities(&Chain::generate(&ChainSpec::n64()).unwrap());
}
