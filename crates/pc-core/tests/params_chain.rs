//! Prime search on both presets: the n32 chain equals the d3 dump's, the n64 chain is a valid chain.
use pc_core::params::{max_log_qp, nbtheory, Chain, ChainSpec};

const D3: &str = "/data02/users/mithrillo/projects/perseus-oracle/fx/dumps/d3/params.json";

fn check_valid(c: &Chain) {
    let m = 2 * c.n as u64;
    let all = c.primes();
    let mut s = all.clone();
    s.sort();
    s.dedup();
    assert_eq!(s.len(), all.len(), "duplicate primes");
    for (&p, &r) in all.iter().zip(c.roots().iter()) {
        assert!(nbtheory::is_prime(p));
        assert_eq!(p % m, 1, "prime {p} not NTT-friendly");
        assert_eq!(pc_core::modmul::powmod(r, c.n as u64, p), p - 1, "root of {p} not primitive 2N-th");
    }
    assert!(c.log_qp <= max_log_qp(c.spec.security, c.n as u32).unwrap());
    assert_eq!(c.q.len(), (c.spec.num_levels * c.spec.composite_degree) as usize);
    assert_eq!(c.digits.iter().map(|d| d.len()).sum::<usize>(), c.q.len());
}

#[test]
fn n32_chain() {
    let c = Chain::generate(&ChainSpec::n32()).unwrap();
    check_valid(&c);
    assert_eq!(c.q.len(), 54);
    assert_eq!(c.p.len(), 9);
    assert!(c.q.iter().chain(&c.p).all(|&p| p < 1 << 28));
    println!("n32: logQP = {}", c.log_qp);
    match std::fs::read_to_string(D3) {
        Ok(s) => {
            let j: serde_json::Value = serde_json::from_str(&s).unwrap();
            let v = |k: &str| j[k].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect::<Vec<u64>>();
            assert_eq!(c.q, v("q"));
            assert_eq!(c.p, v("p"));
            assert_eq!(c.roots(), v("const_root"));
            assert_eq!(c.inv_roots(), v("const_inv_root"));
        }
        Err(_) => println!("SKIP d3 comparison: {D3} not readable"),
    }
}

#[test]
fn n64_chain() {
    let c = Chain::generate(&ChainSpec::n64()).unwrap();
    check_valid(&c);
    assert_eq!(nbtheory::bit_length(c.q[0]), 60);
    for &q in &c.q[1..] {
        assert!((q as f64).log2() > 52.0 && (q as f64).log2() < 54.0);
    }
    assert!(c.p.iter().all(|&p| nbtheory::bit_length(p) == 60));
    println!("n64: {} Q + {} P primes, logQP = {}", c.q.len(), c.p.len(), c.log_qp);
}
