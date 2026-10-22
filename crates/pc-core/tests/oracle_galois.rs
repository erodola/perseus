//! galois: rotate-by-r of an encoded vector == slot shift, via the eval-form permutation (host NTT), the
//! coefficient-form map, and FIDESlib's scatter form; the d3 key names follow the same convention.
mod oracle_common;
use oracle_common::*;
use pc_core::bts_setup::C64;
use pc_core::galois::*;
use pc_core::keys::Prng;

const N: usize = 1 << 12;
const SCALE: f64 = (1u64 << 24) as f64;

struct Fixture {
    q: u64,
    tab: NttTables,
    z: Vec<C64>,
    coef: Vec<u32>,
    eval: Vec<u32>,
}

fn fixture() -> Fixture {
    let (q, psi) = ntt_prime(N);
    let tab = NttTables::new(q, psi, N);
    let mut pr = Prng::from_seed([7; 8], 0);
    let z: Vec<C64> = (0..N / 2).map(|_| C64::new(2.0 * pr.uniform_f64() - 1.0, 2.0 * pr.uniform_f64() - 1.0)).collect();
    let m = encode_slots(&z, N);
    let coef: Vec<u32> = m.iter().map(|&x| ((x * SCALE).round() as i64).rem_euclid(q as i64) as u32).collect();
    let mut eval = coef.clone();
    tab.fwd(&mut eval);
    Fixture { q, tab, z, coef, eval }
}

fn decode(f: &Fixture, eval: &[u32]) -> Vec<C64> {
    let mut c = eval.to_vec();
    f.tab.inv(&mut c);
    let m: Vec<f64> = c.iter().map(|&x| centered(x, f.q as u32) as f64 / SCALE).collect();
    decode_slots(&m)
}

fn maxdiff(a: &[C64], b: &[C64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (*x - *y).abs()).fold(0.0, f64::max)
}

#[test]
fn rotation_is_a_left_slot_shift() {
    let f = fixture();
    assert!(maxdiff(&decode(&f, &f.eval), &f.z) < 1e-5, "encode/decode round trip");
    let h = (N / 2) as i64;
    for r in [1i64, 2, 3, 5, 16, 100, 1000, h - 1, -1, -7, -(h - 1), h + 3] {
        let g = rotation_galois(r, N).unwrap();
        let rot = apply_eval(&f.eval, g).unwrap();
        let want: Vec<C64> = (0..N / 2).map(|j| f.z[((j as i64 + r).rem_euclid(h)) as usize]).collect();
        let d = maxdiff(&decode(&f, &rot), &want);
        assert!(d < 1e-5, "r={r}: max diff {d}");
        // coefficient-form map == eval-form gather
        let mut c = apply_coeff_mod(&f.coef, g, f.q as u32).unwrap();
        f.tab.fwd(&mut c);
        assert_eq!(c, rot, "r={r}: coefficient and eval automorphisms differ");
        // FIDESlib: scatter by k = 5^(2N - r) == gather by g
        let k = fides_kernel_exponent(r, N).unwrap();
        assert_eq!(k, galois_inverse(g, N).unwrap());
        assert_eq!(k * g % (2 * N as u64), 1);
        let mut sc = vec![0u32; N];
        for (i, &v) in f.eval.iter().enumerate() {
            sc[fides_scatter_index(k, i as u32, N.trailing_zeros()) as usize] = v;
        }
        assert_eq!(sc, rot, "r={r}: FIDESlib scatter form differs");
        assert_eq!(galois_to_rotation(g, N).unwrap(), Some(r.rem_euclid(h) as u64));
    }
}

#[test]
fn conjugation_conjugates_slots() {
    let f = fixture();
    let g = conj_galois(N).unwrap();
    assert_eq!(g, 2 * N as u64 - 1);
    assert_eq!(galois_to_rotation(g, N).unwrap(), None);
    let c = apply_eval(&f.eval, g).unwrap();
    let want: Vec<C64> = f.z.iter().map(|x| x.conj()).collect();
    assert!(maxdiff(&decode(&f, &c), &want) < 1e-5);
    let signed: Vec<i64> = f.coef.iter().map(|&x| centered(x, f.q as u32)).collect();
    let cs = apply_coeff_signed(&signed, g).unwrap();
    let mut cm: Vec<u32> = cs.iter().map(|&x| x.rem_euclid(f.q as i64) as u32).collect();
    f.tab.fwd(&mut cm);
    assert_eq!(cm, c);
}

#[test]
fn d3_key_names_follow_the_convention() {
    let p = params();
    let n = p["N"].as_u64().unwrap() as usize;
    let mut checked = 0;
    for (name, k) in p["keys"].as_object().unwrap() {
        if name == "mult" {
            continue;
        }
        let g: u64 = name.parse().unwrap();
        match k.get("rot").and_then(|v| v.as_i64()) {
            Some(-1) => assert_eq!(g, conj_galois(n).unwrap(), "key {name} labelled conjugation"),
            Some(r) => assert_eq!(g, rotation_galois(r, n).unwrap(), "key {name} labelled rotation {r}"),
            None => assert!(galois_to_rotation(g, n).unwrap().is_some() || g == conj_galois(n).unwrap()),
        }
        checked += 1;
    }
    eprintln!("[oracle galois] {checked} d3 key indices == 5^r mod 2N (or 2N - 1)");
}
