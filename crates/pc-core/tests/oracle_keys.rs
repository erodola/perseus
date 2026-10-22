//! Keys gate: the KSKA host twin vs the dumped dense `a` of the d3 dump (every one), and the
//! key-switching relation b + a s_new - [digit] (P mod q) s_old == small error on EVERY dumped key, which checks
//! the seed expansion of all keys plus the Galois convention against the dumped secret.
mod oracle_common;
use oracle_common::*;
use pc_core::galois;
use pc_core::keys::*;
use rayon::prelude::*;

fn key_material(p: &serde_json::Value, name: &str) -> KskMaterial {
    let k = &p["keys"][name];
    let shape = u64v(&k["shape"]);
    let (dnum, nmod, n) = (shape[0] as usize, shape[1] as usize, shape[2] as usize);
    let moduli: Vec<u32> = u64v(&k["b_moduli"][0]).into_iter().map(|x| x as u32).collect();
    assert_eq!(moduli.len(), nmod);
    for d in 0..dnum {
        assert_eq!(u64v(&k["b_moduli"][d]), u64v(&k["b_moduli"][0]), "key {name}: digit {d} limb order differs");
    }
    let seed: Vec<u32> = u64v(&k["a_seed"]).into_iter().map(|x| x as u32).collect();
    assert_eq!(seed.len(), 8, "key {name} is not seeded");
    let file = if name == "mult" { "keys/mult.b.bin".to_string() } else { format!("keys/auto{name}.b.bin") };
    let b = read_u32(&d3().join(file));
    KskMaterial::new(KskSeed(seed.try_into().unwrap()), dnum, n, moduli, b).expect("key shape")
}

#[test]
fn kska_twin_matches_dumped_dense_a() {
    let p = params();
    let mut checked_keys = 0;
    let mut rows = 0usize;
    let mut words = 0usize;
    let mut mismatches = 0usize;
    for (name, k) in p["keys"].as_object().unwrap() {
        if !k["has_a"].as_bool().unwrap() {
            continue;
        }
        let km = key_material(&p, name);
        let file = if name == "mult" { "keys/mult.a.bin".to_string() } else { format!("keys/auto{name}.a.bin") };
        let a = read_u32(&d3().join(file));
        assert_eq!(a.len(), km.dnum * km.moduli.len() * km.n);
        let mm: usize = (0..km.dnum * km.moduli.len())
            .into_par_iter()
            .map(|r| {
                let (d, l) = (r / km.moduli.len(), r % km.moduli.len());
                let row = km.a_row(d, l);
                let o = r * km.n;
                row.iter().zip(&a[o..o + km.n]).filter(|(x, y)| x != y).count()
            })
            .sum();
        // the per-coefficient reference path agrees with the row path
        for &(d, l, s) in &[(0usize, 0usize, 0u32), (km.dnum - 1, km.moduli.len() - 1, km.n as u32 - 1), (3, 17, 12345)] {
            assert_eq!(kska_expand(&km.seed, d as u32, km.moduli[l], s, km.n), a[(d * km.moduli.len() + l) * km.n + s as usize]);
        }
        mismatches += mm;
        checked_keys += 1;
        rows += km.dnum * km.moduli.len();
        words += a.len();
        eprintln!("[oracle kska] key {name}: {} digits x {} limbs x {} = {} words, mismatches {mm}", km.dnum, km.moduli.len(), km.n, a.len());
    }
    eprintln!("[oracle kska] dense-a gate: {checked_keys} keys, {rows} (digit, limb) rows, {words} words, {mismatches} mismatches");
    assert!(checked_keys >= 2, "expected the dumped dense a of auto5 and mult");
    assert_eq!(mismatches, 0);
}

#[test]
fn every_dumped_key_satisfies_the_switch_relation() {
    let p = params();
    let sk = read_npy_i8(&d3().join("sk.npy"));
    let n = sk.len();
    let q = u64v(&p["q"]);
    let pp = u64v(&p["p"]);
    let roots = u64v(&p["const_root"]);
    let pmod = u64v(&p["P_mod"]);
    let digits: Vec<Vec<usize>> = p["digits"].as_array().unwrap().iter().map(|d| u64v(d).into_iter().map(|x| x as usize).collect()).collect();
    let moduli: Vec<u64> = q.iter().chain(pp.iter()).copied().collect();
    assert_eq!(roots.len(), moduli.len());
    let nq = q.len();
    // per limb: NTT tables, NTT(s), NTT(s^2)
    let tabs: Vec<NttTables> = (0..moduli.len()).into_par_iter().map(|i| NttTables::new(moduli[i], roots[i], n)).collect();
    let s_eval: Vec<Vec<u32>> = (0..moduli.len())
        .into_par_iter()
        .map(|i| {
            let m = moduli[i];
            let mut v: Vec<u32> = sk.iter().map(|&x| if x < 0 { (m as i64 + x as i64) as u32 } else { x as u32 }).collect();
            tabs[i].fwd(&mut v);
            v
        })
        .collect();

    let mut keys = 0;
    let mut rows_checked = 0usize;
    let mut max_err = 0i64;
    for name in p["keys"].as_object().unwrap().keys() {
        let km = key_material(&p, name);
        assert_eq!(km.n, n);
        for (i, &m) in km.moduli.iter().enumerate() {
            assert_eq!(m as u64, moduli[i], "key {name}: limb {i} modulus not in prime-id order");
        }
        // OpenFHE: key index g switches s -> sigma_{g^-1}(s); mult switches s^2 -> s
        let (perm, old_sq): (Option<Vec<u32>>, bool) = if name == "mult" {
            (None, true)
        } else {
            let g: u64 = name.parse().unwrap();
            let ginv = galois::galois_inverse(g, n).unwrap();
            (Some(galois::eval_gather_table(ginv, n).unwrap()), false)
        };
        let res: Vec<(usize, i64)> = (0..km.dnum * km.moduli.len())
            .into_par_iter()
            .map(|r| {
                let (d, l) = (r / km.moduli.len(), r % km.moduli.len());
                let m = moduli[l];
                let a = km.a_row(d, l);
                let b = km.b_row(d, l);
                let s = &s_eval[l];
                let in_digit = l < nq && digits[d].contains(&l);
                let mut e: Vec<u32> = (0..n)
                    .map(|i| {
                        let snew = match &perm {
                            Some(t) => s[t[i] as usize] as u64,
                            None => s[i] as u64,
                        };
                        let mut v = (b[i] as u64 + mulmod(a[i] as u64, snew, m)) % m;
                        if in_digit {
                            let sold = if old_sq { mulmod(s[i] as u64, s[i] as u64, m) } else { s[i] as u64 };
                            v = (v + m - mulmod(pmod[l] % m, sold, m)) % m;
                        }
                        v as u32
                    })
                    .collect();
                tabs[l].inv(&mut e);
                let mx = e.iter().map(|&x| centered(x, m as u32).abs()).max().unwrap_or(0);
                (r, mx)
            })
            .collect();
        let kmax = res.iter().map(|x| x.1).max().unwrap_or(0);
        assert!(kmax <= 64, "key {name}: |e| max {kmax} (switch relation or seed expansion wrong)");
        max_err = max_err.max(kmax);
        rows_checked += res.len();
        keys += 1;
    }
    eprintln!("[oracle kska] switch relation: {keys} keys, {rows_checked} (digit, limb) rows, max |e| = {max_err}");
    assert_eq!(keys, p["keys"].as_object().unwrap().len());
}

#[test]
fn samplers_are_well_formed() {
    let mut pr = Prng::from_seed([1, 2, 3, 4, 5, 6, 7, 8], 0);
    let t = sample_ternary(&mut pr, 1 << 16);
    let c: Vec<usize> = (-1..=1).map(|v| t.iter().filter(|&&x| x == v).count()).collect();
    for &x in &c {
        assert!((x as f64 / 65536.0 - 1.0 / 3.0).abs() < 0.01, "ternary counts {c:?}");
    }
    let sp = sample_sparse_ternary(&mut pr, 1 << 16, 32).unwrap();
    assert_eq!(sp.iter().filter(|&&x| x != 0).count(), 32);
    assert!(sp.iter().all(|&x| (-1..=1).contains(&x)));
    assert!(sample_sparse_ternary(&mut pr, 8, 9).is_err());
    let dg = DiscreteGaussian::new(DiscreteGaussian::SIGMA).unwrap();
    let e = dg.sample_vec(&mut pr, 1 << 18);
    let mean = e.iter().sum::<i64>() as f64 / e.len() as f64;
    let var = e.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / e.len() as f64;
    assert!(mean.abs() < 0.03 && (var.sqrt() - 3.19).abs() < 0.03, "gaussian mean {mean} sd {}", var.sqrt());
    let q = 268042241u64;
    let u = sample_uniform_mod(&mut pr, 1 << 16, q);
    assert!(u.iter().all(|&x| x < q));
    let avg = u.iter().map(|&x| x as f64).sum::<f64>() / u.len() as f64;
    assert!((avg / q as f64 - 0.5).abs() < 0.01);
    // determinism and stream separation
    let mut a = Prng::from_seed([9; 8], 0);
    let mut b = Prng::from_seed([9; 8], 0);
    let mut c2 = Prng::from_seed([9; 8], 1);
    let (x, y, z) = (a.next_u64(), b.next_u64(), c2.next_u64());
    assert_eq!(x, y);
    assert_ne!(x, z);
    let mut os1 = Prng::from_os().unwrap();
    let mut os2 = Prng::from_os().unwrap();
    assert_ne!(os1.next_u64(), os2.next_u64());
}

#[test]
fn chacha20_rfc8439_block() {
    // RFC 8439 2.3.2 test vector: key 00..1f, counter 1, nonce 00:00:00:09 00:00:00:4a 00:00:00:00
    let key: [u32; 8] = core::array::from_fn(|i| u32::from_le_bytes([4 * i as u8, 4 * i as u8 + 1, 4 * i as u8 + 2, 4 * i as u8 + 3]));
    let out = chacha_block(&key, 1, 0x0900_0000, 0x4a00_0000, 0, 20);
    assert_eq!(out[0], 0xe4e7f110);
    assert_eq!(out[15], 0x4e3c50a2);
}
