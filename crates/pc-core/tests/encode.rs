//! Encoding: slot roundtrip within 1e-14 relative of the CKKS rounding floor at every level of both presets (full and sparse slot counts, real and
//! complex payloads), every row consistent with the exact integer encoding, coefficient rows exact.
use pc_core::encode::{Complex, Encoder, Payload};
use pc_core::params::{Chain, ChainSpec};
use pc_core::scale::ScaleTables;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// uniform in [-1, 1)
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }
}

fn rows_consistent(enc: &Encoder, rows: &[Vec<u64>], chain: &Chain) {
    let c = enc.crt_bottom(rows).unwrap();
    for (i, row) in rows.iter().enumerate() {
        let q = chain.q[i] as i128;
        assert!(row.iter().zip(&c).all(|(&r, &x)| r as i128 == x.rem_euclid(q)), "row {i} inconsistent");
    }
}

fn roundtrip(spec: ChainSpec, slot_counts: &[usize], seed: u64) {
    let chain = Chain::generate(&spec).unwrap();
    let st = ScaleTables::precompute(&chain).unwrap();
    let enc = Encoder::new(&chain);
    let mut rng = Rng(seed);
    let mut worst = 0.0f64;
    let mut worst_max = 0.0f64;
    let mut worst_excess = 0.0f64;
    let mut worst_sparse_max = 0.0f64;
    let mut cases = 0;
    for lt in (0..=chain.l()).filter(|&l| chain.on_grid(l) && l + 1 >= chain.level_unit) {
        let scale = st.sf(lt as i32).unwrap();
        for &slots in slot_counts {
            for payload in [Payload::Complex, Payload::Real] {
                let vals: Vec<Complex> = (0..slots).map(|_| Complex::new(rng.unit(), rng.unit())).collect();
                let rows = enc.encode_slots(&vals, slots, lt, payload, scale).unwrap();
                assert_eq!(rows.len(), lt + 1);
                let back = enc.decode_slots(&rows, slots, payload, scale).unwrap();
                let (mut e2, mut v2, mut emax, mut vmax) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
                for (a, b) in vals.iter().zip(&back) {
                    let want = if payload == Payload::Real { Complex::new(a.re, 0.0) } else { *a };
                    let e = (want - *b).abs();
                    e2 += e * e;
                    v2 += want.abs() * want.abs();
                    emax = emax.max(e);
                    vmax = vmax.max(want.abs());
                }
                let rel = (e2 / v2).sqrt();
                let rel_max = emax / vmax;
                // Every CKKS encoder rounds each coefficient (uniform error, variance 1/12); decoding sums 2*slots of
                // them per slot, so the slot error has an irreducible RMS of sqrt(slots/6)/scale (Real: half the
                // variance). At scale ~2^53 and 2^15 slots that floor alone is ~1e-14 of an RMS-0.8 input, so the
                // gate is the error in excess of 1.5x that floor: <= 1e-14 relative. Sparse slot counts sit far
                // above the floor and must meet 1e-14 on the max error outright.
                let comps = if payload == Payload::Real { 1.0 } else { 2.0 };
                let floor = (slots as f64 * comps / 12.0).sqrt() / scale;
                let n = slots as f64;
                let excess = ((e2 / n).sqrt() - 1.5 * floor).max(0.0) / (v2 / n).sqrt();
                assert!(excess <= 1e-14, "limb top {lt} slots {slots} {payload:?}: RMS rel {rel:e}, excess {excess:e}");
                if slots <= 512 {
                    assert!(rel_max <= 1e-14, "limb top {lt} slots {slots} {payload:?}: max relative {rel_max:e}");
                    worst_sparse_max = worst_sparse_max.max(rel_max);
                }
                worst_excess = worst_excess.max(excess);
                worst = worst.max(rel);
                worst_max = worst_max.max(rel_max);
                if slots == slot_counts[0] && payload == Payload::Complex {
                    rows_consistent(&enc, &rows, &chain);
                }
                cases += 1;
            }
        }
    }
    println!(
        "{}-bit chain: {cases} encode/decode cases, worst RMS rel {worst:e} (excess over floor {worst_excess:e}), \
         worst max rel {worst_max:e}, worst max rel at <=512 slots {worst_sparse_max:e}",
        chain.word_bits
    );
}

#[test]
fn slots_roundtrip_n32() {
    roundtrip(ChainSpec::n32(), &[32768, 512, 1], 7);
}

#[test]
fn slots_roundtrip_n64() {
    roundtrip(ChainSpec::n64(), &[32768, 64], 9);
}

#[test]
fn coeff_rows_exact() {
    let chain = Chain::generate(&ChainSpec::n32()).unwrap();
    let enc = Encoder::new(&chain);
    let mut rng = Rng(11);
    let scale = 2f64.powi(50);
    let x: Vec<f64> = (0..chain.n).map(|_| rng.unit()).collect();
    let rows = enc.encode_coeff_rows(&x, chain.l(), scale).unwrap();
    let ints = enc.crt_bottom(&rows).unwrap();
    for (j, (&xi, &c)) in x.iter().zip(&ints).enumerate() {
        assert_eq!(c, (xi * scale).round() as i128, "coefficient {j}");
    }
    rows_consistent(&enc, &rows, &chain);
    let back = enc.decode_coeff_rows(&rows, scale).unwrap();
    assert!(x.iter().zip(&back).all(|(a, b)| (a - b).abs() <= 0.5 / scale));
    // values past the 2^126 exact range are refused, not wrapped
    assert!(enc.encode_coeff_rows(&[1.0], 0, 2f64.powi(127)).is_err());
}

#[test]
fn fft_special_inverts() {
    let chain = Chain::generate(&ChainSpec::n32()).unwrap();
    let enc = Encoder::new(&chain);
    let mut rng = Rng(13);
    let v: Vec<Complex> = (0..32768).map(|_| Complex::new(rng.unit(), rng.unit())).collect();
    let mut w = v.clone();
    enc.fft_special_inv(&mut w);
    enc.fft_special(&mut w);
    let e = v.iter().zip(&w).map(|(a, b)| (*a - *b).abs()).fold(0.0, f64::max);
    println!("fft roundtrip 32768: max abs err {e:e}");
    assert!(e < 1e-14, "{e:e}");
}
