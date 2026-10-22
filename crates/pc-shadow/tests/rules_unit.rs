//! Conversions round-trip; each quirk's deviation from the C++ rule on a case that separates them; landings,
//! the StC-first fallback and the bootstrap config refusals; the scale-tracking rules against FIDESlib's adjust
//! arithmetic (every adjusted operand ends on its partner's exact scale).
use pc_core::params::{Chain, ChainSpec};
use pc_core::scale::ScaleTables;
use pc_shadow::meta::*;
use pc_shadow::route::{BtsConfig, Landing, Route};
use pc_shadow::rules::{Consts, Operand, Prelude, PtMeta, Shadow, ShadowConfig};
use pc_shadow::{Quirk, Quirks};
use std::sync::Arc;

fn shadow(q: Quirks) -> Shadow {
    Shadow::new(ShadowConfig { chain: ChainShape::N32, bts: BtsConfig::n32_shipped(), quirks: q }, None)
}
fn ct(level: u32, deg: u8) -> CtMeta {
    CtMeta::new(level, deg)
}
fn pt(level: u32) -> PtMeta {
    PtMeta { level, deg: 1, nf: 0.0, tag: Default::default() }
}
fn ld(m: &CtMeta) -> (u32, u8) {
    (m.level, m.deg)
}

#[test]
fn conversions_round_trip() {
    let c = ChainShape::N32;
    let mut n = 0;
    for level in 0..c.n_q {
        let top = top_limb(c, level).unwrap();
        assert_eq!(level_from_top_limb(c, top).unwrap(), level);
        for bl in [0u32, 16, 28, 34, 46] {
            let cons = planner_consumed(level, bl);
            assert_eq!(level_from_planner(cons, bl).unwrap(), level);
            assert_eq!(level_from_planner_f64(cons as f64, bl).unwrap(), level);
            for deg in [1u8, 2] {
                let m = CtMeta::new(level, deg);
                assert_eq!(planner_effective(&m, bl, c.unit), cons + if deg == 2 { 2 } else { 0 });
                n += 1;
            }
        }
    }
    assert!(top_limb(c, 54).is_err());
    assert!(level_from_top_limb(c, 54).is_err());
    assert!(level_from_planner(-29, 28).is_err());
    assert!(level_from_planner_f64(0.5, 28).is_err());
    println!("[conv] {n} (level, bootstrap level, deg) round trips");
}

#[test]
fn flexibleauto_level_rules() {
    let s = shadow(Quirks::none());
    // mixed-degree add: the deg-2 operand deeper or level -> stays deg 2 at its level
    assert_eq!(ld(&s.add(&ct(30, 1), &ct(30, 2)).unwrap().out), (30, 2));
    assert_eq!(ld(&s.add(&ct(28, 1), &ct(30, 2)).unwrap().out), (30, 2));
    // the deg-2 operand richer: realized onto the deg-1 level
    assert_eq!(ld(&s.add(&ct(28, 2), &ct(30, 1)).unwrap().out), (30, 1));
    assert_eq!(ld(&s.add(&ct(24, 2), &ct(30, 1)).unwrap().out), (30, 1));
    // products: deg 2 at the deeper effective level
    assert_eq!(ld(&s.mult(&ct(28, 2), &ct(30, 1)).unwrap().out), (30, 2));
    assert_eq!(ld(&s.mult(&ct(30, 2), &ct(26, 1)).unwrap().out), (32, 2));
    assert_eq!(ld(&s.square(&ct(30, 2)).unwrap().out), (32, 2));
    assert_eq!(ld(&s.mult_scalar(&ct(30, 1), 3.0).unwrap().out), (30, 2));
    assert_eq!(ld(&s.negate(&ct(30, 2)).unwrap().out), (32, 2));
    // ct * pt rescales first; a deeper plaintext is refused
    let r = s.mult_pt(&ct(30, 2), &pt(32)).unwrap();
    assert_eq!((ld(&r.out), r.pre.clone()), ((32, 2), vec![Prelude::Rescale(Operand::A)]));
    assert!(s.mult_pt(&ct(30, 1), &pt(32)).is_err());
    // ct + pt: realized only against a plaintext one level deeper
    assert_eq!(ld(&s.add_pt(&ct(30, 2), &pt(32)).unwrap().out), (32, 1));
    assert_eq!(ld(&s.add_pt(&ct(30, 2), &pt(30)).unwrap().out), (30, 2));
    assert_eq!(ld(&s.add_pt(&ct(30, 1), &pt(34)).unwrap().out), (34, 1));
    // drop / realize
    assert_eq!(ld(&s.drop_to_level(&ct(30, 2), 34).out), (34, 2));
    assert_eq!(ld(&s.drop_to_level(&ct(30, 2), 26).out), (30, 2));
    assert_eq!(ld(&s.realize(&ct(30, 2)).unwrap().out), (32, 1));
}

#[test]
fn each_quirk_deviates() {
    let exact = shadow(Quirks::none());
    let with = |q: Quirk| shadow(Quirks::of(&[q]));
    // a hint reads the nominal level unless the caller accounts the pending rescale; a fired hint lands realized
    let r = Route::dense();
    assert!(!exact.hint(&ct(44, 2), 44, false, &r).unwrap().fired);
    assert!(exact.hint(&ct(44, 2), 44, true, &r).unwrap().fired);
    assert_eq!(ld(&exact.hint(&ct(46, 1), 44, false, &r).unwrap().recipe.out), (28, 1));
    // AddPtRealizes
    assert_eq!(ld(&with(Quirk::AddPtRealizes).add_pt(&ct(30, 2), &pt(30)).unwrap().out), (32, 1));
    // PtLevelIgnored / CapMultPtMax: a deeper plaintext
    assert_eq!(ld(&with(Quirk::PtLevelIgnored).mult_pt(&ct(30, 1), &pt(34)).unwrap().out), (30, 2));
    assert_eq!(ld(&with(Quirk::CapMultPtMax).mult_pt(&ct(30, 1), &pt(34)).unwrap().out), (34, 2));
    // FakeProductRescaled / FakeAddMaxDeg / FakeNegateFree / FakeAddPtFree
    assert_eq!(ld(&with(Quirk::FakeProductRescaled).mult(&ct(30, 1), &ct(30, 1)).unwrap().out), (32, 1));
    assert_eq!(ld(&with(Quirk::FakeAddMaxDeg).add(&ct(28, 2), &ct(30, 1)).unwrap().out), (30, 2));
    assert_eq!(ld(&with(Quirk::FakeNegateFree).negate(&ct(30, 2)).unwrap().out), (30, 2));
    assert_eq!(ld(&with(Quirk::FakeAddPtFree).add_pt(&ct(30, 2), &pt(32)).unwrap().out), (30, 2));
    // FakeBootstrapRealized: every route lands on the dense landing
    assert_eq!(ld(&exact.bootstrap(&ct(40, 1), &Route::sparse(1)).unwrap().out), (16, 1));
    assert_eq!(ld(&with(Quirk::FakeBootstrapRealized).bootstrap(&ct(40, 1), &Route::sparse(1)).unwrap().out), (28, 1));
    // CapStcEntryNominal: nominal 46 deg 2 (effective 48) is past the entry only effectively
    let stc = Route { stc_first: true, ..Route::dense() };
    assert_eq!(ld(&exact.bootstrap(&ct(46, 2), &stc).unwrap().out), (34, 1));
    assert_eq!(ld(&with(Quirk::CapStcEntryNominal).bootstrap(&ct(46, 2), &stc).unwrap().out), (28, 1));
}

#[test]
fn dense_shipped_landing() {
    // unset: the one dense landing serves both orders (n32_shipped)
    assert_eq!(ld(&shadow(Quirks::none()).bootstrap(&ct(40, 1), &Route::dense()).unwrap().out), (28, 1));
    let mut bts = BtsConfig::n32_shipped();
    bts.dense_shipped = Some(Landing { level: 34, deg: 1 });
    let s = Shadow::new(ShadowConfig { chain: ChainShape::N32, bts, quirks: Quirks::none() }, None);
    let stc = Route { stc_first: true, ..Route::dense() };
    // the shipped order lands at dense_shipped (+ its raise drop); StC-first keeps landings[0] up to its entry
    assert_eq!(ld(&s.bootstrap(&ct(40, 1), &Route::dense()).unwrap().out), (34, 1));
    assert_eq!(ld(&s.bootstrap(&ct(40, 1), &Route { raise_drop: 1, ..Route::dense() }).unwrap().out), (36, 1));
    assert_eq!(ld(&s.bootstrap(&ct(40, 1), &stc).unwrap().out), (28, 1));
    assert_eq!(ld(&s.bootstrap(&ct(46, 2), &stc).unwrap().out), (34, 1));
    assert_eq!(ld(&s.bootstrap(&ct(40, 1), &Route::sparse(512)).unwrap().out), (26, 1));
    // a fired hint on the dense route (not StC-first) lands like the shipped order
    assert_eq!(ld(&s.hint(&ct(46, 1), 44, false, &Route::dense()).unwrap().recipe.out), (34, 1));
}

#[test]
fn landings_and_refusals() {
    let s = shadow(Quirks::none());
    assert_eq!(ld(&s.bootstrap(&ct(40, 2), &Route::sparse(512)).unwrap().out), (26, 1));
    let rd = Route { raise_drop: 3, ..Route::dense() };
    assert_eq!(ld(&s.bootstrap(&ct(40, 1), &rd).unwrap().out), (34, 1));
    assert!(s.bootstrap(&ct(40, 1), &Route { raise_drop: 2, ..Route::dense() }).is_err());
    assert!(s.bootstrap(&ct(50, 1), &Route::dense()).is_err());
    assert!(s.bootstrap(&ct(40, 1), &Route { cf: Some(1), ..Route::dense() }).is_err());
    assert!(s.bootstrap(&ct(40, 1), &Route::sparse(64)).is_err());
    // fold: the s-slot landing, then the recovery product (one level, deg 2); a non-dividing s is refused
    let fold = s.fold_bootstrap(&CtMeta { slots: 32768, ..ct(40, 1) }, 512, 768, 1.0).unwrap();
    assert_eq!(ld(&fold.out), (26, 2));
    assert!(s.fold_bootstrap(&CtMeta { slots: 32768, ..ct(40, 1) }, 3, 768, 1.0).is_err());
    let t = pc_shadow::tag::Tag::new(32768, 32768, pc_shadow::tag::Support::dense());
    let out = s.bootstrap(&CtMeta { tag: t, ..ct(40, 1) }, &Route::sparse(512)).unwrap().out;
    assert_eq!(out.tag.period, 512);
}

fn tables() -> Arc<ScaleTables> {
    Arc::new(ScaleTables::precompute(&Chain::generate(&ChainSpec::n32()).unwrap()).unwrap())
}

#[test]
fn scale_tracking() {
    let st = tables();
    let s = Shadow::new(ShadowConfig { chain: ChainShape::N32, bts: BtsConfig::n32_shipped(), quirks: Quirks::none() }, Some(st.clone()));
    let sf = |level: u32| st.sf((53 - level) as i32).unwrap();
    let fresh = |level: u32| CtMeta { level, deg: 1, nf: sf(level), slots: 32768, tag: Default::default() };
    // a product rescaled lands on the next level's scale exactly (FLEXIBLEAUTO invariant)
    let sq = s.square(&fresh(30)).unwrap().out;
    let r = s.rescale(&sq).unwrap().out;
    assert_eq!((r.level, r.deg), (32, 1));
    assert_eq!(r.nf.to_bits(), sf(32).to_bits());
    // every add adjusts its richer operand onto the other's exact scale
    for (la, da, lb, db) in [(26, 1, 30, 1), (26, 2, 30, 1), (26, 1, 30, 2), (26, 2, 30, 2), (28, 2, 30, 1), (30, 1, 30, 2)] {
        let mk = |l: u32, d: u8| if d == 1 { fresh(l) } else { s.square(&fresh(l - 2)).unwrap().out };
        let (a, b) = (mk(la, da), mk(lb, db));
        let out = s.add(&a, &b).unwrap().out;
        let same = (out.level, out.deg) == (b.level, b.deg) || (out.level, out.deg) == (a.level, a.deg);
        assert!(same || out.deg == 1, "add {la}/{da} + {lb}/{db} -> {out:?}");
        // the operand that was moved ends at the partner's nf (FIDESlib asserts it within 1e-3, then copies it)
        let target = if (out.level, out.deg) == (b.level, b.deg) { b.nf } else { a.nf };
        assert_eq!(out.nf.to_bits(), target.to_bits(), "add {la}/{da} + {lb}/{db}");
    }
    // constants: add_scalar rows are ElemForEvalAddOrSub, negated mod q
    let r = s.add_scalar(&fresh(30), -0.5).unwrap();
    let Some(Consts::Add(row)) = r.consts else { panic!("no constants") };
    let pos = st.elem_for_eval_add(23, 0.5, 1).unwrap();
    assert!(row.iter().zip(&pos).zip(&st.q).all(|((n, p), q)| *n == q - p));
}
