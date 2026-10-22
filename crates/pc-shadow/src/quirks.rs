//! Named deviations from the C++ truth (FIDESlib `Ciphertext.cpp` + the perseus wrapper) that the planner mirror
//! (`perseus/plan/placer/sim.py`) and the numpy fakes (`perseus/impl/fake.py`, `fakecap.py`) carry, and that
//! shipped plans depend on. The rules implement the C++ behaviour; a flag in `ShadowConfig::quirks` switches one
//! rule to the deviating behaviour. `docs/QUIRKS.md` documents each: what, where it came from, who depends on it.
//! Retiring one = re-planning without the flag and passing the plan gates.
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Quirk {
    // ── planner (sim.py) ──
    /// ct(deg 2) + pt(deg 1) lands at (effective level, deg 1) whatever the plaintext's level (sim.py `unit != 1`
    /// branch, fakecap `_add_pt`). C++ `addPt` rescales only when the plaintext sits exactly one level deeper;
    /// otherwise it lifts the plaintext and the sum stays degree 2.
    AddPtRealizes,
    /// Products read the cipher inputs only: a plaintext's level is ignored (sim.py MULT_FAMILY). C++ refuses a
    /// plaintext deeper than its ciphertext (`multPt` asserts after a failed `adjustPlaintextToCiphertext`).
    PtLevelIgnored,
    // ── fake.py ──
    /// Products land rescaled: (level + unit, deg 1). C++: (effective level, deg 2), rescaled lazily.
    FakeProductRescaled,
    /// ct + ct = (max level, max degree). C++: the FLEXIBLEAUTO mixed-degree adjust.
    FakeAddMaxDeg,
    /// negate keeps level and degree. C++ GPU `EvalNegate` is `multScalar(-1)`: (effective level, deg 2).
    FakeNegateFree,
    /// ct + pt keeps the ciphertext's level and degree.
    FakeAddPtFree,
    /// Every refresh lands at `bts_out` with degree 1, any route.
    FakeBootstrapRealized,
    // ── fakecap.py ──
    /// StC-first entry compares the NOMINAL input level with FAKECAP_STC_ENTRY. C++ (`Bootstrap.cu` bootstrapImpl /
    /// btsBegin) compares the effective limb (`top - d` when deg 2) with `stc_first_entry`.
    CapStcEntryNominal,
    /// ct * pt lands at max(effective level, pt level): a deeper plaintext drags the product down (C++ refuses).
    CapMultPtMax,
}

impl Quirk {
    pub const ALL: [Quirk; 9] = [
        Quirk::AddPtRealizes,
        Quirk::PtLevelIgnored,
        Quirk::FakeProductRescaled,
        Quirk::FakeAddMaxDeg,
        Quirk::FakeNegateFree,
        Quirk::FakeAddPtFree,
        Quirk::FakeBootstrapRealized,
        Quirk::CapStcEntryNominal,
        Quirk::CapMultPtMax,
    ];
}

/// A set of enabled quirks.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Quirks(pub BTreeSet<Quirk>);

impl Quirks {
    pub fn none() -> Quirks {
        Quirks::default()
    }
    pub fn of(qs: &[Quirk]) -> Quirks {
        Quirks(qs.iter().copied().collect())
    }
    /// The planner: the C++ rules, its plaintexts at the level the compile encodes them (`Shadow::step`), no quirk.
    pub fn planner() -> Quirks {
        Quirks::none()
    }
    /// What `fake.py::FakeFhe` does.
    pub fn fake() -> Quirks {
        Quirks::of(&[
            Quirk::FakeProductRescaled,
            Quirk::FakeAddMaxDeg,
            Quirk::FakeNegateFree,
            Quirk::FakeAddPtFree,
            Quirk::FakeBootstrapRealized,
            Quirk::PtLevelIgnored,
        ])
    }
    /// What `fakecap.py::CapFhe` does.
    pub fn fakecap() -> Quirks {
        Quirks::of(&[Quirk::AddPtRealizes, Quirk::CapStcEntryNominal, Quirk::CapMultPtMax])
    }
    pub fn has(&self, q: Quirk) -> bool {
        self.0.contains(&q)
    }
}
