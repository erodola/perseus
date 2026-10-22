//! The canonical ciphertext state and its two derived views.
//!
//! ONE representation is stored: `CtMeta::level` counts the primes dropped from the top of Q (OpenFHE `GetLevel`,
//! the perseus wrapper's `level_of`, the level every graph.json and plan file is written in). The two other
//! conventions of the planner and FIDESlib are computed, never stored:
//!   * the planner's consumed depth relative to the bootstrap level (`sim.py` `lc`/`dg`, in primes; `eff` adds one
//!     CKKS level = `unit` primes while a rescale is pending): `consumed = level - bootstrap_level`;
//!   * FIDESlib's top limb index (`Ciphertext::getLevel()`, `sfAtLimb`, `ElemForEvalMult`): `top = |Q| - 1 - level`.
use crate::error::{Error, Result};
use crate::tag::Tag;
use serde::{Deserialize, Serialize};

/// The chain shape the conversions need: `n_q` = |Q| (primes, bottom q0 included) and `unit` = primes per CKKS level
/// (the composite degree; 2 on the n32 chain).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChainShape {
    pub n_q: u32,
    pub unit: u32,
}

impl ChainShape {
    /// The shipped 32-bit composite chain: 54 primes, 2 per level.
    pub const N32: ChainShape = ChainShape { n_q: 54, unit: 2 };

    /// A chain with no bottom, for the planner's mirror: it predicts levels past the real chain (that is how a
    /// plan-free run finds the over-budget nodes), so its rules must not stop at q0. Levels only.
    pub const fn unbounded(unit: u32) -> ChainShape {
        ChainShape { n_q: 1 << 20, unit }
    }

    pub fn from_tables(st: &pc_core::scale::ScaleTables) -> ChainShape {
        ChainShape { n_q: st.q.len() as u32, unit: st.d as u32 }
    }
    /// Top limb index of a fresh (level 0) ciphertext.
    pub fn top_max(&self) -> u32 {
        self.n_q - 1
    }
}

/// Ciphertext bookkeeping: everything an op rule reads or writes.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CtMeta {
    /// Primes dropped (OpenFHE / perseus convention).
    pub level: u32,
    /// Noise-scale degree: 1, or 2 = a rescale is pending.
    pub deg: u8,
    /// NoiseFactor (the scale), 0.0 when the Shadow runs without scale tables (levels only).
    pub nf: f64,
    /// Live slots (N/2 dense).
    pub slots: u32,
    #[serde(default)]
    pub tag: Tag,
}

impl CtMeta {
    pub fn new(level: u32, deg: u8) -> CtMeta {
        CtMeta { level, deg, nf: 0.0, slots: 0, tag: Tag::default() }
    }
    pub fn with_nf(mut self, nf: f64) -> CtMeta {
        self.nf = nf;
        self
    }
    /// The level the value really sits at once its pending rescale is paid (what thresholds and encodes read).
    pub fn effective_level(&self, unit: u32) -> u32 {
        effective_level(self.level, self.deg, unit)
    }
}

pub fn effective_level(level: u32, deg: u8, unit: u32) -> u32 {
    level + if deg == 2 { unit } else { 0 }
}

/// FIDESlib's top limb index of a ciphertext at `level`.
pub fn top_limb(chain: ChainShape, level: u32) -> Result<u32> {
    chain.top_max().checked_sub(level).ok_or_else(|| Error::Range {
        what: "top_limb",
        msg: format!("level {level} is past the bottom of a {}-prime chain", chain.n_q),
    })
}

/// The level of a ciphertext whose FIDESlib top limb index is `top`.
pub fn level_from_top_limb(chain: ChainShape, top: u32) -> Result<u32> {
    chain.top_max().checked_sub(top).ok_or_else(|| Error::Range {
        what: "level_from_top_limb",
        msg: format!("top limb {top} is past the top of a {}-prime chain", chain.n_q),
    })
}

/// The planner's consumed depth (`sim.py` `lc`): primes below `bootstrap_level`; negative for a landing richer
/// than the bootstrap level (the sparse routes).
pub fn planner_consumed(level: u32, bootstrap_level: u32) -> i64 {
    level as i64 - bootstrap_level as i64
}

/// The planner's effective consumed depth (`sim.py` `eff`): consumed plus one unit while a rescale is pending.
pub fn planner_effective(m: &CtMeta, bootstrap_level: u32, unit: u32) -> i64 {
    planner_consumed(m.effective_level(unit), bootstrap_level)
}

/// The level of a planner state `consumed` primes below `bootstrap_level`.
pub fn level_from_planner(consumed: i64, bootstrap_level: u32) -> Result<u32> {
    let l = bootstrap_level as i64 + consumed;
    u32::try_from(l).map_err(|_| Error::Range {
        what: "level_from_planner",
        msg: format!("consumed {consumed} below bootstrap level {bootstrap_level} is above the chain top"),
    })
}

/// The planner keeps consumed depth as a float; every shipped value is integral. Refuse anything else rather than
/// round it silently (a fractional level has no ciphertext meaning).
pub fn level_from_planner_f64(consumed: f64, bootstrap_level: u32) -> Result<u32> {
    if consumed.fract() != 0.0 || !consumed.is_finite() {
        return Err(Error::Range { what: "level_from_planner_f64", msg: format!("non-integral consumed depth {consumed}") });
    }
    level_from_planner(consumed as i64, bootstrap_level)
}
