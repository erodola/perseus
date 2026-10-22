//! `fold_bootstrap`, `tag_reduce` and the sparse auto-routing decision (perseus' wrapper `fold_slots_for` /
//! `fold_bootstrap` / `auto_sparse_route_for` / `planned_route_for`, fideslib_wrapper.h; fake semantics in
//! `perseus/impl/fake.py::fold_bootstrap`).
//!
//! A sparse bootstrap at s slots of a ciphertext whose slots are NOT s-periodic returns, at every slot i, the mean of
//! the residue class i mod s over the N/2s copies (the route's fold sums the copies, its constant divides by them).
//! `fold_bootstrap(s, n_live, prescale)` uses that as a reduction: prescale (input side, one level, the inverse rides
//! the recovery), the s route, then ONE scalar product by copies / n_live / prescale: slot i = sum of the class / n_live.
//! It is destructive by design (only the class sums survive) and is never a planned refresh.
use super::sparse::RouteSet;
use super::BtsCx;
use crate::context::{DevStrategy, DevWord};
use crate::ct::Ct;
use crate::error::{shape, Error, Result};
use pc_core::encode::Complex;
use pc_shadow::tag::{Support, Tag};
use pc_shadow::{Route, Shadow};

/// `cfg.bts.sparse_auto` (SPARSE_AUTO): which unplanned refreshes the tag may route sparse.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SparseAuto {
    /// Never (0).
    Off,
    /// Only s-periodic values, identity refresh (1).
    Periodic,
    /// `min_routable_s`: periodic, or fold-collision-free (2; the fold route, not an identity).
    Fold,
    /// As `Fold` plus the masked fold restore (3).
    MaskedFold,
}

impl SparseAuto {
    pub fn from_mode(m: i64) -> SparseAuto {
        match m {
            i64::MIN..=0 => SparseAuto::Off,
            1 => SparseAuto::Periodic,
            2 => SparseAuto::Fold,
            _ => SparseAuto::MaskedFold,
        }
    }
}

/// An auto-routing decision: the route's slot count; `fold` = the value is not s-periodic (class sums: the caller
/// restores with the support mask, mode 3), `support` = the input's support (the restore mask / output tag).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AutoRoute {
    pub s: u32,
    pub fold: bool,
    pub support: Support,
}

/// The slot count a fold wanting `s_wanted` can run at: `s_wanted` when built, else the smallest built s' > s_wanted
/// (the caller pre-ladders the gap), else `total` (the dense degenerate: full ladder, plain bootstrap).
pub fn fold_slots_for(built: &[u32], s_wanted: u32, total: u32) -> u32 {
    if s_wanted >= total || built.contains(&s_wanted) {
        return s_wanted.min(total);
    }
    let mut b: Vec<u32> = built.to_vec();
    b.sort_unstable();
    b.into_iter().find(|&s| s > s_wanted).unwrap_or(total)
}

/// `auto_sparse_route_for`: None = dense. Eligibility is `periodic_at` (identity) or, in mode 3, collision freedom
/// (the masked fold route); the smallest built multiple wins.
pub fn auto_route(tag: &Tag, mode: SparseAuto, built: &[u32], total: u32) -> Option<AutoRoute> {
    if mode == SparseAuto::Off || built.is_empty() || !tag.known() {
        return None;
    }
    let want = if mode == SparseAuto::Periodic {
        let mut w = 1u32;
        while (w as i64) < tag.period {
            w <<= 1;
        }
        w
    } else {
        tag.min_routable_s() as u32
    };
    let s = fold_slots_for(built, want, total);
    if s >= total || !built.contains(&s) {
        return None;
    }
    if tag.periodic_at(s as i64) {
        return Some(AutoRoute { s, fold: false, support: tag.support });
    }
    if mode == SparseAuto::MaskedFold && !tag.support.is_empty() && !tag.support.is_dense() && tag.fold_collision_free_at(s as i64) {
        return Some(AutoRoute { s, fold: true, support: tag.support });
    }
    None
}

/// `planned_route_for`: the planned s, or the smallest built route the value's period allows when it outgrew s
/// (SM_PERIODIC=2); planned folds stay.
pub fn planned_route_for(tag: &Tag, s: u32, built: &[u32], total: u32) -> u32 {
    if s == 0 || built.is_empty() || !tag.known() || tag.periodic_at(s as i64) || tag.fold_collision_free_at(s as i64) {
        return s;
    }
    let mut want = s;
    while (want as i64) < tag.period {
        want <<= 1;
    }
    let s2 = fold_slots_for(built, want, total);
    if s2 > s && built.contains(&s2) && tag.periodic_at(s2 as i64) {
        s2
    } else {
        s
    }
}

/// `tag_reduce(stride)`: the caller asserts the value is `stride`-periodic (metadata only).
pub fn tag_reduce<W: DevWord>(sh: &Shadow, x: &mut Ct<'_, W>, stride: i64) {
    x.meta = sh.tag_reduce(&x.meta, stride).out;
}

/// The recovery constant copies / n_live / prescale of a fold at `s` of a `total`-slot value.
pub fn recovery(total: u32, s: u32, n_live: u32, prescale: f64) -> Result<f64> {
    if n_live == 0 || !(prescale > 0.0) {
        return Err(shape("fold_bootstrap", format!("n_live {n_live} must be > 0, prescale {prescale} must be > 0")));
    }
    if s == 0 || s > total || !s.is_power_of_two() {
        return Err(shape("fold_bootstrap", format!("slot count {s} must be a power of two in [1, {total}]")));
    }
    Ok((total / s) as f64 / n_live as f64 / prescale)
}

/// `fold_bootstrap(x, s, n_live, prescale)` on the device: s must be a built route of `routes` (refused otherwise:
/// folding without the route's precomputation returns garbage; ask `fold_slots_for` and pre-ladder). `route` gives
/// the call's CF / raise drop (its `slots` is replaced by s). Output: the route's raw landing times the recovery
/// constant (one scalar product, degree 2), or the realized landing when the recovery is 1 (= `Shadow::fold_bootstrap`).
pub fn fold_bootstrap<'a, W: DevWord, M: DevStrategy<W>>(
    cx: &mut BtsCx<'_, 'a, W, M>,
    routes: &RouteSet<'_, 'a, W, M>,
    x: &mut Ct<'a, W>,
    s: u32,
    n_live: u32,
    prescale: f64,
    route: &Route,
) -> Result<()> {
    let total = (cx.ctx.lay.n / 2) as u32;
    let rec = recovery(total, s, n_live, prescale)?;
    if s == total {
        return Err(shape("fold_bootstrap", "s = N/2 is the dense degenerate: ladder and bootstrap on the dense route"));
    }
    if !routes.built().contains(&s) {
        return Err(Error::Mem {
            knob: "bts.sparse_routes",
            value: s as usize,
            msg: format!("fold_bootstrap: no route for s={s} (built {:?}): add it to the Config or use fold_slots_for and pre-ladder", routes.built()),
        });
    }
    if prescale != 1.0 {
        cx.ops.mult_scalar_inplace(cx.ctx, cx.ex, cx.sh, std::slice::from_mut(x), &[Complex::new(prescale, 0.0)])?;
    }
    let r = routes.get(s)?;
    let rt = Route { slots: s, ..*route };
    r.run(cx, std::slice::from_mut(x), &rt)?;
    if rec != 1.0 {
        cx.ops.mult_scalar_inplace(cx.ctx, cx.ex, cx.sh, std::slice::from_mut(x), &[Complex::new(rec, 0.0)])?;
    } else {
        // no recovery product: land realized, as the Shadow's fold_bootstrap rule (and a planned refresh) does
        cx.ops.realize(cx.ctx, cx.ex, cx.sh, std::slice::from_mut(x))?;
    }
    Ok(())
}

/// Host reference of the fold (fake.py `fold_bootstrap`): slot i = sum over the class of i mod s / n_live.
pub fn fold_reference(v: &[Complex], s: usize, n_live: u32) -> Vec<Complex> {
    let total = v.len();
    (0..total)
        .map(|i| {
            let (mut re, mut im) = (0.0, 0.0);
            let mut j = i % s;
            while j < total {
                re += v[j].re;
                im += v[j].im;
                j += s;
            }
            Complex::new(re / n_live as f64, im / n_live as f64)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fold_slots_for_matches_the_wrapper() {
        let built = [1u32, 512];
        assert_eq!(fold_slots_for(&built, 512, 32768), 512);
        assert_eq!(fold_slots_for(&built, 1, 32768), 1);
        assert_eq!(fold_slots_for(&built, 64, 32768), 512);
        assert_eq!(fold_slots_for(&built, 1024, 32768), 32768);
        assert_eq!(fold_slots_for(&built, 65536, 32768), 32768);
    }

    #[test]
    fn auto_route_periodic_and_fold() {
        let built = [1u32, 512];
        let periodic = Tag::new(32768, 512, Support::dense());
        assert_eq!(auto_route(&periodic, SparseAuto::Periodic, &built, 32768).map(|r| (r.s, r.fold)), Some((512, false)));
        let dense = Tag::new(32768, 32768, Support::dense());
        assert_eq!(auto_route(&dense, SparseAuto::Fold, &built, 32768), None);
        assert_eq!(auto_route(&periodic, SparseAuto::Off, &built, 32768), None);
        // 8 live slots at stride 64 (one per class mod 512): the masked fold route, not an identity
        let sparse = Tag::new(32768, 32768, Support::ap(0, 64, 8));
        let r = auto_route(&sparse, SparseAuto::MaskedFold, &built, 32768);
        assert_eq!(r.map(|r| (r.s, r.fold)), Some((512, true)));
        assert_eq!(auto_route(&sparse, SparseAuto::Fold, &built, 32768), None);
    }

    #[test]
    fn recovery_constant() {
        assert_eq!(recovery(32768, 512, 768, 0.5).unwrap(), 64.0 / 768.0 / 0.5);
        assert!(recovery(32768, 3, 1, 1.0).is_err());
        assert!(recovery(32768, 512, 0, 1.0).is_err());
    }
}
