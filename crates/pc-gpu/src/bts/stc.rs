//! SlotsToCoeffs of the shipped order (FIDESlib `btsStCEnter` / `btsStCStage`): the three BSGS stages of
//! `pc_core::bts_setup` (OpenFHE `EvalSlotsToCoeffsPrecompute` in FIDESlib's LT-friendly layout, scaleDec = q0 / sf(0)
//! folded into the last stage), encoded at `pc_shadow::route::stc_stage_levels`, run by the same `LtStage` as CtS.
//! Stages 0..n-2 end with the fused ModDown-rescale; the last one pays the accumulated LT-friendly offset and keeps
//! degree 2 (the planner's landing convention).
//!
//! Real payload (FIDESLIB_BTS_REAL, `stcRealA0`): stage 0 with the entry of every diagonal that reads coefficient 0
//! halved, so one EvalMod chain on coefficients 0..N/2-1 gives w with slots = w + conj w (the finish adds the
//! conjugate). Derivation: docs/ARCHITECTURE.md "pc-gpu StC real payload".
use crate::error::{shape, Result};
use pc_core::bts_setup::{rotate_left, BtsFactors, C64};

/// OpenFHE's StC decoding constant on a composite chain: q0 q1 ... (the bottom level's product, as a double, in
/// OpenFHE's multiplication order) / ScalingFactorReal(0).
pub fn scale_dec(st: &pc_core::scale::ScaleTables) -> f64 {
    let mut q = st.q[0] as f64;
    for j in 1..st.d {
        q *= st.q[j] as f64;
    }
    q / st.sf_real_openfhe[0]
}

/// The real-payload stage-0 diagonals (slot values, LT-friendly pre-rotation applied), aligned with
/// `fides_layout().stc[0].pts`.
pub fn real_stage0(f: &BtsFactors) -> Result<Vec<Vec<C64>>> {
    let st = f.stc.first().ok_or_else(|| shape("stc real", "no StC stage"))?;
    let lay = f.fides_layout();
    let fs = &lay.stc[0];
    let slots = st.len() as i64;
    let offset = (f.dec.num_rotations as i64 + 1) / 2 - 1;
    if f.dec.rem_collapse != 0 && f.dec.level_budget == 1 {
        return Err(shape("stc real", "a remainder stage at s = 0 is not ported"));
    }
    Ok((0..st.num_diags())
        .map(|idx| {
            let mut v = st.bsgs_diag(idx);
            let j = (idx % st.g as usize) as i64;
            let k = (offset - j).rem_euclid(slots) as usize;
            v[k] = v[k].scale(0.5);
            rotate_left(&v, fs.pt_rotation as i64)
        })
        .collect())
}
