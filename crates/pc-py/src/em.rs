//! A read-only view of the EvalMod program (`pc_shadow::evalmod::EmProgram`): ops, constant rows, the
//! ElemForEvalMult requests, built here from an input state and OpenFHE's Paterson-Stockmeyer plan.
use crate::level_err;
use pc_shadow::evalmod::{build_evalmod, EmProgram, PsPlan};
use pc_shadow::fides::FMeta;
use pyo3::prelude::*;

#[pyclass(name = "EmProgram", module = "perseus._pc", frozen)]
pub struct PyEmProgram {
    p: EmProgram,
    out: Option<FMeta>,
}

fn json_err(e: serde_json::Error) -> PyErr {
    crate::ConfigError::new_err(format!("EmProgram json: {e}"))
}

fn to_py(py: Python<'_>, s: String) -> PyResult<Py<PyAny>> {
    Ok(py.import("json")?.call_method1("loads", (s,))?.unbind())
}

#[pymethods]
impl PyEmProgram {
    /// The n32 chain's EvalMod program for an input at FIDESlib (top limb, NoiseLevel, NoiseFactor), the PS plan
    /// as JSON `{"k", "m", "divqr_q", "divcs_q", "s2"}` (d3 `params.json` `cheb_ps`), `double_angle_its` steps.
    #[staticmethod]
    fn build(top: i32, nl: i32, nf: f64, ps_json: &str, double_angle_its: i32) -> PyResult<Self> {
        let ps: PsPlan = serde_json::from_str(ps_json).map_err(json_err)?;
        let chain = pc_core::params::Chain::generate(&pc_core::params::ChainSpec::n32())
            .map_err(|e| crate::ConfigError::new_err(e.to_string()))?;
        let st = pc_core::scale::ScaleTables::precompute(&chain).map_err(|e| crate::ConfigError::new_err(e.to_string()))?;
        let (p, out) = build_evalmod(&st, FMeta { top, nl, nf }, &ps, double_angle_its).map_err(level_err)?;
        Ok(PyEmProgram { p, out: Some(out) })
    }
    #[staticmethod]
    fn from_json(s: &str) -> PyResult<Self> {
        Ok(PyEmProgram { p: serde_json::from_str(s).map_err(json_err)?, out: None })
    }
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.p).map_err(json_err)
    }
    #[getter]
    fn n_ops(&self) -> usize {
        self.p.ops.len()
    }
    #[getter]
    fn n_consts(&self) -> usize {
        self.p.consts.len()
    }
    /// The ops as dicts (`{"Mult": {"dst": .., "a": .., ...}}`).
    #[getter]
    fn ops(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        to_py(py, serde_json::to_string(&self.p.ops).map_err(json_err)?)
    }
    /// The peephole-fused op list the runtime executes.
    #[getter]
    fn fused_ops(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        to_py(py, serde_json::to_string(&self.p.fused()).map_err(json_err)?)
    }
    fn consts(&self, k: usize) -> PyResult<Vec<u64>> {
        self.p.consts.get(k).cloned().ok_or_else(|| pyo3::exceptions::PyIndexError::new_err(k))
    }
    /// Every ElemForEvalMult(level, level_in, operand) request (top limb indices).
    #[getter]
    fn reqs(&self) -> Vec<(i32, i32, f64)> {
        self.p.reqs.clone()
    }
    /// The output metadata (top, NoiseLevel, NoiseFactor) when built here.
    #[getter]
    fn out(&self) -> Option<(i32, i32, f64)> {
        self.out.map(|f| (f.top, f.nl, f.nf))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEmProgram>()?;
    Ok(())
}
