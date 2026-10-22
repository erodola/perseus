//! `PtStore` for Python: the session's plaintext store as the planned driver fills it. A bound program reads its
//! plaintexts by key: `<block>/pt_N` at the level the compiler resolved (`Program.pt_keys`). `stage` encodes on the
//! device (DeviceFft, the slot values the impl layer computed) into the key's slot: a key that is already resident
//! is re-encoded IN PLACE (same slot, same address), so a per-position mask changes value without invalidating the
//! CUDA graphs that read it; stream order puts the rewrite after every earlier reader.
use crate::session::{gerr, nogil, Dev, State};
use pc_core::encode::Complex;
use pc_gpu::encode_dev::{Payload2, PtForm};
use pc_gpu::pt_store::PtKey;
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::rc::Rc;

/// A slot vector (numpy / sequence; real or complex) as host complex values.
pub(crate) fn host_values(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Vec<Complex>> {
    let np = py.import("numpy")?;
    let arr = np.call_method1("ascontiguousarray", (obj,))?.call_method0("ravel")?;
    let kind: String = arr.getattr("dtype")?.getattr("kind")?.extract()?;
    if kind == "c" {
        let a = np.call_method1("ascontiguousarray", (arr, "complex128"))?.call_method1("view", ("float64",))?;
        let v = PyBuffer::<f64>::get(&a)?.to_vec(py)?;
        Ok(v.chunks_exact(2).map(|p| Complex::new(p[0], p[1])).collect())
    } else {
        let a = np.call_method1("ascontiguousarray", (arr, "float64"))?;
        let v = PyBuffer::<f64>::get(&a)?.to_vec(py)?;
        Ok(v.into_iter().map(|x| Complex::new(x, 0.0)).collect())
    }
}

#[pyclass(name = "PtStore", module = "perseus._pc", unsendable)]
pub struct PyPtStore {
    pub(crate) dev: Rc<Dev>,
}

#[pymethods]
impl PyPtStore {
    /// Encode `items` = [(name, level, values[, deg])] on the device into their slots (in place when resident).
    /// Returns the number encoded. Values: N/2 slot values (or a power-of-two count, periodically extended).
    fn stage(&self, py: Python<'_>, items: Vec<Bound<'_, PyAny>>) -> PyResult<usize> {
        let mut parsed: Vec<(PtKey, Vec<Complex>, u8)> = Vec::with_capacity(items.len());
        for it in &items {
            let t: Vec<Bound<'_, PyAny>> = it.extract()?;
            if t.len() < 3 {
                return Err(PyValueError::new_err("PtStore.stage item = (name, level, values[, deg])"));
            }
            let name: String = t[0].extract()?;
            let level: u32 = t[1].extract()?;
            let deg: u8 = if t.len() > 3 { t[3].extract()? } else { 1 };
            parsed.push((PtKey::new(name, level, PtForm::Eval), host_values(py, &t[2])?, deg));
        }
        let dev = &self.dev;
        let payload = dev.payload();
        let n = parsed.len();
        nogil(py, || -> PyResult<()> {
            let mut st = dev.st.borrow_mut();
            let State { pts, ex, ops, .. } = &mut *st;
            for (key, values, deg) in &parsed {
                let slots = values.len().next_power_of_two().max(1);
                pts.encode_into(&dev.ctx, ex, ops, &dev.sh, key.clone(), Payload2::Slots { values, slots, payload }, *deg).map_err(gerr)?;
            }
            Ok(())
        })?;
        Ok(n)
    }

    /// Join staged uploads of `keys` = [(name, level)] into the compute lane.
    fn adopt(&self, keys: Vec<(String, u32)>) -> PyResult<()> {
        let mut st = self.dev.st.borrow_mut();
        let State { pts, ex, .. } = &mut *st;
        let ks: Vec<PtKey> = keys.into_iter().map(|(n, l)| PtKey::new(n, l, PtForm::Eval)).collect();
        pts.adopt(&self.dev.ctx, ex, &ks).map_err(gerr)
    }

    /// Drop `keys` = [(name, level)]; returns how many were resident. A bound program reading one fails its next
    /// run (`resolve_pts`) until it is staged again.
    fn evict(&self, keys: Vec<(String, u32)>) -> PyResult<usize> {
        let mut st = self.dev.st.borrow_mut();
        let State { pts, ex, .. } = &mut *st;
        let mut n = 0;
        for (name, level) in keys {
            n += pts.evict(ex, &PtKey::new(name, level, PtForm::Eval)).map_err(gerr)? as usize;
        }
        Ok(n)
    }

    fn has(&self, name: &str, level: u32) -> bool {
        self.dev.st.borrow().pts.has(&PtKey::new(name, level, PtForm::Eval))
    }

    fn __len__(&self) -> usize {
        self.dev.st.borrow().pts.len()
    }

    fn __repr__(&self) -> PyResult<String> {
        let st = self.dev.st.try_borrow().map_err(|_| PyRuntimeError::new_err("session busy"))?;
        Ok(format!("PtStore({} resident of {} slots)", st.pts.len(), st.pts.cfg.slots))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPtStore>()
}
