//! Pack tags (`pc_shadow::tag`, the fakecap.py algebra ported 1:1): `Support`, `Tag` and the propagation functions
//! as the submodule `_pc.tag`. `conj_of` is an integer ciphertext id the caller chooses (the original stored the
//! ciphertext object and compared identities).
use pc_shadow::tag::{self as t, Support, Tag};
use pyo3::prelude::*;

#[pyclass(name = "Support", module = "perseus._pc", frozen)]
#[derive(Clone, Copy)]
pub struct PySupport(pub Support);

#[pymethods]
impl PySupport {
    #[new]
    #[pyo3(signature = (kind=t::DENSE, offset=0, stride=1, count=0, width=1))]
    fn new(kind: i32, offset: i64, stride: i64, count: i64, width: i64) -> Self {
        PySupport(Support::new(kind, offset, stride, count, width))
    }
    #[staticmethod]
    fn empty() -> Self {
        PySupport(Support::empty())
    }
    #[staticmethod]
    fn dense() -> Self {
        PySupport(Support::dense())
    }
    #[staticmethod]
    fn ap(off: i64, st: i64, cnt: i64) -> Self {
        PySupport(Support::ap(off, st, cnt))
    }
    #[staticmethod]
    fn block(off: i64, st: i64, w: i64, cnt: i64) -> Self {
        PySupport(Support::block(off, st, w, cnt))
    }
    #[getter]
    fn kind(&self) -> i32 {
        self.0.kind
    }
    #[getter]
    fn offset(&self) -> i64 {
        self.0.offset
    }
    #[getter]
    fn stride(&self) -> i64 {
        self.0.stride
    }
    #[getter]
    fn count(&self) -> i64 {
        self.0.count
    }
    #[getter]
    fn width(&self) -> i64 {
        self.0.width
    }
    fn is_dense(&self) -> bool {
        self.0.is_dense()
    }
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    fn __eq__(&self, o: &Self) -> bool {
        self.0 == o.0
    }
    fn __repr__(&self) -> String {
        let s = self.0;
        format!("Support(kind={}, offset={}, stride={}, count={}, width={})", s.kind, s.offset, s.stride, s.count, s.width)
    }
}

/// A mutable pack tag (Python code sets `field` / `conj_of` on fresh tags, as fakecap did).
#[pyclass(name = "Tag", module = "perseus._pc")]
#[derive(Clone, Copy)]
pub struct PyTag(pub Tag);

#[pymethods]
impl PyTag {
    #[new]
    #[pyo3(signature = (slots=0, period=0, support=None, field=t::F_UNKNOWN, conj_of=None))]
    fn new(slots: i64, period: i64, support: Option<PySupport>, field: i32, conj_of: Option<u64>) -> Self {
        let mut x = Tag::new(slots, period, support.map(|s| s.0).unwrap_or_default());
        x.field = field;
        x.conj_of = conj_of;
        PyTag(x)
    }
    #[getter]
    fn slots(&self) -> i64 {
        self.0.slots
    }
    #[setter]
    fn set_slots(&mut self, v: i64) {
        self.0.slots = v;
    }
    #[getter]
    fn period(&self) -> i64 {
        self.0.period
    }
    #[setter]
    fn set_period(&mut self, v: i64) {
        self.0.period = v;
    }
    #[getter]
    fn support(&self) -> PySupport {
        PySupport(self.0.support)
    }
    #[setter]
    fn set_support(&mut self, v: PySupport) {
        self.0.support = v.0;
    }
    #[getter]
    fn field(&self) -> i32 {
        self.0.field
    }
    #[setter]
    fn set_field(&mut self, v: i32) {
        self.0.field = v;
    }
    #[getter]
    fn conj_of(&self) -> Option<u64> {
        self.0.conj_of
    }
    #[setter]
    fn set_conj_of(&mut self, v: Option<u64>) {
        self.0.conj_of = v;
    }
    fn known(&self) -> bool {
        self.0.known()
    }
    fn periodic_at(&self, s: i64) -> bool {
        self.0.periodic_at(s)
    }
    fn fold_collision_free_at(&self, s: i64) -> bool {
        self.0.fold_collision_free_at(s)
    }
    fn min_routable_s(&self) -> i64 {
        self.0.min_routable_s()
    }
    fn copy(&self) -> Self {
        *self
    }
    fn __eq__(&self, o: &Self) -> bool {
        self.0 == o.0
    }
    fn __repr__(&self) -> String {
        let x = self.0;
        format!(
            "Tag(slots={}, period={}, support=({}, {}, {}, {}, {}), field={}, conj_of={:?})",
            x.slots, x.period, x.support.kind, x.support.offset, x.support.stride, x.support.count, x.support.width,
            x.field, x.conj_of
        )
    }
}

#[pyfunction]
fn top(slots: i64) -> PyTag {
    PyTag(t::top(slots))
}
#[pyfunction]
fn with_field(x: &PyTag, f: i32) -> PyTag {
    PyTag(t::with_field(&x.0, f))
}
#[pyfunction]
fn t_mult_scalar(x: &PyTag, c: f64) -> PyTag {
    PyTag(t::t_mult_scalar(&x.0, c))
}
#[pyfunction]
fn t_add_scalar(x: &PyTag, c: f64) -> PyTag {
    PyTag(t::t_add_scalar(&x.0, c))
}
#[pyfunction]
fn t_square(x: &PyTag) -> PyTag {
    PyTag(t::t_square(&x.0))
}
#[pyfunction]
fn t_add(a: &PyTag, b: &PyTag) -> PyTag {
    PyTag(t::t_add(&a.0, &b.0))
}
#[pyfunction]
fn t_mult(a: &PyTag, b: &PyTag) -> PyTag {
    PyTag(t::t_mult(&a.0, &b.0))
}
#[pyfunction]
fn t_mult_i(x: &PyTag) -> PyTag {
    PyTag(t::t_mult_i(&x.0))
}
#[pyfunction]
fn t_rotate(x: &PyTag, k: i64) -> PyTag {
    PyTag(t::t_rotate(&x.0, k))
}
#[pyfunction]
fn t_conjugate(x: &PyTag) -> PyTag {
    PyTag(t::t_conjugate(&x.0))
}
#[pyfunction]
fn t_bootstrap(x: &PyTag, s: i64, slots: i64) -> PyTag {
    PyTag(t::t_bootstrap(&x.0, s, slots))
}
#[pyfunction]
fn t_reduce(x: &PyTag, slots: i64, stride: i64) -> PyTag {
    PyTag(t::t_reduce(&x.0, slots, stride))
}
/// `t` with the field of x + conj(x) (real) / x - conj(x) (imaginary) when one operand (id `a` / `b`) is the
/// other's conjugate.
#[pyfunction]
fn conj_pair(x: &PyTag, a: u64, b: u64, ta: &PyTag, tb: &PyTag, sub: bool) -> PyTag {
    PyTag(t::conj_pair(x.0, a, b, &ta.0, &tb.0, sub))
}
#[pyfunction]
fn from_signature(slots: i64, period: i64, live: i64, stride: i64, window: i64, offset: i64) -> PyTag {
    PyTag(t::from_signature(slots, period, live, stride, window, offset))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySupport>()?;
    m.add_class::<PyTag>()?;
    let sub = PyModule::new(m.py(), "tag")?;
    for (k, v) in [
        ("EMPTY", t::EMPTY),
        ("AP", t::AP),
        ("DENSE", t::DENSE),
        ("F_UNKNOWN", t::F_UNKNOWN),
        ("F_REAL", t::F_REAL),
        ("F_IMAG", t::F_IMAG),
        ("F_COMPLEX", t::F_COMPLEX),
    ] {
        sub.add(k, v)?;
    }
    sub.add_function(wrap_pyfunction!(top, &sub)?)?;
    sub.add_function(wrap_pyfunction!(with_field, &sub)?)?;
    sub.add_function(wrap_pyfunction!(t_mult_scalar, &sub)?)?;
    sub.add_function(wrap_pyfunction!(t_add_scalar, &sub)?)?;
    sub.add_function(wrap_pyfunction!(t_square, &sub)?)?;
    sub.add_function(wrap_pyfunction!(t_add, &sub)?)?;
    sub.add_function(wrap_pyfunction!(t_mult, &sub)?)?;
    sub.add_function(wrap_pyfunction!(t_mult_i, &sub)?)?;
    sub.add_function(wrap_pyfunction!(t_rotate, &sub)?)?;
    sub.add_function(wrap_pyfunction!(t_conjugate, &sub)?)?;
    sub.add_function(wrap_pyfunction!(t_bootstrap, &sub)?)?;
    sub.add_function(wrap_pyfunction!(t_reduce, &sub)?)?;
    sub.add_function(wrap_pyfunction!(conj_pair, &sub)?)?;
    sub.add_function(wrap_pyfunction!(from_signature, &sub)?)?;
    sub.add("Tag", m.py().get_type::<PyTag>())?;
    sub.add("Support", m.py().get_type::<PySupport>())?;
    m.add_submodule(&sub)?;
    Ok(())
}
