//! `Config` / `PlanConfig` (pc-config) for Python: presets, JSON, the legacy env bridge, dotted field access.
//! Values cross as JSON (the `json` module), so a field reads back as bool / int / float / str / list / None.
use crate::config_err;
use pc_config::{presets, Config, FieldInfo, PlanConfig};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::BTreeMap;

fn to_py(py: Python<'_>, v: &serde_json::Value) -> PyResult<Py<PyAny>> {
    Ok(py.import("json")?.call_method1("loads", (v.to_string(),))?.unbind())
}

fn from_py(v: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    let s: String = v.py().import("json")?.call_method1("dumps", (v,))?.extract()?;
    serde_json::from_str(&s).map_err(|e| crate::ConfigError::new_err(e.to_string()))
}

/// A Python mapping name -> str | None (None = unset; other values are str()-ed, as os.environ would).
fn env_items(d: &Bound<'_, PyAny>) -> PyResult<Vec<(String, Option<String>)>> {
    let mut out = vec![];
    for item in d.call_method0("items")?.try_iter()? {
        let (k, v): (String, Bound<'_, PyAny>) = item?.extract()?;
        let v = if v.is_none() { None } else { Some(v.str()?.to_string()) };
        out.push((k, v));
    }
    Ok(out)
}

fn fields_py<'py>(py: Python<'py>, fs: &[&'static FieldInfo]) -> PyResult<Bound<'py, PyList>> {
    let l = PyList::empty(py);
    for f in fs {
        let d = PyDict::new(py);
        d.set_item("name", f.dotted())?;
        d.set_item("env", f.env)?;
        d.set_item("kind", f.kind.name())?;
        d.set_item("default", f.default)?;
        d.set_item("doc", f.doc)?;
        d.set_item("contract", f.in_contract())?;
        l.append(d)?;
    }
    Ok(l)
}

#[pyclass(name = "Config", module = "perseus._pc")]
#[derive(Clone)]
pub struct PyConfig(pub Config);

#[pymethods]
impl PyConfig {
    #[new]
    fn new() -> Self {
        PyConfig(Config::default())
    }
    #[staticmethod]
    fn preset(name: &str) -> PyResult<Self> {
        Config::preset(name).map(PyConfig).map_err(config_err)
    }
    #[staticmethod]
    fn presets() -> Vec<&'static str> {
        presets::NAMES.to_vec()
    }
    #[staticmethod]
    fn from_json(s: &str) -> PyResult<Self> {
        Config::from_json(s).map(PyConfig).map_err(config_err)
    }
    fn to_json(&self) -> PyResult<String> {
        self.0.to_json().map_err(config_err)
    }
    /// From an env mapping (os.environ, a capture_env dict, a parsed `runtime=` line). Names no Config field owns
    /// are ignored (see `split_legacy_env`); a malformed value raises ConfigError naming the knob.
    #[staticmethod]
    fn from_legacy_env(env: &Bound<'_, PyAny>) -> PyResult<Self> {
        let items = env_items(env)?;
        let (c, _) = Config::from_legacy_env(items.iter().map(|(k, v)| (k.as_str(), v.as_deref()))).map_err(config_err)?;
        Ok(PyConfig(c))
    }
    /// (Config, PlanConfig, [names neither owns]).
    #[staticmethod]
    fn split_legacy_env(env: &Bound<'_, PyAny>) -> PyResult<(Self, PyPlanConfig, Vec<String>)> {
        let items = env_items(env)?;
        let map: BTreeMap<&str, Option<&str>> = items.iter().map(|(k, v)| (k.as_str(), v.as_deref())).collect();
        let (c, rest) = Config::from_legacy_env(map.iter().map(|(k, v)| (*k, *v))).map_err(config_err)?;
        let (p, rest) = PlanConfig::from_legacy_env(rest.iter().map(|k| (k.as_str(), map[k.as_str()]))).map_err(config_err)?;
        Ok((PyConfig(c), PyPlanConfig(p), rest))
    }
    /// Every set field under its legacy env name and spelling.
    fn to_legacy_env(&self) -> BTreeMap<String, String> {
        self.0.to_legacy_env()
    }
    /// The plan-bound subset (chain / bts / model / quirks + the legacy contract's names).
    fn load_bearing(&self) -> Self {
        PyConfig(self.0.load_bearing())
    }
    /// `self` with every field `other` sets replaced.
    fn overlay(&self, other: &PyConfig) -> Self {
        PyConfig(self.0.overlay(&other.0))
    }
    /// With legacy env values layered on top (`{"LN_CHEB": "0"}`, a `--set` list).
    fn with_env(&self, env: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut c = self.0.clone();
        for (k, v) in env_items(env)? {
            if !c.set_env(&k, v.as_deref()).map_err(config_err)? {
                return Err(crate::ConfigError::new_err(format!("config: no field for env name {k:?}")));
            }
        }
        Ok(PyConfig(c))
    }
    /// A field by dotted name (`model.ln_cheb`) or by legacy env name (`LN_CHEB`); None when unset.
    fn get(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        if !name.contains('.') {
            return match self.0.get_env(name) {
                Some(_) => {
                    let f = Config::fields().into_iter().find(|f| f.env == name).map(|f| f.dotted());
                    to_py(py, &self.0.get(&f.unwrap_or_default()).map_err(config_err)?)
                }
                None => Err(crate::ConfigError::new_err(format!("config: no field for env name {name:?}"))),
            };
        }
        to_py(py, &self.0.get(name).map_err(config_err)?)
    }
    fn set(&mut self, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let dotted = if name.contains('.') {
            name.to_string()
        } else {
            Config::fields()
                .into_iter()
                .find(|f| f.env == name)
                .map(|f| f.dotted())
                .ok_or_else(|| crate::ConfigError::new_err(format!("config: no field for env name {name:?}")))?
        };
        self.0.set(&dotted, from_py(value)?).map_err(config_err)
    }
    fn __getitem__(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        self.get(py, name)
    }
    fn __setitem__(&mut self, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.set(name, value)
    }
    /// The legacy env value of one name (None = unset), as the legacy readers see it.
    fn env(&self, name: &str) -> PyResult<Option<String>> {
        self.0
            .get_env(name)
            .ok_or_else(|| crate::ConfigError::new_err(format!("config: no field for env name {name:?}")))
    }
    #[staticmethod]
    fn fields(py: Python<'_>) -> PyResult<Bound<'_, PyList>> {
        fields_py(py, &Config::fields())
    }
    fn copy(&self) -> Self {
        self.clone()
    }
    fn __eq__(&self, o: &Self) -> bool {
        self.0 == o.0
    }
    fn __repr__(&self) -> String {
        format!("Config({} fields set)", self.0.to_legacy_env().len())
    }
}

#[pyclass(name = "PlanConfig", module = "perseus._pc")]
#[derive(Clone)]
pub struct PyPlanConfig(pub PlanConfig);

#[pymethods]
impl PyPlanConfig {
    #[new]
    fn new() -> Self {
        PyPlanConfig(PlanConfig::default())
    }
    #[staticmethod]
    fn preset(name: &str) -> PyResult<Self> {
        PlanConfig::preset(name).map(PyPlanConfig).map_err(config_err)
    }
    #[staticmethod]
    fn from_legacy_env(env: &Bound<'_, PyAny>) -> PyResult<Self> {
        let items = env_items(env)?;
        let (c, _) =
            PlanConfig::from_legacy_env(items.iter().map(|(k, v)| (k.as_str(), v.as_deref()))).map_err(config_err)?;
        Ok(PyPlanConfig(c))
    }
    fn to_legacy_env(&self) -> BTreeMap<String, String> {
        self.0.to_legacy_env()
    }
    #[staticmethod]
    fn from_json(s: &str) -> PyResult<Self> {
        serde_json::from_str(s).map(PyPlanConfig).map_err(|e| crate::ConfigError::new_err(e.to_string()))
    }
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string_pretty(&self.0).map_err(|e| crate::ConfigError::new_err(e.to_string()))
    }
    fn get(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        let n = name.strip_prefix("plan.").unwrap_or(name);
        match self.0.get_json(n) {
            Some(v) => to_py(py, &v),
            None => Err(crate::ConfigError::new_err(format!("config: no field plan.{n}"))),
        }
    }
    #[staticmethod]
    fn fields(py: Python<'_>) -> PyResult<Bound<'_, PyList>> {
        fields_py(py, &PlanConfig::fields())
    }
    fn __eq__(&self, o: &Self) -> bool {
        self.0 == o.0
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyConfig>()?;
    m.add_class::<PyPlanConfig>()?;
    Ok(())
}
