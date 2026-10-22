//! Typed compile errors; each names the block, the var or the knob it refuses.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// graph.json content the compiler cannot lower (unknown op kind, malformed literal, missing input).
    #[error("graph {block}: {msg}")]
    Graph { block: String, msg: String },
    /// placement.json / PLAN_CMD.txt / pcmm_calib.json content that is missing or inconsistent.
    #[error("plan {path}: {msg}")]
    Plan { path: String, msg: String },
    /// A Config / PlanConfig value the compiler needs and cannot use.
    #[error("config {knob}: {msg}")]
    Config { knob: &'static str, msg: String },
    /// The compile-time level/degree check failed (the perseus runtime's \[plan_level_error\] / \[plan_deg_error\]).
    #[error("check {block}: {n} error(s), first: {first}")]
    Check { block: String, n: usize, first: String },
    #[error(transparent)]
    Shadow(#[from] pc_shadow::Error),
    #[error(transparent)]
    PcConfig(#[from] pc_config::Error),
    #[error("io {path}: {err}")]
    Io { path: String, err: std::io::Error },
    #[error("json {path}: {err}")]
    Json { path: String, err: serde_json::Error },
    /// Binding a program to the device runtime failed (missing plaintexts / keys / routes, slot aliasing).
    #[error("bind {block}: {msg}")]
    Bind { block: String, msg: String },
    /// Replay refused (state the replay cannot run from).
    #[error("replay {block}: {msg}")]
    Replay { block: String, msg: String },
    #[cfg(feature = "gpu")]
    #[error(transparent)]
    Gpu(#[from] pc_gpu::Error),
    #[cfg(feature = "gpu")]
    #[error(transparent)]
    Cuda(#[from] pc_cuda::Error),
}
pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn read(path: &std::path::Path) -> Result<Vec<u8>> {
    std::fs::read(path).map_err(|err| Error::Io { path: path.display().to_string(), err })
}

pub(crate) fn json<T: serde::de::DeserializeOwned>(path: &std::path::Path) -> Result<T> {
    serde_json::from_slice(&read(path)?).map_err(|err| Error::Json { path: path.display().to_string(), err })
}
