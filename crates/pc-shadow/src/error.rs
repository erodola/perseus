//! Typed errors; every variant names the op, the value and the rule that refused it.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A level/degree combination the rule cannot meet (e.g. a plaintext deeper than its ciphertext).
    #[error("rule {op}: {msg}")]
    Rule { op: &'static str, msg: String },
    /// A conversion out of the chain's range (level past the bottom, negative consumed depth below level 0, ...).
    #[error("conversion {what}: {msg}")]
    Range { what: &'static str, msg: String },
    /// A route or bootstrap configuration value that is missing or inconsistent.
    #[error("bootstrap config {knob}: {msg}")]
    Bts { knob: &'static str, msg: String },
    /// graph.json content the loader refuses (cycle, bad pin, ...).
    #[error("graph: {0}")]
    Graph(String),
    #[error(transparent)]
    Core(#[from] pc_core::error::Error),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}
pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn rule(op: &'static str, msg: impl Into<String>) -> Error {
    Error::Rule { op, msg: msg.into() }
}
