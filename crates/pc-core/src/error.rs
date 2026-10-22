//! One error type for pc-core; variants name the offending knob or value so setup failures are actionable.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("parameter error: {0}")]
    Params(String),
    #[error("encoding error: {0}")]
    Encode(String),
    #[error("key error: {0}")]
    Keys(String),
    #[error("bootstrap setup error: {0}")]
    BtsSetup(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}
pub type Result<T> = std::result::Result<T, Error>;
