//! One error type for the client; variants name the offending field or value.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("parameter error: {0}")]
    Params(String),
    #[error("key spec: {0}")]
    Spec(String),
    #[error("wire format: {0}")]
    Wire(String),
    #[error("{0}")]
    Core(#[from] pc_core::error::Error),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
