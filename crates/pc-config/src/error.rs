//! Typed errors naming the knob and the value.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("config {knob}={value:?}: expected {want}")]
    Value { knob: &'static str, value: String, want: String },
    #[error("config: no field {0:?}")]
    NoField(String),
    #[error("config: unknown preset {0:?} (known: {1})")]
    NoPreset(String, String),
    #[error("config json: {0}")]
    Json(#[from] serde_json::Error),
}
pub type Result<T> = std::result::Result<T, Error>;
