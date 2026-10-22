//! Typed errors of the device runtime. Setup failures name the knob to change.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Cuda(#[from] pc_cuda::Error),
    #[error(transparent)]
    Kernels(#[from] pc_kernels::TableError),
    #[error(transparent)]
    Core(#[from] pc_core::error::Error),
    #[error(transparent)]
    Shadow(#[from] pc_shadow::Error),
    /// Memory planning: `knob` is the MemCfg field / MemPlan section to raise.
    #[error("memory: {knob} = {value}: {msg}")]
    Mem { knob: &'static str, value: usize, msg: String },
    /// An argument whose shape does not fit the context (layout, rows, k, level).
    #[error("{what}: {msg}")]
    Shape { what: &'static str, msg: String },
}
pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn shape(what: &'static str, msg: impl Into<String>) -> Error {
    Error::Shape { what, msg: msg.into() }
}
