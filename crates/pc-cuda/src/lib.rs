//! pc-cuda: the CUDA runtime as perseus uses it — a hand-written cudart FFI (no cudarc: the kernels are nvcc
//! static archives with `extern "C"` launchers, so ~30 runtime entry points suffice) behind safe wrappers whose types
//! rule out whole race classes:
//!
//! * `Device::open` is the only way to get a `Device` (with the opt-in `PERSEUS_GPU_GUARD`, only under
//!   `tools/gpu.sh`'s lock). Streams, events and the arena all need a `&Device`.
//! * Streams are typed by a zero-sized lane (`Compute`, `Copy`, `Side`); a device buffer `DBuf<'a, S, T>` carries the
//!   lane whose stream orders every access to it, and moving it to another lane goes through an event
//!   (`DBuf::handoff`, `Ready::join`).
//! * Device memory is ONE `cudaMalloc` (`Arena`), carved at setup and frozen (`mem.rs`).
//! * Every fallible call returns `Result<_, Error>`; panics only on documented programming errors.
pub mod blas;
pub mod device;
pub mod error;
mod ffi;
pub mod lane;
pub mod mem;
pub mod stream;

pub use device::Device;
pub use error::{launch_result, Error, Result};
pub use lane::{Compute, Lane, Side};
pub use mem::{Arena, ConstBuf, DBuf, PinnedBuf, Pod, ARENA_ALIGN};
pub use stream::{Event, EventPool, GraphExec, OnLane, PoolEvent, Ready, Stream};

/// `cudaStream_t` as the kernel launchers take it.
pub type RawStream = *mut std::ffi::c_void;
