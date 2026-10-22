//! pc-shadow: the single level / degree / scale / pack-tag engine of perseus (CUDA-free).
//!
//! The planner simulation, the numpy fake, the bootstrap setup and the GPU runtime all ask the same `Shadow` what
//! an op does to a ciphertext's bookkeeping (`rules`), where a refresh lands (`route`), how the pack tag propagates
//! (`tag`) and what the EvalMod program is (`evalmod`). Deviations of the Python planner/fakes from the C++ truth are
//! named `quirks`, switched per configuration.
pub mod error;
pub mod evalmod;
pub mod fides;
pub mod graph;
pub mod meta;
pub mod quirks;
pub mod route;
pub mod rules;
pub mod sim;
pub mod tag;

pub use error::{Error, Result};
pub use meta::{ChainShape, CtMeta};
pub use quirks::{Quirk, Quirks};
pub use route::{BtsConfig, Finish, Landing, Route};
pub use rules::{Consts, Operand, Prelude, PtMeta, Recipe, Shadow, ShadowConfig};
