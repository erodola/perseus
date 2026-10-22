//! pc-kernels: the CUDA kernels of perseus as C++ templates over (Word, ModMul), one instantiation per
//! translation unit, each exporting a `KernelTable` of `extern "C"` launchers (see `kernels/args.cuh`).
//! Rust side: `args` mirrors the ABI structs with size asserts, `table` resolves an instantiation's table.
pub mod args;
pub mod ring;
pub mod table;

pub use args::*;
pub use ring::{ring_table, RingTable, RingWord};
pub use table::{table, Instance, TableError};
