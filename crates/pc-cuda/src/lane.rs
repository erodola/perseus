//! Stream lanes: every ciphertext kernel runs on `Compute`; H2D/D2H run on `Copy` and hand
//! their results over as `Ready` tokens; `Side` is only for compiler-marked independent sub-programs. Lanes are
//! zero-sized types, so lane checks are static.
//!
//! `Copy` shadows the prelude's `Copy` trait when imported by name: refer to it as `lane::Copy`.

/// Marker for a stream lane.
pub trait Lane: 'static {
    const NAME: &'static str;
}

/// The ciphertext lane: every kernel that touches ciphertext data.
pub struct Compute;
/// The transfer lane: host <-> device copies (encode staging, KV pages, outputs).
pub struct Copy;
/// Independent sub-programs the compiler marks (only where measured to pay).
pub struct Side;

impl Lane for Compute {
    const NAME: &'static str = "compute";
}
impl Lane for Copy {
    const NAME: &'static str = "copy";
}
impl Lane for Side {
    const NAME: &'static str = "side";
}
