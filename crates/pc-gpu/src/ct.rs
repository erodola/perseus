//! A ciphertext batch on the compute lane with its bookkeeping.
use crate::poly::CtBuf;
use pc_cuda::Compute;
use pc_shadow::CtMeta;

/// `buf.k` ciphertexts that share one `CtMeta` (same level, degree, scale, slots, tag): the unit a launch works on
/// (K ciphertexts per launch is how concurrency inside a block is expressed). Eager code uses `k = 1`.
///
/// `meta` is the single source of truth for which rows are live: Q rows `0..=top` with `top = |Q| - 1 - level`
/// (`pc_shadow::meta::top_limb`); ops execute the `pc_shadow::Recipe` for the op and set `meta = recipe.out`.
pub struct Ct<'a, W: pc_cuda::Pod> {
    pub meta: CtMeta,
    pub buf: CtBuf<'a, Compute, W>,
}

impl<'a, W: pc_cuda::Pod> Ct<'a, W> {
    pub fn new(meta: CtMeta, buf: CtBuf<'a, Compute, W>) -> Self {
        Ct { meta, buf }
    }
    pub fn k(&self) -> usize {
        self.buf.k
    }
    /// FIDESlib top limb index of the live Q rows.
    pub fn top(&self, shape: pc_shadow::ChainShape) -> crate::Result<usize> {
        Ok(pc_shadow::meta::top_limb(shape, self.meta.level)? as usize)
    }
}
