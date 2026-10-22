//! Bootstrap finish (FIDESlib `btsFinish` / `btsStcFirstOutput`): the integer post-factor (`multIntScalar` of
//! corFactor = 2^correction, StC-first: corFactor << stc_first_deg), applied per prime as `x * (2^e mod p)`; the
//! real-payload conjugate-add and the sparse rotate-add live with their routes.
use crate::context::{Context, DevStrategy, DevWord};
use crate::error::{shape, Result};
use crate::ks::View;
use pc_core::modmul::shoup;
use pc_cuda::{launch_result, Arena, Compute, ConstBuf, Lane, Stream};
use pc_kernels::EwArgs;

/// Largest power-of-two post-factor (2^63 still fits FIDESlib's u64 `op`).
pub const POW2_MAX: u32 = 63;

/// `2^e mod p` by (e, prime id) with Shoup companions: every integer post-factor a call can ask for, address-stable
/// (graph capture) and built at setup.
pub struct Pow2<'a, W: DevWord> {
    v: ConstBuf<'a, W>,
    s: ConstBuf<'a, W>,
    rows: usize,
}

impl<'a, W: DevWord> Pow2<'a, W> {
    pub fn bytes(rows: usize) -> usize {
        2 * Arena::bytes_for::<W>((POW2_MAX as usize + 1) * rows)
    }
    pub fn new<S: Lane, M: DevStrategy<W>>(arena: &'a Arena, s: &Stream<S>, ctx: &Context<'_, W, M>) -> Result<Self> {
        let primes = ctx.chain.primes();
        let rows = primes.len();
        let (mut v, mut sh) = (Vec::new(), Vec::new());
        for e in 0..=POW2_MAX {
            for &p in &primes {
                let c = shoup(W::from_u64((1u64 << e) % p), W::from_u64(p));
                v.push(c.w);
                sh.push(c.w_prime);
            }
        }
        Ok(Pow2 { v: ConstBuf::new(arena, "bts.pow2", s, &v)?, s: ConstBuf::new(arena, "bts.pow2", s, &sh)?, rows })
    }
    /// `x *= 2^e` on Q rows 0..=top of k cts (FIDESlib `multIntScalar(ct, 2^e)`; metadata unchanged).
    pub fn mult<M: DevStrategy<W>>(&self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, x: View, k: usize, top: usize, e: u32) -> Result<()> {
        if e > POW2_MAX {
            return Err(shape("bts post-factor", format!("2^{e} above 2^{POW2_MAX}")));
        }
        if e == 0 {
            return Ok(());
        }
        let off = e as usize * self.rows;
        let a = EwArgs {
            out: x.r(),
            a: x.r(),
            b: x.r(),
            rs: ctx.lay.rs_q(top),
            sc: self.v.dptr_at(off)? as *const std::ffi::c_void,
            scs: self.s.dptr_at(off)? as *const std::ffi::c_void,
            pr: ctx.pr(),
            k: k as i32,
            nparts: 2,
            sub: 0,
            zero_p: 0,
        };
        // SAFETY: k-ct view of a live context-layout buffer; the table row covers every prime id
        launch_result("ew_scale", unsafe { (ctx.table.ew_scale)(&a, s.raw()) })?;
        Ok(())
    }
}

/// log2 of a power-of-two post-factor.
pub fn pow2_exp(v: u64) -> Result<u32> {
    if v == 0 || !v.is_power_of_two() {
        return Err(shape("bts post-factor", format!("{v} is not a power of two")));
    }
    Ok(v.trailing_zeros())
}
