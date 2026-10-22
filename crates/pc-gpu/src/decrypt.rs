//! Decryption: `m = c0 + c1 s` on the device over the bottom composite level only (the `level_unit`
//! lowest Q rows: every live ciphertext has them, and the decoded value needs no more), INTT, D2H of those rows, exact
//! CRT + FFT decode on the host (`pc_core::encode::Encoder::decode_slots`).
//!
//! The product runs through the `tensor` kernel with the key pair `(s, 1)`: for a ciphertext `(c0, c1)` and key
//! `(x0, x1) = (s, 1)` the tensor's middle term is `x0 c1 + x1 c0 = c0 + c1 s` (part 1 of the output); the
//! squared term goes to part 0 of the same output (`d2` aliases it), which is discarded.
use crate::context::{Context, DevStrategy, DevWord};
use crate::ct::Ct;
use crate::error::{shape, Result};
use crate::keys::DevSecret;
use crate::ntt::{self, Dir, RowList};
use crate::poly::Layout;
use pc_core::encode::{Complex, Encoder, Payload};
use pc_cuda::{launch_result, Arena, Compute, DBuf, Stream};
use pc_kernels::{PolyRef, TensorArgs};

/// Which secret decrypts (the sparse one only for encapsulated ciphertexts in tests / checks).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Which {
    Dense = 0,
    Sparse = 1,
}

/// Bottom-level decryption buffers: the key pairs `(s, 1)` / `(s_sparse, 1)` on the bottom rows and an output of
/// `k_max` ciphertexts `[c][part][row < d][N]`.
pub struct Decryptor<'a, W: DevWord> {
    /// `[which][part][d][N]`
    key: DBuf<'a, Compute, W>,
    out: DBuf<'a, Compute, W>,
    /// INTT rows of the outputs' part 1 for batches of 1..=k_max.
    lists: Vec<RowList<'a>>,
    d: usize,
    n: usize,
    k_max: usize,
    loaded: [bool; 2],
}

impl<'a, W: DevWord> Decryptor<'a, W> {
    /// Arena bytes (reserve as "decrypt").
    pub fn bytes(lay: &Layout, d: usize, k_max: usize) -> usize {
        let lists: usize = (1..=k_max).map(|k| Arena::bytes_for::<u32>(k * d) + Arena::bytes_for::<i32>(k * d)).sum();
        Arena::bytes_for::<W>(4 * d * lay.n) + Arena::bytes_for::<W>(k_max * 2 * d * lay.n) + lists
    }

    /// Carve (setup only). `d` = the chain's `level_unit`.
    pub fn new(arena: &'a Arena, s: &Stream<Compute>, lay: Layout, d: usize, k_max: usize) -> Result<Self> {
        let n = lay.n;
        let mut lists = Vec::with_capacity(k_max);
        for k in 1..=k_max {
            let rows: Vec<(u32, i32)> = (0..k).flat_map(|c| (0..d).map(move |r| ((c * 2 * d + d + r) as u32, r as i32))).collect();
            lists.push(RowList::new(arena, s, &rows)?);
        }
        Ok(Decryptor {
            key: arena.alloc("decrypt.key", 4 * d * n)?,
            out: arena.alloc("decrypt.out", k_max * 2 * d * n)?,
            lists,
            d,
            n,
            k_max,
            loaded: [false; 2],
        })
    }

    /// Copy the bottom rows of the device secret (dense, and sparse when present) into the key pairs (synced).
    pub fn load_secret(&mut self, s: &Stream<Compute>, secret: &DevSecret<'_, W>) -> Result<()> {
        let (d, n) = (self.d, self.n);
        let rows: Vec<usize> = (0..d).collect();
        let ones = vec![W::from_u64(1); d * n];
        let mut host = Vec::with_capacity(4 * d * n);
        for which in 0..2 {
            host.extend(secret.buf.download_rows(s, 0, which, &rows)?);
            host.extend_from_slice(&ones);
        }
        self.key.upload(s, &host)?;
        self.loaded = [true, secret.has_sparse];
        Ok(())
    }

    /// Coefficient rows `[k][d rows][N]` (u64) of `c0 + c1 s` for every ciphertext of `ct` (syncs `s`).
    pub fn decrypt_rows<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, ct: &Ct<'_, W>, which: Which) -> Result<Vec<Vec<Vec<u64>>>> {
        let (d, n, k) = (self.d, self.n, ct.k());
        if !self.loaded[which as usize] {
            return Err(shape("Decryptor", format!("{which:?} secret not loaded (load_secret; the sparse part needs DevSecret::has_sparse)")));
        }
        if k > self.k_max {
            return Err(shape("Decryptor", format!("batch of {k} ciphertexts > k_max {}", self.k_max)));
        }
        if ct.top(ctx.shape)? + 1 < d {
            return Err(shape("Decryptor", format!("ciphertext top limb {} is below the bottom composite level", ct.top(ctx.shape)?)));
        }
        let ob = self.out.vptr_mut();
        let out = PolyRef { base: ob, ct_stride: (2 * d * n) as u64, part_stride: (d * n) as u64 };
        let a = TensorArgs {
            out,
            // part 1 of d2 = part 0 of out: the discarded c1 * 1 term lands where the s c0 term was
            d2: PolyRef { part_stride: 0, ..out },
            a0: PolyRef {
                base: (self.key.vptr() as *const W).wrapping_add(which as usize * 2 * d * n) as *mut _,
                ct_stride: 0,
                part_stride: (d * n) as u64,
            },
            b0: ct.buf.r(),
            pr: ctx.pr(),
            npairs: 1,
            acc: 0,
            n: d as i32,
            k: k as i32,
            ..Default::default()
        };
        // SAFETY: out holds k_max >= k compact ciphertexts of d rows; the key pair and the ciphertext rows 0..d are
        // read only; row index == prime id for every operand (rows < d < nq)
        launch_result("tensor", unsafe { (ctx.table.tensor)(&a, s.raw()) })?;
        ntt::transform_rows(ctx, s, Dir::Inverse, &mut self.out, &self.lists[k - 1])?;
        let mut res = Vec::with_capacity(k);
        for c in 0..k {
            let v = self.out.download_range(s, c * 2 * d * n + d * n, d * n)?;
            res.push(v.chunks(n).map(|r| r.iter().map(|w| w.to_u64()).collect()).collect());
        }
        Ok(res)
    }

    /// Decoded slots of every ciphertext of `ct` (`meta.slots` slots, 0 = N/2; scale `meta.nf`).
    pub fn decrypt_slots<M: DevStrategy<W>>(
        &mut self,
        ctx: &Context<'_, W, M>,
        s: &Stream<Compute>,
        enc: &Encoder,
        ct: &Ct<'_, W>,
        payload: Payload,
        which: Which,
    ) -> Result<Vec<Vec<Complex>>> {
        if !(ct.meta.nf > 0.0) {
            return Err(shape("decrypt_slots", format!("ciphertext scale (meta.nf) = {}", ct.meta.nf)));
        }
        let slots = if ct.meta.slots == 0 { ctx.lay.n / 2 } else { ct.meta.slots as usize };
        let rows = self.decrypt_rows(ctx, s, ct, which)?;
        rows.iter().map(|r| Ok(enc.decode_slots(r, slots, payload, ct.meta.nf)?)).collect()
    }
}
