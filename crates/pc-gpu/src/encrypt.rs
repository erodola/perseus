//! Encryption on the device: secret-key (`c0 = -a s + e + m`, `c1 = a`, `encrypt` kernel with `a`
//! expanded in-kernel from the encryptor's seed, "ENCA" tag, one nonce per ciphertext) and public-key (`c0 = v pk0 +
//! e0 + m`, `c1 = v pk1 + e1` through the `tensor` kernel) encryption of coefficient-form message rows at any level.
//!
//! Messages are host-encoded rows (`pc_core::encode::Encoder::encode_slots`, coefficient form over Q ids
//! `0..=top`); errors (Gaussian sigma 3.19) and the public-key ternary `v` are sampled on the host from the caller's
//! `Prng`, added to the message rows on the host and NTT'd on the device. The device encoder hands
//! coefficient rows already on the device; the kernel path after the upload is the same.
use crate::context::{Context, DevStrategy, DevWord};
use crate::ct::Ct;
use crate::error::{shape, Result};
use crate::executor::Executor;
use crate::keys::DevSecret;
use crate::ntt::{self, Parts, Span};
use crate::poly::{CtBuf, Layout};
use pc_core::keys::{sample_ternary, DiscreteGaussian, Prng};
use pc_cuda::{launch_result, Arena, Compute, DBuf, Stream};
use pc_kernels::{EncryptArgs, PolyRef, TensorArgs};
use pc_shadow::CtMeta;
use rayon::prelude::*;

/// Public key `(pk0, pk1) = (-a s + e, a)` in eval form over every Q id (one context-layout slot).
pub struct PublicKey<'a, W: DevWord> {
    pub buf: CtBuf<'a, Compute, W>,
}

/// Seeded `a` source + error sampler; owns the device seed. Nonces are per ciphertext, never reused.
pub struct Encryptor<'a, W: DevWord> {
    seed: DBuf<'a, Compute, u32>,
    nonce: u32,
    gauss: DiscreteGaussian,
    pub pk: Option<PublicKey<'a, W>>,
}

/// Rows `0..=top` of each message, `[k][row][N]` residues of the coefficient-form encoding.
pub type MsgRows = [Vec<Vec<u64>>];

fn check_msgs(lay: &Layout, msgs: &MsgRows, k: usize, top: usize) -> Result<()> {
    if msgs.len() != k {
        return Err(shape("encrypt", format!("{} messages for a batch of {k} ciphertexts", msgs.len())));
    }
    if top >= lay.nq {
        return Err(shape("encrypt", format!("top limb {top} >= Q primes {}", lay.nq)));
    }
    for m in msgs {
        if m.len() < top + 1 || m.iter().take(top + 1).any(|r| r.len() != lay.n) {
            return Err(shape("encrypt", format!("message has {} rows (need {} of N = {})", m.len(), top + 1, lay.n)));
        }
    }
    Ok(())
}

impl<'a, W: DevWord> Encryptor<'a, W> {
    /// Arena bytes (reserve as "encrypt"); `public_key` adds one context-layout slot.
    pub fn bytes(lay: &Layout, public_key: bool) -> usize {
        Arena::bytes_for::<u32>(8) + if public_key { Arena::bytes_for::<W>(CtBuf::<Compute, W>::words(lay, 1)) } else { 0 }
    }

    /// Carve (setup only) and key the `a` expander from `prng`.
    pub fn new(arena: &'a Arena, s: &Stream<Compute>, lay: Layout, public_key: bool, prng: &mut Prng) -> Result<Self> {
        let mut seed = arena.alloc("encrypt.seed", 8)?;
        seed.upload(s, &prng.ksk_seed().0)?;
        let pk = if public_key { Some(PublicKey { buf: CtBuf::new(arena, "encrypt.pk", lay, 1)? }) } else { None };
        Ok(Encryptor { seed, nonce: 0, gauss: DiscreteGaussian::new(DiscreteGaussian::SIGMA)?, pk })
    }

    fn next_nonce(&mut self) -> Result<u32> {
        let n = self.nonce;
        self.nonce = n.checked_add(1).ok_or_else(|| shape("Encryptor", "nonce space exhausted (2^32 ciphertexts); build a new Encryptor"))?;
        Ok(n)
    }

    /// Host rows `[row][N]` (rows `0..nrows`) of `m + e` (e fresh Gaussian), or of `e` alone when `m` is None.
    fn noisy_rows(&self, primes: &[u64], n: usize, nrows: usize, m: Option<&[Vec<u64>]>, prng: &mut Prng) -> Vec<W> {
        let e = self.gauss.sample_vec(prng, n);
        let mut out = vec![W::default(); nrows * n];
        out.par_chunks_mut(n).enumerate().for_each(|(r, o)| {
            let p = primes[r];
            for (i, x) in o.iter_mut().enumerate() {
                let ev = e[i].rem_euclid(p as i64) as u64;
                let v = match m {
                    Some(m) => (m[r][i] % p + ev) % p,
                    None => ev,
                };
                *x = W::from_u64(v);
            }
        });
        out
    }

    /// Secret-key encryption of `ct.k()` messages at the level of `meta` (`top = ctx.top_of(meta.level)`); `ct` gets
    /// `meta` (set `nf` to the encoding scale and `slots`). Rows above `top` are left untouched.
    pub fn encrypt_sk<M: DevStrategy<W>>(
        &mut self,
        ctx: &Context<'_, W, M>,
        s: &Stream<Compute>,
        secret: &DevSecret<'_, W>,
        msgs: &MsgRows,
        meta: CtMeta,
        ct: &mut Ct<'_, W>,
        prng: &mut Prng,
    ) -> Result<()> {
        let top = ctx.top_of(meta.level)?;
        let k = ct.k();
        check_msgs(&ctx.lay, msgs, k, top)?;
        let primes = ctx.chain.primes();
        let rows: Vec<usize> = (0..=top).collect();
        for (c, m) in msgs.iter().enumerate() {
            let h = self.noisy_rows(&primes, ctx.lay.n, top + 1, Some(m), prng);
            ct.buf.upload_rows(s, c, 0, &rows, &h)?;
        }
        ntt::forward(ctx, s, &mut ct.buf, Span::Q { top }, Parts::C0)?;
        let base = ct.buf.w();
        for c in 0..k {
            let c0 = (base.base as *mut W).wrapping_add(c * ctx.lay.ct_words());
            let a = EncryptArgs {
                c0: c0 as *mut _,
                c1: c0.wrapping_add(ctx.lay.poly_words()) as *mut _,
                s: secret.buf.buf.vptr(),
                e: c0 as *const _,
                m: std::ptr::null(),
                seed: self.seed.dptr(),
                pr: ctx.pr(),
                nonce: self.next_nonce()?,
                nl: (top + 1) as i32,
            };
            // SAFETY: c0/c1 rows 0..=top of ciphertext c (exclusive &mut ct); e aliases c0 element-wise (each thread
            // reads its four words before writing them); secret rows by prime id
            launch_result("encrypt", unsafe { (ctx.table.encrypt)(&a, s.raw()) })?;
        }
        ct.meta = meta;
        Ok(())
    }

    /// Generate the public key `(-a s + e, a)` over every Q id (needs `public_key` at `new`). Synced.
    pub fn gen_public_key<M: DevStrategy<W>>(&mut self, ctx: &Context<'_, W, M>, s: &Stream<Compute>, secret: &DevSecret<'_, W>, prng: &mut Prng) -> Result<()> {
        let nonce = self.next_nonce()?;
        let primes = ctx.chain.primes();
        let nq = ctx.lay.nq;
        let h = self.noisy_rows(&primes, ctx.lay.n, nq, None, prng);
        let pk = self.pk.as_mut().ok_or_else(|| shape("Encryptor::gen_public_key", "built without public_key"))?;
        let rows: Vec<usize> = (0..nq).collect();
        pk.buf.upload_rows(s, 0, 0, &rows, &h)?;
        ntt::forward(ctx, s, &mut pk.buf, Span::Q { top: nq - 1 }, Parts::C0)?;
        let c0 = pk.buf.w().base as *mut W;
        let a = EncryptArgs {
            c0: c0 as *mut _,
            c1: c0.wrapping_add(ctx.lay.poly_words()) as *mut _,
            s: secret.buf.buf.vptr(),
            e: c0 as *const _,
            m: std::ptr::null(),
            seed: self.seed.dptr(),
            pr: ctx.pr(),
            nonce,
            nl: nq as i32,
        };
        // SAFETY: as encrypt_sk, on the public key's own slot
        launch_result("encrypt", unsafe { (ctx.table.encrypt)(&a, s.raw()) })?;
        s.sync()?;
        Ok(())
    }

    /// Public-key encryption of `ct.k()` messages at `meta`'s level: `c0 = v pk0 + e0 + m`, `c1 = v pk1 + e1`, `v`
    /// uniform ternary. Uses `ex.scratch` (k ciphertexts) for `v`.
    pub fn encrypt_pk<M: DevStrategy<W>>(
        &mut self,
        ctx: &Context<'_, W, M>,
        ex: &mut Executor<'_, W>,
        msgs: &MsgRows,
        meta: CtMeta,
        ct: &mut Ct<'_, W>,
        prng: &mut Prng,
    ) -> Result<()> {
        let top = ctx.top_of(meta.level)?;
        let k = ct.k();
        check_msgs(&ctx.lay, msgs, k, top)?;
        if ex.scratch.k < k {
            return Err(shape("encrypt_pk", format!("executor scratch holds {} ciphertexts, batch is {k}", ex.scratch.k)));
        }
        let primes = ctx.chain.primes();
        let n = ctx.lay.n;
        let rows: Vec<usize> = (0..=top).collect();
        let s = &ex.compute;
        let zero = vec![W::default(); (top + 1) * n];
        for (c, m) in msgs.iter().enumerate() {
            let h0 = self.noisy_rows(&primes, n, top + 1, Some(m), prng);
            let h1 = self.noisy_rows(&primes, n, top + 1, None, prng);
            ct.buf.upload_rows(s, c, 0, &rows, &h0)?;
            ct.buf.upload_rows(s, c, 1, &rows, &h1)?;
            let v: Vec<i64> = sample_ternary(prng, n).into_iter().map(|x| x as i64).collect();
            let vr = crate::keys::signed_rows::<W>(&v, &primes[..=top]);
            ex.scratch.upload_rows(s, c, 0, &rows, &vr)?;
            ex.scratch.upload_rows(s, c, 1, &rows, &zero)?;
        }
        let pk = self.pk.as_ref().ok_or_else(|| shape("Encryptor::encrypt_pk", "no public key (gen_public_key)"))?;
        ntt::forward(ctx, s, &mut ct.buf, Span::Q { top }, Parts::Both)?;
        // only the first k scratch cts hold `v` (the scratch is scratch_cts x k_max cts; the NTT takes the buffer's k)
        let k_scratch = ex.scratch.k;
        ex.scratch.k = k;
        let r = ntt::forward(ctx, s, &mut ex.scratch, Span::Q { top }, Parts::C0);
        ex.scratch.k = k_scratch;
        r?;
        let pkr = PolyRef { ct_stride: 0, ..pk.buf.r() };
        let a = TensorArgs {
            out: ct.buf.w(),
            d2: ex.scratch.w(),
            a0: pkr,
            b0: ex.scratch.r(),
            pr: ctx.pr(),
            npairs: 1,
            acc: 1,
            n: (top + 1) as i32,
            k: k as i32,
            ..Default::default()
        };
        // SAFETY: out = ct rows 0..=top (exclusive), d2 = scratch part 1 (zeros, rewritten with a * 0 = 0), pk shared
        launch_result("tensor", unsafe { (ctx.table.tensor)(&a, s.raw()) })?;
        ct.meta = meta;
        Ok(())
    }
}
