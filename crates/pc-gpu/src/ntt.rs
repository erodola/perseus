//! Batched NTT over ciphertext rows (seam `Ntt<W, M>`: the 4-step register-pass kernel, one launch per row
//! chunk of 128 rows inside the launcher). Rows are addressed by tables `(roff, rprime)` in device memory (limb
//! offset in units of N from the buffer base, prime id), built at setup: `RowTables` for the spans every op needs,
//! `RowList` for anything else (key-switching digit sets).
use crate::context::{Context, DevStrategy, DevWord};
use crate::error::{shape, Result};
use crate::poly::{CtBuf, Layout};
use pc_cuda::{launch_result, Arena, Compute, ConstBuf, DBuf, Lane, Stream};
use pc_kernels::NttArgs;
use std::ffi::c_void;

/// Rows per launch chunk (both passes of a chunk run back to back, so the intermediate stays in L2).
pub const NTT_CHUNK: i32 = 128;

/// The live rows of every ciphertext in a span.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Span {
    /// Q rows `0..=top`.
    Q { top: usize },
    /// Q rows `0..=top` and every special row.
    QP { top: usize },
    /// Special rows only.
    P,
}

/// Which polynomials of each ciphertext.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Parts {
    Both,
    C0,
    C1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dir {
    Forward,
    Inverse,
}

/// Row tables for every `Span` of up to `k_max` ciphertexts of one layout. Per span two sections: `[part][c <
/// k_max][live row]` (the rows of one part of the first `k` ciphertexts are a contiguous prefix of that part's half)
/// and `[c < k_max][part][live row]` (both parts of the first `k` ciphertexts are a prefix: one launch for any k).
pub struct RowTables<'a> {
    roff: ConstBuf<'a, u32>,
    rprime: ConstBuf<'a, i32>,
    /// (start entry, live rows per ciphertext) by span key (`Q{top}` -> top, `QP{top}` -> nq + top, `P` -> 2 nq)
    index: Vec<(usize, usize)>,
    lay: Layout,
    k_max: usize,
}

impl<'a> RowTables<'a> {
    fn spans(lay: &Layout) -> impl Iterator<Item = Span> + '_ {
        (0..lay.nq).map(|top| Span::Q { top }).chain((0..lay.nq).map(|top| Span::QP { top })).chain([Span::P])
    }
    fn ids(lay: &Layout, span: Span) -> Vec<usize> {
        match span {
            Span::Q { top } => (0..=top).collect(),
            Span::QP { top } => (0..=top).chain(lay.nq..lay.rows()).collect(),
            Span::P => (lay.nq..lay.rows()).collect(),
        }
    }
    fn key(&self, span: Span) -> Result<usize> {
        let nq = self.lay.nq;
        match span {
            Span::Q { top } | Span::QP { top } if top >= nq => Err(shape("ntt span", format!("top {top} >= |Q| = {nq}"))),
            Span::Q { top } => Ok(top),
            Span::QP { top } => Ok(nq + top),
            Span::P => Ok(2 * nq),
        }
    }
    /// Table entries for `lay` and `k_max` (MemPlan).
    pub fn entries(lay: &Layout, k_max: usize) -> usize {
        Self::spans(lay).map(|s| 4 * k_max * Self::ids(lay, s).len()).sum()
    }
    pub fn new<S: Lane>(arena: &'a Arena, s: &Stream<S>, lay: Layout, k_max: usize) -> Result<Self> {
        let (mut roff, mut rprime, mut index) = (Vec::new(), Vec::new(), Vec::new());
        let rows = lay.rows();
        for span in Self::spans(&lay) {
            let ids = Self::ids(&lay, span);
            index.push((roff.len(), ids.len()));
            for part in 0..2 {
                for c in 0..k_max {
                    for &id in &ids {
                        roff.push(((c * 2 + part) * rows + id) as u32);
                        rprime.push(id as i32);
                    }
                }
            }
            for c in 0..k_max {
                for part in 0..2 {
                    for &id in &ids {
                        roff.push(((c * 2 + part) * rows + id) as u32);
                        rprime.push(id as i32);
                    }
                }
            }
        }
        Ok(RowTables {
            roff: ConstBuf::new(arena, "ctx.rows.roff", s, &roff)?,
            rprime: ConstBuf::new(arena, "ctx.rows.rprime", s, &rprime)?,
            index,
            lay,
            k_max,
        })
    }
    /// The launches `(first entry, rows)` covering `parts` of the first `k` ciphertexts over `span`.
    pub fn launches(&self, span: Span, parts: Parts, k: usize) -> Result<Vec<(usize, usize)>> {
        if k == 0 || k > self.k_max {
            return Err(shape("ntt k", format!("{k} ciphertexts, row tables cover 1..={} (MemCfg::k_max)", self.k_max)));
        }
        let (start, live) = self.index[self.key(span)?];
        let sec = self.k_max * live;
        Ok(match parts {
            Parts::Both if k == self.k_max => vec![(start, 2 * sec)],
            // one launch (the by-ciphertext section) instead of one per part
            Parts::Both => vec![(start + 2 * sec, 2 * k * live)],
            Parts::C0 => vec![(start, k * live)],
            Parts::C1 => vec![(start + sec, k * live)],
        })
    }
}

/// An arbitrary row list `(limb offset in units of N, prime id)` for buffers other than the canonical ciphertext
/// layout (built at setup by its owner).
pub struct RowList<'a> {
    pub roff: ConstBuf<'a, u32>,
    pub rprime: ConstBuf<'a, i32>,
    /// Largest limb offset + 1 (bounds check against the buffer).
    pub span_limbs: usize,
}

impl<'a> RowList<'a> {
    pub fn new<S: Lane>(arena: &'a Arena, s: &Stream<S>, rows: &[(u32, i32)]) -> Result<Self> {
        let roff: Vec<u32> = rows.iter().map(|r| r.0).collect();
        let rprime: Vec<i32> = rows.iter().map(|r| r.1).collect();
        let span_limbs = roff.iter().map(|&o| o as usize + 1).max().unwrap_or(0);
        Ok(RowList {
            roff: ConstBuf::new(arena, "rowlist.roff", s, &roff)?,
            rprime: ConstBuf::new(arena, "rowlist.rprime", s, &rprime)?,
            span_limbs,
        })
    }
    pub fn len(&self) -> usize {
        self.roff.len()
    }
    pub fn is_empty(&self) -> bool {
        self.roff.is_empty()
    }
}

fn launch<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    s: &Stream<Compute>,
    dir: Dir,
    base: *mut c_void,
    roff: *const u32,
    rprime: *const i32,
    nrows: usize,
) -> Result<()> {
    if nrows == 0 {
        return Ok(());
    }
    let a = NttArgs {
        base,
        roff,
        rprime,
        fw: ctx.ntt.fw.vptr(),
        fws: ctx.ntt.fws.vptr(),
        iw: ctx.ntt.iw.vptr(),
        iws: ctx.ntt.iws.vptr(),
        ninv: ctx.ntt.ninv.vptr(),
        ninvs: ctx.ntt.ninvs.vptr(),
        pr: ctx.pr(),
        nrows: nrows as i32,
        chunk: NTT_CHUNK,
    };
    let (f, name) = match dir {
        Dir::Forward => (ctx.table.ntt_fwd, "ntt_fwd"),
        Dir::Inverse => (ctx.table.ntt_inv, "ntt_inv"),
    };
    // SAFETY: the tables are context constants of this chain; the row tables address rows inside `base`'s buffer
    // (checked by the callers); `base` is exclusively borrowed by the caller for the launch's duration on `s`.
    launch_result(name, unsafe { f(&a, s.raw()) })?;
    Ok(())
}

/// In-place NTT of `parts` over `span` of every ciphertext in `ct` (one or two launches).
pub fn transform<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    s: &Stream<Compute>,
    dir: Dir,
    ct: &mut CtBuf<'_, Compute, W>,
    span: Span,
    parts: Parts,
) -> Result<()> {
    let k = ct.k;
    transform_k(ctx, s, dir, ct, span, parts, k)
}

/// `transform` over the first `k` ciphertexts of `ct` only.
pub fn transform_k<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    s: &Stream<Compute>,
    dir: Dir,
    ct: &mut CtBuf<'_, Compute, W>,
    span: Span,
    parts: Parts,
    k: usize,
) -> Result<()> {
    if ct.lay != ctx.lay || k > ct.k {
        return Err(shape("ntt", format!("{k} of {} cts, buffer layout {:?}, context {:?}", ct.k, ct.lay, ctx.lay)));
    }
    let base = ct.buf.vptr_mut();
    for (start, n) in ctx.rows.launches(span, parts, k)? {
        launch(ctx, s, dir, base, ctx.rows.roff.dptr_at(start)?, ctx.rows.rprime.dptr_at(start)?, n)?;
    }
    Ok(())
}

pub fn forward<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    s: &Stream<Compute>,
    ct: &mut CtBuf<'_, Compute, W>,
    span: Span,
    parts: Parts,
) -> Result<()> {
    transform(ctx, s, Dir::Forward, ct, span, parts)
}

pub fn inverse<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    s: &Stream<Compute>,
    ct: &mut CtBuf<'_, Compute, W>,
    span: Span,
    parts: Parts,
) -> Result<()> {
    transform(ctx, s, Dir::Inverse, ct, span, parts)
}

/// In-place NTT of the rows of `list` in an arbitrary compute buffer (limb offsets relative to `buf`).
pub fn transform_rows<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    s: &Stream<Compute>,
    dir: Dir,
    buf: &mut DBuf<'_, Compute, W>,
    list: &RowList<'_>,
) -> Result<()> {
    if list.span_limbs * ctx.lay.n > buf.len() {
        return Err(shape("ntt rows", format!("row list spans {} limbs, buffer holds {}", list.span_limbs, buf.len() / ctx.lay.n)));
    }
    launch(ctx, s, dir, buf.vptr_mut(), list.roff.dptr(), list.rprime.dptr(), list.len())
}

/// `transform_rows` over the first `nrows` entries of `list` only (lists ordered by ciphertext serve every k).
pub fn transform_rows_prefix<W: DevWord, M: DevStrategy<W>>(
    ctx: &Context<'_, W, M>,
    s: &Stream<Compute>,
    dir: Dir,
    buf: &mut DBuf<'_, Compute, W>,
    list: &RowList<'_>,
    nrows: usize,
) -> Result<()> {
    if nrows > list.len() || list.span_limbs * ctx.lay.n > buf.len() {
        return Err(shape("ntt rows", format!("{nrows} of {} rows; list spans {} limbs, buffer holds {}", list.len(), list.span_limbs, buf.len() / ctx.lay.n)));
    }
    launch(ctx, s, dir, buf.vptr_mut(), list.roff.dptr(), list.rprime.dptr(), nrows)
}
