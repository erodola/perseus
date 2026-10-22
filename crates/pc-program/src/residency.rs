//! The weight ring (seam `PtStorePolicy::StageAhead`): block b+1's plaintexts are encoded by the `PtStore`'s rayon
//! pool (`encode_threads`) straight into its pinned ring and uploaded on the `Copy` lane while block b runs; their
//! `Ready` tokens are joined into `compute` (plus the NTT of the Eval rows) at the graph boundary, before block b+1
//! launches (`ring.stage` -> `replayer.run` -> `ring.adopt`).
//!
//! Block `seq` uses frame `seq % n_in_flight`; staging a frame evicts what it held, and evicted slots are refilled
//! lowest-first so a block's rows land in the same slots every token once the ring cycles. Per-block plaintexts
//! (`pt_N`) are namespaced by block in `bind`; masks and named slot plaintexts are shared names. Details in
//! docs/ARCHITECTURE.md "pc-program residency".
use crate::bind::{Bound, StreamLayout};
use crate::error::{Error, Result};
use pc_cuda::{lane, Arena, Compute, DBuf, Device, Event, EventPool, PinnedBuf, Ready};
use pc_gpu::pt_store::{PtKey, PtStore, StageItem, StageValues};
use pc_gpu::{Context, DevStrategy, DevWord, Executor};
use pc_shadow::Shadow;

/// Host values of a block's plaintexts (the driver: weights, masks), by store key.
pub trait WeightSource {
    fn item(&self, key: &PtKey) -> Result<StageItem>;
}

/// Ring knobs (the Config's `host.n_in_flight` / `host.encode_threads`; the encode pool is the PtStore's).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RingCfg {
    pub n_in_flight: usize,
}

impl Default for RingCfg {
    fn default() -> Self {
        RingCfg { n_in_flight: 2 }
    }
}

/// Counters.
#[derive(Clone, Copy, Debug, Default)]
pub struct RingStats {
    pub staged: usize,
    /// Keys already resident when staged (shared masks, or everything when the store holds all blocks).
    pub hits: usize,
    pub evicted: usize,
    pub stage_ms: f64,
}

pub struct WeightRing {
    pub cfg: RingCfg,
    /// Keys each frame holds.
    frames: Vec<Vec<PtKey>>,
    pub stats: RingStats,
}

fn rerr(msg: impl Into<String>) -> Error {
    Error::Replay { block: "ring".into(), msg: msg.into() }
}

impl WeightRing {
    pub fn new(cfg: RingCfg) -> Result<Self> {
        if cfg.n_in_flight < 2 {
            return Err(Error::Config { knob: "host.n_in_flight", msg: format!("{} (>= 2: one frame runs while the next stages)", cfg.n_in_flight) });
        }
        Ok(WeightRing { cfg, frames: vec![vec![]; cfg.n_in_flight], stats: RingStats::default() })
    }

    /// PtStore slots a block sequence needs: the largest sum of distinct keys over `n_in_flight` consecutive
    /// blocks (cyclic: the last blocks of a token run next to the first of the next one).
    pub fn slots_needed<W: DevWord>(blocks: &[&Bound<'_, W>], n_in_flight: usize) -> usize {
        let n = blocks.len();
        if n == 0 {
            return 0;
        }
        (0..n)
            .map(|s| {
                let mut keys: Vec<PtKey> = vec![];
                for j in 0..n_in_flight.min(n) {
                    for k in blocks[(s + j) % n].pt_keys() {
                        if !keys.contains(&k) {
                            keys.push(k);
                        }
                    }
                }
                keys.len()
            })
            .max()
            .unwrap_or(0)
    }

    /// Setup: run every pinned ring buffer once through the encode pool (threads spawned, FFT tables and pages
    /// touched before the first token), then drop the dummy keys. Synchronizes the lanes (setup only).
    pub fn prewarm<W: DevWord, M: DevStrategy<W>>(&mut self, store: &mut PtStore<'_, '_, W>, ctx: &Context<'_, W, M>, ex: &Executor<'_, W>, sh: &Shadow) -> Result<()> {
        let n = store.cfg.ring.max(1);
        let slots = ctx.lay.n / 2;
        let items: Vec<StageItem> = (0..n)
            .map(|i| StageItem {
                key: PtKey::new(format!("__prewarm.{i}"), 0, pc_gpu::encode_dev::PtForm::Eval),
                values: StageValues::Slots { values: vec![pc_core::encode::Complex::new(0.0, 0.0); slots], slots, payload: pc_core::encode::Payload::Complex },
                deg: 1,
                scale: None,
            })
            .collect();
        store.stage(ex, sh, &items)?;
        let keys: Vec<PtKey> = items.iter().map(|it| it.key.clone()).collect();
        store.adopt(ctx, ex, &keys)?;
        for k in &keys {
            store.evict(ex, k)?;
        }
        ex.copy.sync()?;
        ex.compute.sync()?;
        Ok(())
    }

    /// Stage block `seq`'s plaintexts into frame `seq % n_in_flight` (host encode now, uploads on Copy; no wait).
    pub fn stage<W: DevWord>(&mut self, store: &mut PtStore<'_, '_, W>, ex: &Executor<'_, W>, sh: &Shadow, seq: usize, b: &Bound<'_, W>, src: &dyn WeightSource) -> Result<()> {
        let t = std::time::Instant::now();
        let f = seq % self.cfg.n_in_flight;
        let want = b.pt_keys();
        // evict what the frame held, unless this block (or another frame) still reads it
        let old = std::mem::take(&mut self.frames[f]);
        for k in old {
            let elsewhere = self.frames.iter().any(|fr| fr.contains(&k));
            if !want.contains(&k) && !elsewhere {
                store.evict(ex, &k)?;
                self.stats.evicted += 1;
            }
        }
        let mut items = vec![];
        for k in &want {
            if store.has(k) {
                self.stats.hits += 1;
            } else {
                items.push(src.item(k)?);
            }
        }
        store.stage(ex, sh, &items)?;
        self.stats.staged += items.len();
        self.frames[f] = want;
        self.stats.stage_ms += t.elapsed().as_secs_f64() * 1e3;
        Ok(())
    }

    /// Join block `seq`'s uploads into compute (stream waits + the NTT of Eval rows) and resolve the bound block's
    /// plaintext rows. Returns whether any row address moved (the replay then re-captures that block).
    pub fn adopt<W: DevWord, M: DevStrategy<W>>(&mut self, store: &mut PtStore<'_, '_, W>, ctx: &Context<'_, W, M>, ex: &Executor<'_, W>, seq: usize, b: &mut Bound<'_, W>) -> Result<bool> {
        let f = seq % self.cfg.n_in_flight;
        let keys = &self.frames[f];
        if keys.is_empty() && !b.pts.is_empty() {
            return Err(rerr(format!("{}: frame {f} holds nothing (stage seq {seq} first)", b.label)));
        }
        store.adopt(ctx, ex, keys)?;
        b.resolve_pts(store)
    }
}

// ── streamed weights ──────────────────────────────────────────────────────────────────────────────────────────────
//
// A block that streams its weights (`Kind::CoeffSum`, Llama decode: 4608 plaintexts, ~1.2 GB of float32 rows per
// block) reads them from a device *frame*: the block's linears back to back (`bind::StreamLayout`), uploaded whole
// from their pinned host rows on the Copy lane while the block before runs. `frames` frames rotate; the driver gives
// each block a frame and calls, per block in run order:
//
// ```text
// base = ring.adopt(ex, f, tag)        compute waits the frame's uploads (staged earlier, or cold now)
// replay block (Bound::stream_base = base)
// ring.release(ex, f)                  the frame goes to Copy behind the block's last read (an event, no host wait)
// ring.stage(ex, f', next, layout, host)   the next block's rows into its frame, overlapping this block's compute
// ```
// A frame keeps what it holds after a release: staging the same tag again uploads nothing (`StreamStats::reuses`).

/// The pinned host rows of every streamed linear (`[rows][N]` float32 each), by linear name.
pub trait StreamHost {
    fn pinned(&self, linear: &str) -> Option<&PinnedBuf<f32>>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamRingCfg {
    /// Device frames (>= 2: one is read while the next one fills).
    pub frames: usize,
    /// Rows of N floats a frame holds (the largest block's `StreamLayout::total`).
    pub frame_rows: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct StreamStats {
    /// Frames filled (uploads enqueued) / restaged with what they already held.
    pub stages: u64,
    pub reuses: u64,
    /// Host -> device bytes enqueued (PCIe).
    pub bytes: u64,
    /// Adopts whose frame was staged right before them (no overlap).
    pub cold: u64,
}

enum Frame<'a, 'e> {
    /// On compute, holding `tag`'s rows (or nothing).
    Compute(DBuf<'a, Compute, f32>, Option<usize>),
    /// On Copy (released), still holding `tag`'s rows.
    Copy(DBuf<'a, lane::Copy, f32>, Option<usize>),
    /// Uploads of `tag`'s rows in flight on Copy.
    Loading(Ready<'e, lane::Copy, DBuf<'a, lane::Copy, f32>>, usize),
    Moving,
}

pub struct StreamRing<'a, 'e> {
    pub cfg: StreamRingCfg,
    n: usize,
    frames: Vec<Frame<'a, 'e>>,
    hand: Event,
    events: &'e EventPool,
    pub stats: StreamStats,
}

fn serr(msg: impl Into<String>) -> Error {
    Error::Replay { block: "stream".into(), msg: msg.into() }
}

impl<'a, 'e> StreamRing<'a, 'e> {
    /// Device bytes (MemPlan section "stream").
    pub fn bytes(n: usize, cfg: &StreamRingCfg) -> usize {
        cfg.frames * Arena::bytes_for::<f32>(cfg.frame_rows.max(1) * n)
    }

    pub fn new(arena: &'a Arena, dev: &Device, n: usize, cfg: StreamRingCfg, events: &'e EventPool) -> Result<Self> {
        if cfg.frames < 2 || cfg.frame_rows == 0 {
            return Err(Error::Config { knob: "host.n_in_flight", msg: format!("{} weight frames of {} rows (>= 2 frames, >= 1 row)", cfg.frames, cfg.frame_rows) });
        }
        let frames = (0..cfg.frames).map(|_| Ok(Frame::Compute(arena.alloc("stream.frame", cfg.frame_rows * n)?, None))).collect::<Result<Vec<_>>>()?;
        Ok(StreamRing { cfg, n, frames, hand: Event::new(dev)?, events, stats: StreamStats::default() })
    }

    fn frame(&self, f: usize) -> Result<&Frame<'a, 'e>> {
        self.frames.get(f).ok_or_else(|| serr(format!("frame {f} of {}", self.frames.len())))
    }

    /// The tag frame `f` holds or is loading.
    pub fn holds(&self, f: usize) -> Option<usize> {
        match self.frames.get(f)? {
            Frame::Compute(_, t) | Frame::Copy(_, t) => *t,
            Frame::Loading(_, t) => Some(*t),
            Frame::Moving => None,
        }
    }

    /// Whether frame `f` has `tag`'s rows staged (loading or loaded).
    pub fn staged(&self, f: usize, tag: usize) -> bool {
        matches!(self.frames.get(f), Some(Frame::Loading(_, t)) if *t == tag) || matches!(self.frames.get(f), Some(Frame::Compute(_, Some(t))) if *t == tag)
    }

    /// Hand frame `f` to Copy behind everything compute issued so far (right after the block that read it).
    pub fn release<W: DevWord>(&mut self, ex: &Executor<'a, W>, f: usize) -> Result<()> {
        self.frame(f)?;
        self.frames[f] = match std::mem::replace(&mut self.frames[f], Frame::Moving) {
            Frame::Compute(b, t) => Frame::Copy(b.handoff(&ex.compute, &ex.copy, &self.hand)?, t),
            other => other,
        };
        Ok(())
    }

    /// Enqueue the uploads of `layout`'s linears into frame `f` (tag `tag` = the block they are for) on Copy.
    pub fn stage<W: DevWord>(&mut self, ex: &Executor<'a, W>, f: usize, tag: usize, layout: &StreamLayout, host: &dyn StreamHost) -> Result<()> {
        self.frame(f)?;
        if layout.total as usize > self.cfg.frame_rows {
            return Err(serr(format!("a {}-row weight frame, the frames hold {} rows", layout.total, self.cfg.frame_rows)));
        }
        let (mut buf, held) = match std::mem::replace(&mut self.frames[f], Frame::Moving) {
            Frame::Loading(r, t) if t == tag => {
                self.frames[f] = Frame::Loading(r, t);
                return Ok(());
            }
            Frame::Loading(r, _) => (r.join_as_is(&ex.copy)?, None),
            // a frame never released (cold): Copy waits for compute's work so far
            Frame::Compute(b, t) => (b.handoff(&ex.compute, &ex.copy, &self.hand)?, t),
            Frame::Copy(b, t) => (b, t),
            Frame::Moving => return Err(serr(format!("frame {f} in an inconsistent state"))),
        };
        if held == Some(tag) {
            self.stats.reuses += 1;
        } else {
            for (lin, off, rows) in &layout.linears {
                let src = host.pinned(lin).ok_or_else(|| serr(format!("streamed linear {lin} has no pinned host rows")))?;
                if src.len() != *rows as usize * self.n {
                    return Err(serr(format!("streamed linear {lin}: {} pinned floats, the frame layout says {rows} rows of {}", src.len(), self.n)));
                }
                // SAFETY: the pinned rows are the runtime's for its lifetime and never written after registration;
                // the frame region [off, off + rows) N floats is inside the frame (checked against frame_rows)
                unsafe { buf.upload_pinned_async(&ex.copy, *off as usize * self.n, src)? };
                self.stats.bytes += (src.len() * 4) as u64;
            }
            self.stats.stages += 1;
        }
        self.frames[f] = Frame::Loading(Ready::record(buf, &ex.copy, self.events.get()?)?, tag);
        Ok(())
    }

    /// Make compute wait for frame `f`'s uploads of `tag`'s rows; returns the frame's device address.
    pub fn adopt<W: DevWord>(&mut self, ex: &Executor<'a, W>, f: usize, tag: usize) -> Result<*const f32> {
        self.frame(f)?;
        let fr = match std::mem::replace(&mut self.frames[f], Frame::Moving) {
            Frame::Loading(r, t) if t == tag => Frame::Compute(r.join(&ex.compute)?, Some(t)),
            Frame::Compute(b, Some(t)) if t == tag => Frame::Compute(b, Some(t)),
            other => {
                self.frames[f] = other;
                return Err(serr(format!("frame {f} holds {:?}, not block {tag} (stage it first)", self.holds(f))));
            }
        };
        let p = match &fr {
            Frame::Compute(b, _) => b.dptr(),
            _ => std::ptr::null(),
        };
        self.frames[f] = fr;
        Ok(p)
    }
}
