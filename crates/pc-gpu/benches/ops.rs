//! Host overhead of the leaf ops: per op, 10k issues at K = 1 and K = 8 at a mid-chain level, in chunks
//! of 16 issues (under the launch queue depth even for the multi-launch ops at K = 8) followed by a stream sync. Host us/op = wall time of the issue loops only (the syncs and the
//! `cudaEventElapsedTime` reads are excluded, and chunking keeps the launch queue from filling, so the issue time
//! is not GPU time leaking back). Device us/op = timing events around each chunk.
//!
//! GPU only through tools/gpu.sh:
//!   cargo bench -p pc-gpu --bench ops --no-run   (prints the binary)
//!   tools/gpu.sh out/m5b.bench.log target/release/deps/ops-<hash> [--n 10000]
use pc_core::encode::{Complex, Payload};
use pc_core::modmul::Barrett64;
use pc_core::params::{Chain, ChainSpec};
use pc_core::scale::ScaleTables;
use pc_cuda::{Device, Event, EventPool};
use pc_gpu::encode_dev::{EncoderKind, Payload2, PtForm};
use pc_gpu::ops::{Ops, OpsCfg, Triple};
use pc_gpu::pt_store::{PtKey, PtStore, PtStoreCfg};
use pc_gpu::{Context, CtPool, Executor, MemCfg, MemPlan, PoolClass};
use pc_shadow::{BtsConfig, CtMeta, Quirks, Shadow, ShadowConfig};
use std::sync::Arc;
use std::time::Instant;

type W = u32;
type M = Barrett64;
const CHUNK: usize = 16;

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let n_ops: usize = args.iter().position(|a| a == "--n").and_then(|i| args.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(10_000);
    let dev = Device::open()?;
    let chain = Chain::generate(&ChainSpec::n32())?;
    let cfg = MemCfg { k_max: 8, pool: vec![PoolClass { k: 1, count: 8 }, PoolClass { k: 8, count: 8 }], events: 64, scratch_cts: 2 };
    let mut plan = MemPlan::build::<W, M>(&chain, cfg)?;
    let ocfg = OpsCfg::default();
    Ops::<W>::reserve(&mut plan, &ocfg);
    let scfg = PtStoreCfg { slots: 8, ring: 2, encode_threads: 4, ..PtStoreCfg::default() };
    PtStore::<W>::reserve(&mut plan, &scfg);
    let arena = plan.allocate(&dev)?;
    let mut ex = Executor::<W>::new(&dev, &arena, &plan)?;
    let ctx = Context::<W, M>::new(&arena, &ex.copy, chain.clone(), &plan)?;
    let pool = CtPool::<W>::new(&arena, &plan)?;
    let mut ops = Ops::new(&arena, &dev, &ex.compute, &ctx, ocfg)?;
    let sev = EventPool::new(&dev, 8)?;
    let mut store = PtStore::new(&arena, &dev, &ex.compute, &ctx, &sev, scfg)?;
    ex.sync_compute()?;
    arena.freeze();
    let st = Arc::new(ScaleTables::precompute(&chain)?);
    let sh = Shadow::new(ShadowConfig { chain: ctx.shape, bts: BtsConfig::n32_shipped(), quirks: Quirks::none() }, Some(st.clone()));

    // operands at level 26 (top 27, the g0 block-0 entry level); contents: whatever the pool holds (timing only)
    let level = 26u32;
    let top = (ctx.lay.nq - 1) as u32 - level;
    let sf = st.sf(top as i32)?;
    let m1 = CtMeta { level, deg: 1, nf: sf, slots: 32768, tag: Default::default() };
    let m2 = CtMeta { level, deg: 2, nf: sf * sf, slots: 32768, tag: Default::default() };
    let deeper = CtMeta { level: level + 2, deg: 1, nf: st.sf(top as i32 - 2)?, slots: 32768, tag: Default::default() };
    let vals: Vec<Complex> = (0..32768).map(|i| Complex::new((i as f64 * 0.001).sin(), (i as f64 * 0.002).cos() * 0.5)).collect();
    let key = PtKey::new("bench.pt", level, PtForm::Eval);
    store.encode_into(&ctx, &ex, &mut ops, &sh, key.clone(), Payload2::Slots { values: &vals, slots: 32768, payload: Payload::Complex }, 1)?;
    ex.sync_compute()?;

    println!("[bench] level {level} (top {top}), {n_ops} issues per row, chunks of {CHUNK} + sync");
    println!("| op | K | host us/op | device us/op |");
    println!("|---|---|---|---|");
    let (e0, e1) = (Event::timing(&dev)?, Event::timing(&dev)?);
    for k in [1usize, 8] {
        let mut a = pool.acquire(k, m1)?;
        let mut b = pool.acquire(k, m1)?;
        let mut c2 = pool.acquire(k, m2)?;
        let mut dp = pool.acquire(k, deeper)?;
        let mut out = pool.acquire(k, m1)?;
        let mut d2 = pool.acquire(k, m1)?;
        // zero-filled valid residues so every kernel reads defined words
        for h in [&mut a, &mut b, &mut c2, &mut dp, &mut out, &mut d2] {
            h.get_mut().ok_or("shared")?.buf.buf.zero(&ex.compute)?;
        }
        ex.sync_compute()?;
        let names = [
            "add", "sub", "add (B deeper: ScalarAdjust+Rescale+DropTo)", "add_inplace", "negate", "mult_scalar (real)",
            "mult_scalar (complex)", "add_scalar", "mult_pt", "add_pt", "mult_pt (deg-2 ct: Rescale prelude)", "mult_i",
            "square_tensor", "tensor", "rescale", "drop_to_level", "mult_scalar_sum (8 terms)", "eval_linear_wsum (8)",
            "mult_add_many (2 terms)",
        ];
        for (oi, name) in names.iter().enumerate() {
            let (mut host, mut devt) = (0.0f64, 0.0f64);
            let mut done = 0;
            while done < n_ops {
                let m = CHUNK.min(n_ops - done);
                e0.record(&ex.compute)?;
                let t = Instant::now();
                for _ in 0..m {
                    let pt = store.view(&key)?;
                    let o = out.get_mut().ok_or("shared")?;
                    match oi {
                        0 => ops.add(&ctx, &mut ex, &sh, std::slice::from_mut(o), &[&a], &[&b])?,
                        1 => ops.sub(&ctx, &mut ex, &sh, std::slice::from_mut(o), &[&a], &[&b])?,
                        2 => ops.add(&ctx, &mut ex, &sh, std::slice::from_mut(o), &[&dp], &[&a])?,
                        3 => {
                            o.meta = m1;
                            ops.add_inplace(&ctx, &mut ex, &sh, std::slice::from_mut(o), &[&b])?
                        }
                        4 => ops.negate(&ctx, &mut ex, &sh, std::slice::from_mut(o), &[&a])?,
                        5 => ops.mult_scalar(&ctx, &mut ex, &sh, std::slice::from_mut(o), &[&a], &[Complex::new(0.731, 0.0)])?,
                        6 => ops.mult_scalar(&ctx, &mut ex, &sh, std::slice::from_mut(o), &[&a], &[Complex::new(0.731, -0.2)])?,
                        7 => ops.add_scalar(&ctx, &mut ex, &sh, std::slice::from_mut(o), &[&a], &[0.25])?,
                        8 => ops.mult_pt(&ctx, &mut ex, &sh, std::slice::from_mut(o), &[&a], &[pt])?,
                        9 => ops.add_pt(&ctx, &mut ex, &sh, std::slice::from_mut(o), &[&a], &[pt])?,
                        10 => {
                            let ptd = pc_shadow::PtMeta { level: level + 2, deg: 1, nf: deeper.nf, tag: Default::default() };
                            // the plaintext rows at the deeper level are the same buffer's prefix
                            let p2 = unsafe { pc_gpu::ops::Pt::from_raw(ptd, pt.rows) };
                            ops.mult_pt(&ctx, &mut ex, &sh, std::slice::from_mut(o), &[&c2], &[p2])?
                        }
                        11 => ops.mult_i(&ctx, &mut ex, &sh, std::slice::from_mut(o), &[&a])?,
                        12 | 13 => {
                            let d = d2.get_mut().ok_or("shared")?;
                            let mut tr = Triple { c: o, d2: &mut d.buf };
                            if oi == 12 {
                                ops.square_tensor(&ctx, &mut ex, &sh, &mut tr, &a)?
                            } else {
                                ops.tensor(&ctx, &mut ex, &sh, &mut tr, &a, &b)?
                            }
                        }
                        14 => {
                            o.meta = m2;
                            ops.rescale(&ctx, &mut ex, &sh, std::slice::from_mut(o))?
                        }
                        15 => {
                            o.meta = m1;
                            ops.drop_to_level(&sh, std::slice::from_mut(o), level + 2)
                        }
                        16 => ops.mult_scalar_sum(&ctx, &mut ex, &sh, o, &[&a, &b, &a, &b, &a, &b, &a, &b], &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])?,
                        17 => ops.eval_linear_wsum(&ctx, &mut ex, o, &[&a, &b, &a, &b, &a, &b, &a, &b], &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])?,
                        _ => {
                            o.meta = CtMeta { deg: 2, nf: sf * sf, ..m1 };
                            ops.mult_add_many(&ctx, &mut ex, &sh, o, &[&a, &b], &[pt, pt])?
                        }
                    }
                }
                host += t.elapsed().as_secs_f64();
                e1.record(&ex.compute)?;
                e1.sync()?;
                devt += e0.elapsed_ms(&e1)? as f64 * 1e-3;
                done += m;
            }
            println!("| {name} | {k} | {:.1} | {:.1} |", host * 1e6 / n_ops as f64, devt * 1e6 / n_ops as f64);
        }
        // eager device encode (miss path of mult_cached), fewer issues (the FFT dominates)
        let ne = (n_ops / 10).max(10);
        let (mut host, mut devt) = (0.0, 0.0);
        let mut done = 0;
        while done < ne {
            let m = CHUNK.min(ne - done);
            e0.record(&ex.compute)?;
            let t = Instant::now();
            for _ in 0..m {
                store.encode_with(&ctx, &ex, &mut ops, &sh, EncoderKind::DeviceFft, key.clone(), Payload2::Slots { values: &vals, slots: 32768, payload: Payload::Complex }, 1)?;
            }
            host += t.elapsed().as_secs_f64();
            e1.record(&ex.compute)?;
            e1.sync()?;
            devt += e0.elapsed_ms(&e1)? as f64 * 1e-3;
            done += m;
        }
        if k == 1 {
            println!("| encode (device FFT, eager, 32768 slots, top {top}) | 1 | {:.1} | {:.1} |", host * 1e6 / ne as f64, devt * 1e6 / ne as f64);
        }
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("[bench] error: {e}");
        std::process::exit(2);
    }
}
