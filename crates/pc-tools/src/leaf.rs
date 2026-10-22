//! `pc-check leaf`: per-op checks of the pc-gpu leaf ops against the g0 oracle dump (GPT-2 decode token 0, block 0).
//!
//! Every selected dumped op of block 0 is replayed through `pc_gpu::ops` from its dumped operands, executing the Shadow
//! recipe as the runtime does, at K = 1 and K = 8, under both `Rescaler`s. Compared: output words (bit-exact for every
//! key-free op), decrypted slots (relative error; tensor outputs only this way), level / degree / NoiseFactor. Groups
//! `rescale` / `drop` / `encode` add the composite-vs-sequential rescale, `drop_to_level` and device-vs-host-vs-oracle
//! encode checks. ops.jsonl prints constants with 6 significant digits: the exact scalar is recovered from the dumped
//! output by CRT and the op re-run with it.
use pc_core::encode::{Complex, Encoder, Payload};
use pc_core::modmul::{invmod, mulmod, Barrett64};
use pc_core::params::{Chain, ChainSpec};
use pc_core::scale::ScaleTables;
use pc_cuda::{Device, EventPool};
use pc_gpu::encode_dev::{CoeffRowsDevice, EncoderKind, Payload2, PtForm};
use pc_gpu::ops::{Ops, OpsCfg, Pt, Triple};
use pc_gpu::pt_store::{PtKey, PtStore, PtStoreCfg};
use pc_gpu::rescale::RescaleKind;
use pc_gpu::{Context, Ct, CtPool, Executor, MemCfg, MemPlan, PoolClass};
use crate::dump::{Dump, HostCt};
use crate::dump_g0::{npy_header, G0};
use pc_shadow::{BtsConfig, CtMeta, Prelude, PtMeta, Quirks, Shadow, ShadowConfig};
use rayon::prelude::*;
use serde_json::Value;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

type W = u32;
type M = Barrett64;
const GARNER_LIMBS: usize = 6;
const TOL: f64 = 1.0 / (1u64 << 40) as f64;

// ── host reference: NTT, decryption, decoding ───────────────────────────────────────────────────────────────────

fn brev(i: usize, logn: u32) -> usize {
    (i as u32).reverse_bits() as usize >> (32 - logn)
}

struct HostNtt {
    p: u64,
    w: Vec<u64>,
    iw: Vec<u64>,
    ninv: u64,
}

impl HostNtt {
    fn new(p: u64, psi: u64, n: usize) -> Self {
        let logn = n.trailing_zeros();
        let ipsi = invmod(psi, p);
        let (mut pw, mut ipw) = (vec![1u64; n], vec![1u64; n]);
        for i in 1..n {
            pw[i] = mulmod(pw[i - 1], psi, p);
            ipw[i] = mulmod(ipw[i - 1], ipsi, p);
        }
        HostNtt { p, w: (0..n).map(|i| pw[brev(i, logn)]).collect(), iw: (0..n).map(|i| ipw[brev(i, logn)]).collect(), ninv: invmod(n as u64 % p, p) }
    }
    /// OpenFHE ForwardTransformToBitReverse.
    fn fwd(&self, a: &mut [u64]) {
        let (n, p) = (a.len(), self.p);
        let (mut t, mut m) = (n, 1);
        while m < n {
            t /= 2;
            for i in 0..m {
                let s = self.w[m + i];
                for j in 2 * i * t..2 * i * t + t {
                    let (u, v) = (a[j], mulmod(a[j + t], s, p));
                    a[j] = (u + v) % p;
                    a[j + t] = (u + p - v) % p;
                }
            }
            m *= 2;
        }
    }
    /// OpenFHE InverseTransformFromBitReverse.
    fn inv(&self, a: &mut [u64]) {
        let (n, p) = (a.len(), self.p);
        let mut t = 1;
        let mut m = n / 2;
        while m >= 1 {
            for i in 0..m {
                let s = self.iw[m + i];
                for j in 2 * i * t..2 * i * t + t {
                    let (u, v) = (a[j], a[j + t]);
                    a[j] = (u + v) % p;
                    a[j + t] = mulmod((u + p - v) % p, s, p);
                }
            }
            t *= 2;
            m /= 2;
        }
        a.iter_mut().for_each(|x| *x = mulmod(*x, self.ninv, p));
    }
}

struct Host {
    n: usize,
    q: Vec<u64>,
    ntt: Vec<HostNtt>,
    /// s in the NTT domain for the bottom limbs
    s: Vec<Vec<u64>>,
    enc: Encoder,
}

impl Host {
    fn new(chain: &Chain, sk: &[i8]) -> Self {
        let n = chain.n;
        let roots = chain.roots();
        let ntt: Vec<HostNtt> = (0..GARNER_LIMBS).into_par_iter().map(|l| HostNtt::new(chain.q[l], roots[l], n)).collect();
        let s = (0..GARNER_LIMBS)
            .into_par_iter()
            .map(|l| {
                let p = chain.q[l];
                let mut v: Vec<u64> = sk.iter().map(|&x| if x < 0 { p - (-x) as u64 } else { x as u64 }).collect();
                ntt[l].fwd(&mut v);
                v
            })
            .collect();
        Host { n, q: chain.q.clone(), ntt, s, enc: Encoder::new(chain) }
    }

    /// Balanced Garner over the given coefficient-domain limbs -> f64 coefficients (exact digits, f64 sum).
    fn garner(&self, rows: &[Vec<u64>]) -> Vec<f64> {
        let l = rows.len();
        let q = &self.q[..l];
        let inv: Vec<Vec<u64>> = (0..l).map(|i| (0..l).map(|j| if j < i { invmod(q[j] % q[i], q[i]) } else { 0 }).collect()).collect();
        let mut qprod = vec![1.0f64; l];
        for i in 1..l {
            qprod[i] = qprod[i - 1] * q[i - 1] as f64;
        }
        (0..self.n)
            .into_par_iter()
            .map(|k| {
                let mut d = vec![0i64; l];
                let mut x = 0.0f64;
                for i in 0..l {
                    let qi = q[i];
                    let mut t = rows[i][k] % qi;
                    for j in 0..i {
                        let dj = d[j].rem_euclid(qi as i64) as u64;
                        t = mulmod((t + qi - dj) % qi, inv[i][j], qi);
                    }
                    d[i] = if t > qi / 2 { t as i64 - qi as i64 } else { t as i64 };
                }
                for i in (0..l).rev() {
                    x += d[i] as f64 * qprod[i];
                }
                x
            })
            .collect()
    }

    /// Decode NTT-domain message limbs (rows 0..GARNER_LIMBS) at `nf` into `slots` complex values.
    fn decode_ntt(&self, mut m: Vec<Vec<u64>>, nf: f64, slots: usize) -> Vec<Complex> {
        m.par_iter_mut().enumerate().for_each(|(l, r)| self.ntt[l].inv(r));
        let c = self.garner(&m);
        let gap = self.n / (2 * slots);
        let mut v: Vec<Complex> = (0..slots).map(|i| Complex::new(c[gap * i] / nf, c[gap * i + self.n / 2] / nf)).collect();
        self.enc.fft_special(&mut v);
        v
    }

    /// Decrypt `c0 + c1 s (+ d2 s^2)` (rows by limb, NTT domain).
    fn decrypt(&self, c0: &[Vec<u64>], c1: &[Vec<u64>], d2: Option<&[Vec<u64>]>, nf: f64, slots: usize) -> Vec<Complex> {
        let m: Vec<Vec<u64>> = (0..GARNER_LIMBS)
            .into_par_iter()
            .map(|l| {
                let (p, s) = (self.q[l], &self.s[l]);
                (0..self.n)
                    .map(|i| {
                        let mut v = (c0[l][i] + mulmod(c1[l][i], s[i], p)) % p;
                        if let Some(d) = d2 {
                            v = (v + mulmod(mulmod(d[l][i], s[i], p), s[i], p)) % p;
                        }
                        v
                    })
                    .collect()
            })
            .collect();
        self.decode_ntt(m, nf, slots)
    }
}

fn rel_err(a: &[Complex], b: &[Complex]) -> f64 {
    let mx = b.iter().map(|x| x.abs()).fold(0.0, f64::max).max(1e-300);
    a.iter().zip(b).map(|(x, y)| (*x - *y).abs()).fold(0.0, f64::max) / mx
}

fn read_npy_bytes(p: &Path) -> Result<(String, Vec<u8>), String> {
    let (d, _shape, off) = npy_header(p)?;
    let b = std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
    Ok((d, b[off..].to_vec()))
}
fn read_c16(p: &Path) -> Result<Vec<Complex>, String> {
    let (d, b) = read_npy_bytes(p)?;
    match d.trim_start_matches(['<', '|', '=']) {
        "c16" => Ok(b.chunks_exact(16).map(|c| Complex::new(f64::from_le_bytes(c[..8].try_into().unwrap()), f64::from_le_bytes(c[8..].try_into().unwrap()))).collect()),
        "f8" => Ok(b.chunks_exact(8).map(|c| Complex::new(f64::from_le_bytes(c.try_into().unwrap()), 0.0)).collect()),
        o => Err(format!("{}: dtype {o}", p.display())),
    }
}
fn read_i8(p: &Path) -> Result<Vec<i8>, String> {
    let (d, b) = read_npy_bytes(p)?;
    match d.trim_start_matches(['<', '|', '=']) {
        "i1" => Ok(b.iter().map(|&x| x as i8).collect()),
        "i8" => Ok(b.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as i8).collect()),
        o => Err(format!("{}: dtype {o}", p.display())),
    }
}

// ── cases ────────────────────────────────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Kind {
    Add,
    Sub,
    AddScalar,
    AddPt,
    MultScalar,
    MultPt,
    Tensor,
    SquareTensor,
    MultI,
    Negate,
}

impl Kind {
    fn group(self) -> &'static str {
        match self {
            Kind::Add => "add",
            Kind::Sub => "sub",
            Kind::AddScalar => "add_scalar",
            Kind::AddPt => "add_pt",
            Kind::MultScalar => "mult_scalar",
            Kind::MultPt => "mult_pt",
            Kind::Tensor => "tensor",
            Kind::SquareTensor => "square_tensor",
            Kind::MultI => "mult_i",
            Kind::Negate => "negate",
        }
    }
    fn exact(self) -> bool {
        !matches!(self, Kind::Tensor | Kind::SquareTensor)
    }
}

#[derive(Clone, Debug)]
enum Arg {
    Ct(String),
    Pt(String, u32),
    Const(f64),
}

#[derive(Clone, Debug)]
struct Case {
    i: u64,
    kind: Kind,
    a: String,
    b: Option<Arg>,
    out: String,
    /// operand levels at op time when an earlier op left them changed (ops.jsonl `in_levels` are the operands'
    /// levels AFTER their op: the wrapper realizes a product's pending rescale on the operand object in place)
    a_at: Option<u32>,
    b_at: Option<u32>,
    /// the dumped output is a fused accumulator (`mult_slot_pt_sum`: the loop's later terms were replayed only)
    fused: bool,
}

fn parse_cases(ops: &[Value]) -> Vec<Case> {
    let mut v = Vec::new();
    // level of each name after the last op that read it
    let mut after: HashMap<String, i64> = HashMap::new();
    let ins_of = |o: &Value| -> Vec<String> { o["ins"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default() };
    let lv_of = |o: &Value| -> Vec<i64> { o["in_levels"].as_array().map(|a| a.iter().filter_map(|x| x.as_i64()).collect()).unwrap_or_default() };
    // names never materialized (fused replays) and the in-place adds that fold them into an accumulator
    let unmat: HashSet<String> = ops.iter().filter(|o| o["out_level"].as_i64() == Some(-1)).filter_map(|o| o["out"].as_str().map(String::from)).collect();
    let fused_acc: HashSet<String> = ops
        .iter()
        .filter(|o| o["op"].as_str() == Some("add_inplace"))
        .filter_map(|o| {
            let ins = ins_of(o);
            (ins.len() == 2 && unmat.contains(&ins[1])).then(|| ins[0].clone())
        })
        .collect();
    for o in ops {
        let ins = ins_of(o);
        let lv = lv_of(o);
        let prior: Vec<Option<i64>> = ins.iter().map(|x| after.get(x).copied()).collect();
        for (x, &l) in ins.iter().zip(&lv) {
            if l >= 0 {
                after.insert(x.clone(), l);
            }
        }
        if o["dumped"].as_bool() != Some(true) || !o["step"].as_str().is_some_and(|s| s.starts_with("blk0")) {
            continue;
        }
        let op = o["op"].as_str().unwrap_or("");
        let arg = |j: usize| -> Option<Arg> {
            let s = ins.get(j)?.as_str();
            if let Some(c) = s.strip_prefix("const(").and_then(|x| x.strip_suffix(')')) {
                return c.parse().ok().map(Arg::Const);
            }
            if s.starts_with("pt_") {
                return Some(Arg::Pt(s.to_string(), *lv.get(j)? as u32));
            }
            Some(Arg::Ct(s.to_string()))
        };
        let b = arg(1);
        let kind = match (op, &b) {
            ("add" | "add_inplace", Some(Arg::Const(_))) => Kind::AddScalar,
            ("add" | "add_inplace", Some(Arg::Pt(..))) => Kind::AddPt,
            ("add" | "add_inplace", Some(Arg::Ct(_))) => Kind::Add,
            ("sub_ct", Some(Arg::Ct(_))) => Kind::Sub,
            ("mult", Some(Arg::Const(_))) => Kind::MultScalar,
            ("mult", Some(Arg::Pt(..))) => Kind::MultPt,
            ("mult", Some(Arg::Ct(_))) => Kind::Tensor,
            ("square", _) => Kind::SquareTensor,
            ("mult_i", _) => Kind::MultI,
            ("negate", _) => Kind::Negate,
            _ => continue,
        };
        let a = match ins.first() {
            Some(a) => a.to_string(),
            None => continue,
        };
        let b = if matches!(kind, Kind::SquareTensor | Kind::MultI | Kind::Negate) { None } else { b };
        let at = |j: usize| prior.get(j).copied().flatten().map(|l| l as u32);
        let out = o["out"].as_str().unwrap_or("").to_string();
        let fused = fused_acc.contains(&out);
        let b_at = if matches!(&b, Some(Arg::Ct(_))) { at(1) } else { None };
        v.push(Case { i: o["i"].as_u64().unwrap_or(0), kind, a, b, out, a_at: at(0), b_at, fused });
    }
    v
}

// ── results ──────────────────────────────────────────────────────────────────────────────────────────────────────

#[derive(Default, Clone)]
struct Row {
    cases: usize,
    words_bad: usize,
    words: usize,
    rel: f64,
    meta_bad: usize,
    exact_req: bool,
    errors: usize,
    tol: f64,
}

#[derive(Default)]
struct Ledger {
    rows: BTreeMap<String, Row>,
}

impl Ledger {
    fn add(&mut self, group: &str, exact: bool, bad: usize, words: usize, rel: f64, meta_bad: bool) {
        self.add_tol(group, exact, bad, words, rel, meta_bad, TOL);
    }
    #[allow(clippy::too_many_arguments)]
    fn add_tol(&mut self, group: &str, exact: bool, bad: usize, words: usize, rel: f64, meta_bad: bool, tol: f64) {
        let r = self.rows.entry(group.to_string()).or_insert_with(|| Row { tol: f64::INFINITY, ..Row::default() });
        r.tol = r.tol.min(tol);
        r.cases += 1;
        r.words_bad += bad;
        r.words += words;
        r.rel = r.rel.max(rel);
        r.meta_bad += meta_bad as usize;
        r.exact_req |= exact;
    }
    fn err(&mut self, group: &str) {
        self.rows.entry(group.to_string()).or_insert_with(|| Row { tol: TOL, ..Row::default() }).errors += 1;
    }
    fn pass(r: &Row) -> bool {
        r.errors == 0 && r.meta_bad == 0 && r.rel <= r.tol && (!r.exact_req || r.words_bad == 0)
    }
}

// ── the run ──────────────────────────────────────────────────────────────────────────────────────────────────────

struct Env<'a, 'e> {
    ctx: &'a Context<'a, W, M>,
    ex: Executor<'a, W>,
    pool: &'a CtPool<'a, W>,
    ops: Ops<'a, W>,
    ops_seq: Ops<'a, W>,
    store: PtStore<'a, 'e, W>,
    sh: Shadow,
    host: Host,
    g0: Dump,
    pts: PathBuf,
}

fn meta_of(h: &HostCt, nq: usize) -> CtMeta {
    CtMeta { level: (nq - 1 - h.level) as u32, deg: h.noise_level as u8, nf: h.noise_factor, slots: h.slots as u32, tag: Default::default() }
}

fn upload(env: &Env<'_, '_>, ct: &mut Ct<'_, W>, h: &HostCt) -> Result<(), String> {
    let s = &env.ex.compute;
    for c in 0..ct.k() {
        for part in 0..2 {
            ct.buf.upload_rows(s, c, part, &h.ids, &h.c[part]).map_err(|e| e.to_string())?;
        }
    }
    ct.meta = meta_of(h, env.ctx.lay.nq);
    Ok(())
}

/// Rows (by limb) of ciphertext c of `ct`, parts 0/1, live Q rows.
fn rows_of(env: &Env<'_, '_>, buf: &pc_gpu::CtBuf<'_, pc_cuda::Compute, W>, c: usize, part: usize, top: usize) -> Result<Vec<Vec<u64>>, String> {
    let ids: Vec<usize> = (0..=top).collect();
    let v = buf.download_rows(&env.ex.compute, c, part, &ids).map_err(|e| e.to_string())?;
    let n = env.ctx.lay.n;
    Ok(v.chunks(n).map(|r| r.iter().map(|&x| x as u64).collect()).collect())
}

fn host_rows(h: &HostCt, part: usize, n: usize) -> Vec<Vec<u64>> {
    h.c[part].chunks(n).map(|r| r.iter().map(|&x| x as u64).collect()).collect()
}

fn load_pt(env: &mut Env<'_, '_>, name: &str, level: u32) -> Result<PtKey, String> {
    let key = PtKey::new(name, level, PtForm::Eval);
    if env.store.has(&key) {
        return Ok(key);
    }
    let base = env.pts.join(format!("{name}@{level}"));
    let m: Value = serde_json::from_str(&std::fs::read_to_string(format!("{}.json", base.display())).map_err(|e| format!("{}: {e}", base.display()))?).map_err(|e| e.to_string())?;
    let n = env.ctx.lay.n;
    let ids: Vec<usize> = m["c0"]["ids"].as_array().ok_or("pt ids")?.iter().filter_map(|x| x.as_u64().map(|v| v as usize)).collect();
    let sizes: Vec<usize> = m["c0"]["q_sizes"].as_array().ok_or("pt q_sizes")?.iter().filter_map(|x| x.as_u64().map(|v| v as usize)).collect();
    let raw = crate::dump::read_u32(Path::new(&format!("{}.q.bin", base.display())));
    let enc = m["compact_mask"].as_u64().unwrap_or(0xFFFF_FFFF) as u32;
    let mut rows = vec![0u32; ids.len() * n];
    let mut off = 0;
    for (j, &w) in sizes.iter().enumerate() {
        for i in 0..n {
            let src = if w >= n {
                i
            } else if enc >> 24 == 0x7F {
                i >> (enc & 0xFF)
            } else {
                i & enc as usize
            };
            rows[j * n + i] = raw[off + src.min(w - 1)];
        }
        off += w;
    }
    if ids != (0..ids.len()).collect::<Vec<_>>() {
        return Err(format!("{name}@{level}: ids are not 0..{}", ids.len()));
    }
    let top = m["level"].as_u64().ok_or("pt level")? as usize;
    let meta = PtMeta { level: (env.ctx.lay.nq - 1 - top) as u32, deg: m["noise_level"].as_u64().unwrap_or(1) as u8, nf: m["noise_factor"].as_f64().unwrap_or(0.0), tag: Default::default() };
    if meta.level != level {
        return Err(format!("{name}@{level}: json top {top} is level {}", meta.level));
    }
    env.store.insert_rows(&env.ex, key.clone(), meta, &rows).map_err(|e| e.to_string())?;
    Ok(key)
}

/// CRT of the first 3 limbs (centered) of per-limb values.
fn crt3(v: &[u64], q: &[u64]) -> i128 {
    let (q0, q1, q2) = (q[0] as u128, q[1] as u128, q[2] as u128);
    let big = q0 * q1 * q2;
    let mut x: u128 = 0;
    for (i, &qi) in [q0, q1, q2].iter().enumerate() {
        let hat = big / qi;
        let hinv = invmod((hat % qi) as u64, qi as u64) as u128;
        x = (x + (v[i] as u128 % qi) * hinv % qi * hat) % big;
    }
    if x > big / 2 {
        -((big - x) as i128)
    } else {
        x as i128
    }
}

/// Per-limb ratio/difference at a coefficient where `ours` is invertible, consistent over 8 probes.
fn recover_rows(ours: &[Vec<u64>], dump: &[Vec<u64>], q: &[u64], mult: bool) -> Option<Vec<u64>> {
    let mut out = Vec::with_capacity(3);
    for l in 0..3 {
        let p = q[l];
        let mut val = None;
        let mut seen = 0;
        for i in (0..ours[l].len()).step_by(4099) {
            let (o, d) = (ours[l][i] % p, dump[l][i] % p);
            let r = if mult {
                if o == 0 {
                    continue;
                }
                mulmod(d, invmod(o, p), p)
            } else {
                (d + p - o) % p
            };
            match val {
                None => val = Some(r),
                Some(x) if x != r => return None,
                _ => {}
            }
            seen += 1;
            if seen >= 8 {
                break;
            }
        }
        out.push(val?);
    }
    Some(out)
}

struct Outcome {
    bad: usize,
    words: usize,
    rel: f64,
    meta_bad: bool,
    note: String,
    pre: Vec<Prelude>,
}

impl<'a, 'e> Env<'a, 'e> {
    fn run_case(&mut self, cs: &Case, k: usize, seq: bool, scalar: Option<f64>) -> Result<(Outcome, Vec<Vec<u64>>), String> {
        let nq = self.ctx.lay.nq;
        let n = self.ctx.lay.n;
        let ha = self.g0.ct(&cs.a);
        let hout = self.g0.ct(&cs.out);
        let e = |x: pc_gpu::Error| x.to_string();
        let mut a = self.pool.acquire(k, CtMeta::new(0, 1)).map_err(e)?;
        upload(self, a.get_mut().ok_or("shared")?, &ha)?;
        let mut out = self.pool.acquire(k, CtMeta::new(0, 1)).map_err(e)?;
        let mut d2 = self.pool.acquire(k, CtMeta::new(0, 1)).map_err(e)?;
        let mut b = self.pool.acquire(k, CtMeta::new(0, 1)).map_err(e)?;
        let bkey = match &cs.b {
            Some(Arg::Ct(nm)) => {
                let hb = self.g0.ct(nm);
                upload(self, b.get_mut().ok_or("shared")?, &hb)?;
                None
            }
            Some(Arg::Pt(nm, lv)) => Some(load_pt(self, nm, *lv)?),
            _ => None,
        };
        // operands an earlier op realized in place: pay the pending rescale first (the state the op saw)
        for (x, at, name) in [(&mut a, cs.a_at, "A"), (&mut b, cs.b_at, "B")] {
            if let Some(l) = at {
                let m = x.meta;
                if l == m.level {
                    continue;
                }
                if l == m.level + self.sh.unit() && m.deg == 2 {
                    let xm = x.get_mut().ok_or("shared")?;
                    self.ops.realize(self.ctx, &mut self.ex, &self.sh, std::slice::from_mut(xm)).map_err(e)?;
                } else {
                    return Err(format!("operand {name} was at level {l} at op time, dumped at {} deg {}: not a rescale", m.level, m.deg));
                }
            }
        }
        let pt: Option<Pt<'_>> = match &bkey {
            Some(kk) => Some(self.store.view(kk).map_err(e)?),
            None => None,
        };
        let c = match (&cs.b, scalar) {
            (_, Some(x)) => x,
            (Some(Arg::Const(x)), None) => *x,
            _ => 0.0,
        };
        let ops = if seq { &mut self.ops_seq } else { &mut self.ops };
        let (ctx, ex, sh) = (self.ctx, &mut self.ex, &self.sh);
        let o = out.get_mut().ok_or("shared")?;
        let recipe_pre = match cs.kind {
            Kind::Add | Kind::Sub => sh.add(&a.meta, &b.meta).map_err(|x| x.to_string())?.pre,
            Kind::AddPt => sh.add_pt(&a.meta, &pt.unwrap().meta).map_err(|x| x.to_string())?.pre,
            Kind::MultPt => sh.mult_pt(&a.meta, &pt.unwrap().meta).map_err(|x| x.to_string())?.pre,
            Kind::MultScalar => sh.mult_scalar(&a.meta, c).map_err(|x| x.to_string())?.pre,
            Kind::Negate => sh.negate(&a.meta).map_err(|x| x.to_string())?.pre,
            Kind::Tensor => sh.mult(&a.meta, &b.meta).map_err(|x| x.to_string())?.pre,
            Kind::SquareTensor => sh.square(&a.meta).map_err(|x| x.to_string())?.pre,
            _ => vec![],
        };
        let mut triple = false;
        match cs.kind {
            Kind::Add => ops.add(ctx, ex, sh, std::slice::from_mut(o), &[&a], &[&b]),
            Kind::Sub => ops.sub(ctx, ex, sh, std::slice::from_mut(o), &[&a], &[&b]),
            Kind::AddScalar => ops.add_scalar(ctx, ex, sh, std::slice::from_mut(o), &[&a], &[c]),
            Kind::AddPt => ops.add_pt(ctx, ex, sh, std::slice::from_mut(o), &[&a], &[pt.unwrap()]),
            Kind::MultScalar => ops.mult_scalar(ctx, ex, sh, std::slice::from_mut(o), &[&a], &[Complex::new(c, 0.0)]),
            Kind::MultPt => ops.mult_pt(ctx, ex, sh, std::slice::from_mut(o), &[&a], &[pt.unwrap()]),
            Kind::MultI => ops.mult_i(ctx, ex, sh, std::slice::from_mut(o), &[&a]),
            Kind::Negate => ops.negate(ctx, ex, sh, std::slice::from_mut(o), &[&a]),
            Kind::Tensor | Kind::SquareTensor => {
                triple = true;
                let d = d2.get_mut().ok_or("shared")?;
                let mut t = Triple { c: o, d2: &mut d.buf };
                if cs.kind == Kind::Tensor {
                    ops.tensor(ctx, ex, sh, &mut t, &a, &b)
                } else {
                    ops.square_tensor(ctx, ex, sh, &mut t, &a)
                }
            }
        }
        .map_err(e)?;
        let om = out.meta;
        let top = nq - 1 - om.level as usize;
        let want = meta_of(&hout, nq);
        let nf_rel = ((om.nf - want.nf) / want.nf).abs();
        let meta_bad = om.level != want.level || om.deg != want.deg || nf_rel > 1e-15;
        let mut note = format!("out L{} d{} nf {:.3e}", om.level, om.deg, nf_rel);
        if meta_bad {
            note.push_str(&format!(" (dump L{} d{})", want.level, want.deg));
        }
        let (mut bad, mut words, mut rel) = (0usize, 0usize, 0.0f64);
        let ref_slots = {
            let c0 = host_rows(&hout, 0, n);
            let c1 = host_rows(&hout, 1, n);
            self.host.decrypt(&c0, &c1, None, want.nf, want.slots.max(1) as usize)
        };
        let mut first_c0 = Vec::new();
        for c in 0..k {
            let r0 = rows_of(self, &out.buf, c, 0, top)?;
            let r1 = rows_of(self, &out.buf, c, 1, top)?;
            let d2r = if triple { Some(rows_of(self, &d2.buf, c, 1, top)?) } else { None };
            if !triple && top == hout.level {
                for (part, r) in [&r0, &r1].iter().enumerate() {
                    let h = &hout.c[part];
                    words += h.len();
                    bad += r.iter().flatten().zip(h).filter(|(x, y)| **x != **y as u64).count();
                }
            } else if !triple {
                bad += 1;
                words += 1;
            }
            let ours = self.host.decrypt(&r0, &r1, d2r.as_deref(), om.nf, want.slots.max(1) as usize);
            rel = rel.max(rel_err(&ours, &ref_slots));
            if c == 0 {
                first_c0 = r0;
            }
        }
        Ok((Outcome { bad, words, rel, meta_bad, note, pre: recipe_pre }, first_c0))
    }
}

fn recover_scalar(env: &mut Env<'_, '_>, cs: &Case) -> Result<Option<f64>, String> {
    let c6 = match cs.b {
        Some(Arg::Const(x)) => x,
        _ => return Ok(None),
    };
    let (_o, ours) = env.run_case(cs, 1, false, None)?;
    let hout = env.g0.ct(&cs.out);
    let n = env.ctx.lay.n;
    let dump = host_rows(&hout, 0, n);
    let st = env.ctx.scale.clone();
    let q = &env.ctx.chain.q;
    let top = hout.level;
    let sf = st.sf(top as i32).map_err(|e| e.to_string())?;
    match cs.kind {
        Kind::MultScalar => {
            // dump = x * E, ours = x * E6 (E6 our rows for c6): E = dump / ours * E6
            let e6 = st.elem_for_eval_mult(top as i32, c6, -1).map_err(|e| e.to_string())?;
            let r = recover_rows(&ours, &dump, q, true).ok_or("inconsistent ratio")?;
            let ev: Vec<u64> = (0..3).map(|l| mulmod(r[l], e6[l], q[l])).collect();
            let e = crt3(&ev, q);
            let target = if e < 0 { e as f64 - 1.0 } else { e as f64 };
            let mut c = target / sf;
            for _ in 0..16 {
                let rows = st.elem_for_eval_mult(top as i32, c, -1).map_err(|e| e.to_string())?;
                let got = crt3(&rows, q);
                if got == e {
                    return Ok(Some(c));
                }
                c = if got < e { next_up(c) } else { next_down(c) };
            }
            Err(format!("no double rounds to {e}"))
        }
        Kind::AddScalar => {
            let deg = hout.noise_level as i32;
            let a6 = st.elem_for_eval_add(top as i32, c6.abs(), deg).map_err(|e| e.to_string())?;
            let a6: Vec<u64> = a6.iter().zip(q).map(|(&v, &p)| if c6 < 0.0 { (p - v) % p } else { v }).collect();
            let r = recover_rows(&ours, &dump, q, false).ok_or("inconsistent difference")?;
            let isf = (sf + 0.5) as u64;
            let ev: Vec<u64> = (0..3)
                .map(|l| {
                    let mut v = (r[l] + a6[l]) % q[l];
                    for _ in 1..deg {
                        v = mulmod(v, invmod(isf % q[l], q[l]), q[l]);
                    }
                    v
                })
                .collect();
            let e = crt3(&ev, q);
            let mut c = e as f64 / sf;
            for _ in 0..16 {
                let rows = st.elem_for_eval_add(top as i32, c.abs(), 1).map_err(|e| e.to_string())?;
                let got = crt3(&rows, q) * if c < 0.0 { -1 } else { 1 };
                if got == e {
                    return Ok(Some(c));
                }
                c = if got < e { next_up(c) } else { next_down(c) };
            }
            Err(format!("no double rounds to {e}"))
        }
        _ => Ok(None),
    }
}

fn next_up(x: f64) -> f64 {
    if x == 0.0 {
        return f64::from_bits(1);
    }
    if x > 0.0 {
        f64::from_bits(x.to_bits() + 1)
    } else {
        f64::from_bits(x.to_bits() - 1)
    }
}
fn next_down(x: f64) -> f64 {
    -next_up(-x)
}

pub fn cmd_leaf(args: &[String]) -> ExitCode {
    match run(args) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("[leaf] error: {e}");
            ExitCode::from(2)
        }
    }
}

fn run(args: &[String]) -> Result<bool, String> {
    let mut groups: Vec<String> = ["add", "sub", "add_scalar", "add_pt", "mult_scalar", "mult_pt", "tensor", "square_tensor", "mult_i", "negate", "rescale", "drop", "encode", "composite", "stage", "coeffrows"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let mut dir = None;
    let mut ks = vec![1usize, 8];
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--ops" => {
                i += 1;
                groups = args.get(i).ok_or("--ops needs a list")?.split(',').map(|s| s.to_string()).collect();
            }
            "--k" => {
                i += 1;
                ks = args.get(i).ok_or("--k needs a list")?.split(',').map(|s| s.parse().map_err(|_| "--k: integers")).collect::<Result<_, _>>()?;
            }
            d if !d.starts_with("--") => dir = Some(d.to_string()),
            o => return Err(format!("unknown argument {o}")),
        }
        i += 1;
    }
    let dir = dir.ok_or("usage: pc-check leaf [--ops a,b] [--k 1,8] <g0 dump>")?;
    let want = |g: &str| groups.iter().any(|x| x == g);
    let g0 = G0::open(&dir).map_err(|e| e.to_string())?;
    let chain = Chain::generate(&ChainSpec::n32()).map_err(|e| e.to_string())?;
    let dq: Vec<u64> = g0.params["q"].as_array().ok_or("params.q")?.iter().filter_map(|x| x.as_u64()).collect();
    if dq != chain.q {
        return Err("g0 primes differ from the n32 preset".into());
    }
    let sk = read_i8(&Path::new(&dir).join("sk.npy"))?;
    let t0 = std::time::Instant::now();
    let host = Host::new(&chain, &sk);
    // host NTT self-check
    {
        let mut x: Vec<u64> = (0..chain.n as u64).map(|i| (i * 7919 + 3) % chain.q[0]).collect();
        let y = x.clone();
        host.ntt[0].fwd(&mut x);
        host.ntt[0].inv(&mut x);
        if x != y {
            return Err("host NTT roundtrip failed".into());
        }
    }
    println!("[leaf] host reference ready in {:.1} s", t0.elapsed().as_secs_f64());

    let dev = Device::open().map_err(|e| e.to_string())?;
    let kmax = *ks.iter().max().unwrap_or(&1);
    let cfg = MemCfg { k_max: kmax.max(1), pool: vec![PoolClass { k: 1, count: 6 }, PoolClass { k: kmax.max(1), count: 6 }], events: 64, scratch_cts: 2 };
    let mut plan = MemPlan::build::<W, M>(&chain, cfg.clone()).map_err(|e| e.to_string())?;
    let ocfg = OpsCfg::default();
    let ocfg_seq = OpsCfg { rescale: RescaleKind::Sequential, ..OpsCfg::default() };
    Ops::<W>::reserve(&mut plan, &ocfg);
    Ops::<W>::reserve(&mut plan, &ocfg_seq);
    let scfg = PtStoreCfg { slots: 24, ring: 2, encode_threads: 4, ..PtStoreCfg::default() };
    PtStore::<W>::reserve(&mut plan, &scfg);
    // the device coefficient-row expander (limbs of a level-26 ciphertext) and COEFF_TERMS device rows
    let lay = plan.lay;
    if want("coeffrows") {
        plan.reserve("encode.coeff", CoeffRowsDevice::<W>::bytes(&lay, 32) + pc_cuda::Arena::bytes_for::<f32>(COEFF_TERMS * lay.n));
    }
    println!("[leaf] MemPlan: {}", plan.report());
    let arena = plan.allocate(&dev).map_err(|e| e.to_string())?;
    let ex = Executor::<W>::new(&dev, &arena, &plan).map_err(|e| e.to_string())?;
    let ctx = Context::<W, M>::new(&arena, &ex.copy, chain.clone(), &plan).map_err(|e| e.to_string())?;
    let pool = CtPool::<W>::new(&arena, &plan).map_err(|e| e.to_string())?;
    let ops = Ops::new(&arena, &dev, &ex.compute, &ctx, ocfg).map_err(|e| e.to_string())?;
    let ops_seq = Ops::new(&arena, &dev, &ex.compute, &ctx, ocfg_seq).map_err(|e| e.to_string())?;
    let sev = EventPool::new(&dev, 16).map_err(|e| e.to_string())?;
    let store = PtStore::new(&arena, &dev, &ex.compute, &ctx, &sev, scfg).map_err(|e| e.to_string())?;
    let mut crd = if want("coeffrows") {
        let c = CoeffRowsDevice::<W>::new(&arena, &ex.compute, lay, 32).map_err(|e| e.to_string())?;
        let r: pc_cuda::DBuf<'_, pc_cuda::Compute, f32> = arena.alloc("leaf.coeff_rows", COEFF_TERMS * lay.n).map_err(|e| e.to_string())?;
        Some((c, r))
    } else {
        None
    };
    ex.sync_compute().map_err(|e| e.to_string())?;
    arena.freeze();
    println!("[leaf] device setup {:.1} s (arena {} MB)", t0.elapsed().as_secs_f64(), arena.used_bytes() >> 20);
    let tables = Arc::new(ScaleTables::precompute(&chain).map_err(|e| e.to_string())?);
    let sh = Shadow::new(ShadowConfig { chain: ctx.shape, bts: BtsConfig::n32_shipped(), quirks: Quirks::none() }, Some(tables));
    let mut env = Env { ctx: &ctx, ex, pool: &pool, ops, ops_seq, store, sh, host, g0: g0.cts(), pts: Path::new(&dir).join("pts") };
    let mut led = Ledger::default();

    // decryptor sanity: our decrypt of a dumped ct vs its dumped slots
    {
        let h = env.g0.ct("v_3");
        let sl = read_c16(&Path::new(&dir).join("cts/v_3.slots.npy"))?;
        let n = ctx.lay.n;
        let d = env.host.decrypt(&host_rows(&h, 0, n), &host_rows(&h, 1, n), None, h.noise_factor, h.slots as usize);
        println!("[leaf] host decrypt of v_3 vs dumped slots: rel {:.2e}", rel_err(&d, &sl));
    }

    let ops_all = g0.ops().map_err(|e| e.to_string())?;
    let cases = parse_cases(&ops_all);
    let need_pre = want("rescale") || want("drop");
    let cts_dir = Path::new(&dir).join("cts");
    let mut skipped = 0;
    for cs in &cases {
        let g = cs.kind.group();
        if !want(g) && !need_pre {
            continue;
        }
        // the dump's closure is per selected op; an operand of a non-selected producer may be missing
        let missing: Vec<&str> = [Some(cs.a.as_str()), match &cs.b {
            Some(Arg::Ct(b)) => Some(b.as_str()),
            _ => None,
        }]
        .into_iter()
        .flatten()
        .filter(|v| !cts_dir.join(format!("{v}.json")).exists())
        .collect();
        if cs.fused {
            println!("[leaf] {g:<13} i={:<5} {} -> {}: skipped (dumped output is a fused accumulator: the loop's later terms were replayed, not materialized)", cs.i, cs.a, cs.out);
            skipped += 1;
            continue;
        }
        if !missing.is_empty() {
            println!("[leaf] {g:<13} i={:<5} {} -> {}: skipped (operand {missing:?} not dumped)", cs.i, cs.a, cs.out);
            skipped += 1;
            continue;
        }
        // the exact scalar (ops.jsonl prints 6 digits)
        let scalar = if matches!(cs.kind, Kind::MultScalar | Kind::AddScalar) {
            match recover_scalar(&mut env, cs) {
                Ok(s) => s,
                Err(e) => {
                    println!("[leaf] {g:<13} i={:<5} {} -> {}: scalar recovery failed: {e}", cs.i, cs.a, cs.out);
                    led.err(g);
                    continue;
                }
            }
        } else {
            None
        };
        for &k in &ks {
            let (o, _) = match env.run_case(cs, k, false, scalar) {
                Ok(x) => x,
                Err(e) => {
                    println!("[leaf] {g:<13} i={:<5} K={k} {} -> {}: ERROR {e}", cs.i, cs.a, cs.out);
                    led.err(g);
                    continue;
                }
            };
            let has_r = o.pre.iter().any(|p| matches!(p, Prelude::Rescale(_)));
            let has_d = o.pre.iter().any(|p| matches!(p, Prelude::DropTo(..)));
            let sc = match (scalar, &cs.b) {
                (Some(x), Some(Arg::Const(c6))) => format!(" c={x:e} (ops.jsonl {c6})"),
                _ => String::new(),
            };
            println!(
                "[leaf] {g:<13} i={:<5} K={k} {} -> {}: words {}/{} rel {:.2e} {} pre {:?}{sc}",
                cs.i, cs.a, cs.out, o.bad, o.words, o.rel, o.note, o.pre
            );
            if want(g) {
                led.add(g, cs.kind.exact(), o.bad, o.words, o.rel, o.meta_bad);
            }
            if has_r && want("rescale") {
                led.add("rescale", cs.kind.exact(), o.bad, o.words, o.rel, o.meta_bad);
            }
            if has_d && want("drop") {
                led.add("drop", cs.kind.exact(), o.bad, o.words, o.rel, o.meta_bad);
            }
            // seam contract: the sequential rescaler gives the same words
            if has_r && want("rescale") && k == 1 {
                match (env.run_case(cs, 1, false, scalar), env.run_case(cs, 1, true, scalar)) {
                    (Ok((_, a)), Ok((os, b))) => {
                        let diff = a.iter().flatten().zip(b.iter().flatten()).filter(|(x, y)| x != y).count();
                        println!("[leaf] rescale-contract i={:<5} composite vs sequential: {diff} words differ; sequential vs dump {}/{}", cs.i, os.bad, os.words);
                        led.add("rescale", true, diff + os.bad, a.iter().map(|r| r.len()).sum::<usize>() + os.words, os.rel, os.meta_bad);
                    }
                    _ => led.err("rescale"),
                }
            }
        }
    }

    println!("[leaf] {skipped} cases skipped (operands not in the dump)");
    if want("rescale") {
        d3_rescale(&mut env, &dir, &mut led)?;
    }
    if want("drop") {
        synthetic_drop(&mut env, &cases, &mut led)?;
    }
    if want("encode") {
        encode_check(&mut env, &cases, &mut led)?;
    }
    if want("composite") {
        composite_check(&mut env, &mut led)?;
    }
    if want("stage") {
        stage_check(&mut env, &cases, &mut led)?;
    }
    if let Some((c, r)) = crd.as_mut() {
        coeffrows_check(&mut env, &dev, c, r, &mut led)?;
    }

    println!("\n[leaf] | op | cases | words mismatched | max rel (decrypted) | meta | required | result |");
    println!("[leaf] |---|---|---|---|---|---|---|");
    let mut ok = true;
    for (g, r) in &led.rows {
        let pass = Ledger::pass(r);
        ok &= pass;
        println!(
            "[leaf] | {g} | {} | {}/{} | {:.2e} | {} bad | {} | {} |",
            r.cases,
            r.words_bad,
            r.words,
            r.rel,
            r.meta_bad,
            if r.exact_req { "exact".to_string() } else { format!("<= {:.1e}", r.tol) },
            if pass { "PASS" } else if r.errors > 0 { "ERROR" } else { "FAIL" }
        );
    }
    println!("[leaf] {}", if ok { "PASS" } else { "FAIL" });
    Ok(ok)
}

/// d3: `MR-entry` (top 3, degree 2) rescaled is `MR-bottom` (the bootstrap's rescale to the bottom).
fn d3_rescale(env: &mut Env<'_, '_>, dir: &str, led: &mut Ledger) -> Result<(), String> {
    let d3 = Path::new(dir).parent().map(|p| p.join("d3")).ok_or("no d3 path")?;
    if !d3.join("it2.stash.MR-entry.json").exists() {
        println!("[leaf] d3 rescale: {} not found, skipped", d3.display());
        return Ok(());
    }
    let d = Dump::open(&d3);
    let (hi, ho) = (d.ct("it2.stash.MR-entry"), d.ct("it2.stash.MR-bottom"));
    let nq = env.ctx.lay.nq;
    for seq in [false, true] {
        let e = |x: pc_gpu::Error| x.to_string();
        let mut x = env.pool.acquire(1, CtMeta::new(0, 1)).map_err(e)?;
        upload(env, x.get_mut().ok_or("shared")?, &hi)?;
        let ops = if seq { &mut env.ops_seq } else { &mut env.ops };
        ops.rescale(env.ctx, &mut env.ex, &env.sh, std::slice::from_mut(x.get_mut().ok_or("shared")?)).map_err(e)?;
        let top = nq - 1 - x.meta.level as usize;
        let mut bad = 0;
        let mut words = 0;
        for part in 0..2 {
            let r = rows_of(env, &x.buf, 0, part, top)?;
            words += ho.c[part].len();
            bad += r.iter().flatten().zip(&ho.c[part]).filter(|(a, b)| **a != **b as u64).count();
        }
        let want = meta_of(&ho, nq);
        let nf_rel = ((x.meta.nf - want.nf) / want.nf).abs();
        // the bootstrap stores its own NoiseFactor on MR-bottom (not the rescale's): level and degree are compared
        let meta_bad = x.meta.level != want.level || x.meta.deg != want.deg || top != ho.level;
        println!("[leaf] d3 rescale MR-entry -> MR-bottom ({}): words {bad}/{words}, out L{} d{} (nf vs stash {:.1e}, information)", if seq { "sequential" } else { "composite" }, x.meta.level, x.meta.deg, nf_rel);
        led.add("rescale", true, bad, words, 0.0, meta_bad);
    }
    // d3 mult_i: the EvalMod monomial branch (EM-DA-Im -> EM-mono-Im), reported (the stash may hold a sign)
    if d.exists("it2.stash.EM-mono-Im.json") {
        let (hi, ho) = (d.ct("it2.stash.EM-DA-Im"), d.ct("it2.stash.EM-mono-Im"));
        let e = |x: pc_gpu::Error| x.to_string();
        let mut x = env.pool.acquire(1, CtMeta::new(0, 1)).map_err(e)?;
        let mut y = env.pool.acquire(1, CtMeta::new(0, 1)).map_err(e)?;
        upload(env, x.get_mut().ok_or("shared")?, &hi)?;
        env.ops.mult_i(env.ctx, &mut env.ex, &env.sh, std::slice::from_mut(y.get_mut().ok_or("shared")?), &[&x]).map_err(e)?;
        let top = hi.level;
        let mut bad = 0;
        for part in 0..2 {
            let r = rows_of(env, &y.buf, 0, part, top)?;
            bad += r.iter().flatten().zip(&ho.c[part]).filter(|(a, b)| **a != **b as u64).count();
        }
        println!("[leaf] d3 mult_i EM-DA-Im -> EM-mono-Im: words {bad}/{} (information)", 2 * ho.c[0].len());
    }
    Ok(())
}

/// `drop_to_level` moves no data: every dumped input dropped by 2 levels keeps its words on the new live rows.
fn synthetic_drop(env: &mut Env<'_, '_>, cases: &[Case], led: &mut Ledger) -> Result<(), String> {
    let nq = env.ctx.lay.nq;
    let mut seen = std::collections::HashSet::new();
    for cs in cases.iter().filter(|c| seen.insert(c.a.clone())).take(12) {
        let h = env.g0.ct(&cs.a);
        let e = |x: pc_gpu::Error| x.to_string();
        let mut x = env.pool.acquire(1, CtMeta::new(0, 1)).map_err(e)?;
        upload(env, x.get_mut().ok_or("shared")?, &h)?;
        let lv = x.meta.level + 2;
        if (lv as usize) >= nq {
            continue;
        }
        env.ops.drop_to_level(&env.sh, std::slice::from_mut(x.get_mut().ok_or("shared")?), lv);
        let top = nq - 1 - x.meta.level as usize;
        let n = env.ctx.lay.n;
        let mut bad = 0;
        let mut words = 0;
        for part in 0..2 {
            let r = rows_of(env, &x.buf, 0, part, top)?;
            let want = &h.c[part][..(top + 1) * n];
            words += want.len();
            bad += r.iter().flatten().zip(want).filter(|(a, b)| **a != **b as u64).count();
        }
        led.add("drop", true, bad, words, 0.0, x.meta.level != lv || x.meta.deg != h.noise_level as u8);
    }
    println!("[leaf] drop_to_level on dumped inputs: done");
    Ok(())
}

/// Device FFT vs host FFT vs the oracle device rows, for a sample of dumped plaintexts.
fn encode_check(env: &mut Env<'_, '_>, cases: &[Case], led: &mut Ledger) -> Result<(), String> {
    let mut pts: Vec<(String, u32)> = cases
        .iter()
        .filter_map(|c| match &c.b {
            Some(Arg::Pt(n, l)) => Some((n.clone(), *l)),
            _ => None,
        })
        .collect();
    pts.dedup();
    let n = env.ctx.lay.n;
    let nq = env.ctx.lay.nq;
    let e = |x: pc_gpu::Error| x.to_string();
    for (name, level) in pts.iter().take(8) {
        let base = env.pts.join(format!("{name}@{level}"));
        let m: Value = serde_json::from_str(&std::fs::read_to_string(format!("{}.json", base.display())).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let vals = read_c16(Path::new(&format!("{}.values.npy", base.display())))?;
        let top = m["level"].as_u64().ok_or("level")? as usize;
        // the level's scaling factor is what both encoders use (and what the oracle rows match)
        let nf = env.ctx.scale.sf(top as i32).map_err(|e| e.to_string())?;
        let deg = m["noise_level"].as_u64().unwrap_or(1) as u8;
        let slots = m["slots"].as_u64().unwrap_or((n / 2) as u64) as usize;
        let what = Payload2::Slots { values: &vals, slots, payload: Payload::Complex };
        let mut out = Vec::new();
        for kind in [EncoderKind::DeviceFft, EncoderKind::HostFft] {
            let key = PtKey::new(format!("enc.{kind:?}"), (nq - 1 - top) as u32, PtForm::Eval);
            env.store.encode_with(env.ctx, &env.ex, &mut env.ops, &env.sh, kind, key.clone(), what, deg).map_err(e)?;
            let pt = env.store.view(&key).map_err(e)?;
            out.push(env.store_rows(&pt, top)?);
        }
        // the same in coefficient form: the two FFTs round a coefficient differently when their double errors
        // straddle a .5 boundary (one integer apart); in the NTT domain that touches every word of the limb
        let mut coef = Vec::new();
        for kind in [EncoderKind::DeviceFft, EncoderKind::HostFft] {
            let key = PtKey::new(format!("encc.{kind:?}"), (nq - 1 - top) as u32, PtForm::Coeff);
            env.store.encode_with(env.ctx, &env.ex, &mut env.ops, &env.sh, kind, key.clone(), what, deg).map_err(e)?;
            let pt = env.store.view(&key).map_err(e)?;
            coef.push(env.store_rows(&pt, 0)?);
        }
        let coef_diff = coef[0].iter().zip(&coef[1]).filter(|(a, b)| a != b).count();
        let dump = crate::dump::read_u32(Path::new(&format!("{}.q.bin", base.display())));
        let dev_vs_host = out[0].iter().zip(&out[1]).filter(|(a, b)| a != b).count();
        let dev_vs_old = out[0].iter().zip(&dump).filter(|(a, b)| a != b).count();
        let host_vs_old = out[1].iter().zip(&dump).filter(|(a, b)| a != b).count();
        // decoded slots
        let dec = |r: &[u32]| -> Vec<Complex> {
            let m: Vec<Vec<u64>> = (0..GARNER_LIMBS).map(|l| r[l * n..(l + 1) * n].iter().map(|&x| x as u64).collect()).collect();
            env.host.decode_ntt(m, nf.powi(deg as i32), slots)
        };
        let (dd, dh) = (dec(&out[0]), dec(&out[1]));
        let vmax = vals.iter().map(|x| x.abs()).fold(0.0, f64::max).max(1e-300);
        let dev_host = dd.iter().zip(&dh).map(|(a, b)| (*a - *b).abs()).fold(0.0, f64::max) / vmax;
        let dev_vals = dd.iter().zip(&vals).map(|(a, b)| (*a - *b).abs()).fold(0.0, f64::max) / vmax;
        println!(
            "[leaf] encode {name}@{level} (top {top}, {slots} slots): NTT words dev!=host {dev_vs_host}, dev!=old-stack {dev_vs_old}, host!=old-stack {host_vs_old} of {}; coefficients dev!=host {coef_diff}/{n} (limb 0); decoded |dev-host|/max {dev_host:.2e}, |dev-values|/max {dev_vals:.2e}",
            out[0].len()
        );
        // words that differ between the two FFTs are reported, not gated (rounding at .5 boundaries); the gate is
        // the decoded slots
        led.add_tol("encode", false, 0, 0, dev_host, false, 1e-14);
        // the device encoder reproduces the oracle device rows word for word
        led.add("encode_dev_vs_old", true, dev_vs_old, out[0].len(), 0.0, false);
    }
    Ok(())
}

impl<'a, 'e> Env<'a, 'e> {
    /// Rows `0..=top` of a resident plaintext (copied through a pool ciphertext: `ew_copy` of part 0).
    fn store_rows(&self, pt: &Pt<'_>, top: usize) -> Result<Vec<u32>, String> {
        let e = |x: pc_gpu::Error| x.to_string();
        let mut x = self.pool.acquire(1, CtMeta::new(0, 1)).map_err(e)?;
        let xm = x.get_mut().ok_or("shared")?;
        let n = self.ctx.lay.n;
        // the plaintext rows are contiguous [row][N]: copy them device-to-device into part 0 of the slot
        let words = (top + 1) * n;
        let src = pc_kernels::PolyRef { base: pt.rows as *mut std::ffi::c_void, ct_stride: 0, part_stride: 0 };
        let args = pc_kernels::EwArgs {
            out: xm.buf.w(),
            a: src,
            b: src,
            rs: self.ctx.lay.rs_q(top),
            sc: std::ptr::null(),
            scs: std::ptr::null(),
            pr: self.ctx.pr(),
            k: 1,
            nparts: 1,
            sub: 0,
            zero_p: 0,
        };
        // SAFETY: rows 0..=top of the plaintext slot into part 0 of a pool ciphertext
        pc_cuda::launch_result("ew_copy", unsafe { (self.ctx.table.ew_copy)(&args, self.ex.compute.raw()) }).map_err(|e| e.to_string())?;
        let ids: Vec<usize> = (0..=top).collect();
        let v = xm.buf.download_rows(&self.ex.compute, 0, 0, &ids).map_err(e)?;
        debug_assert_eq!(v.len(), words);
        Ok(v)
    }
}

// ── composite ops and the staging path (no g0 op of their own: checked by identities on dumped operands) ─────────

fn words_of(env: &Env<'_, '_>, ct: &Ct<'_, W>, c: usize) -> Result<Vec<u64>, String> {
    let top = env.ctx.lay.nq - 1 - ct.meta.level as usize;
    let mut v = rows_of(env, &ct.buf, c, 0, top)?.concat();
    v.extend(rows_of(env, &ct.buf, c, 1, top)?.concat());
    Ok(v)
}

/// `x` as rows per limb for both parts -> (scaled by `f` mod q) flat words.
fn scaled(h: &HostCt, f: u64, q: &[u64], n: usize) -> Vec<u64> {
    let mut v = Vec::with_capacity(2 * h.c[0].len());
    for part in 0..2 {
        for (j, &x) in h.c[part].iter().enumerate() {
            let p = q[h.ids[j / n]];
            v.push(mulmod(x as u64, f % p, p));
        }
    }
    v
}

fn diff(a: &[u64], b: &[u64]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len())
}

fn composite_check(env: &mut Env<'_, '_>, led: &mut Ledger) -> Result<(), String> {
    let e = |x: pc_gpu::Error| x.to_string();
    let n = env.ctx.lay.n;
    let q = env.ctx.chain.primes();
    let (h2, h3) = (env.g0.ct("v_2"), env.g0.ct("v_3"));
    let pkey = load_pt(env, "pt_1", 26)?;
    let mut x = env.pool.acquire(1, CtMeta::new(0, 1)).map_err(e)?;
    upload(env, x.get_mut().ok_or("shared")?, &h2)?;

    // mult_add_many: v_3 + v_2 pt_1 + v_2 pt_1 == 3 v_3 (words)
    {
        let mut acc = env.pool.acquire(1, CtMeta::new(0, 1)).map_err(e)?;
        upload(env, acc.get_mut().ok_or("shared")?, &h3)?;
        let pt = env.store.view(&pkey).map_err(e)?;
        env.ops.mult_add_many(env.ctx, &mut env.ex, &env.sh, acc.get_mut().ok_or("shared")?, &[&x, &x], &[pt, pt]).map_err(e)?;
        let bad = diff(&words_of(env, &acc, 0)?, &scaled(&h3, 3, &q, n));
        let mb = acc.meta.level != meta_of(&h3, env.ctx.lay.nq).level || acc.meta.deg != 2;
        println!("[leaf] composite mult_add_many v_3 += 2 x (v_2 * pt_1) == 3 v_3: words {bad}/{}", 2 * h3.c[0].len());
        led.add("mult_add_many", true, bad, 2 * h3.c[0].len(), 0.0, mb);
    }
    // mult_scalar_sum (10 terms: two wsum launches) == the loop of mult_scalar + add_inplace; eval_linear_wsum ==
    // the same words (ElemForEvalMult at the inputs' own top)
    {
        let cs: Vec<f64> = (0..10).map(|j| 0.37 * (j as f64 - 4.5)).collect();
        let terms: Vec<&Ct<'_, W>> = vec![&x; cs.len()];
        let mut s1 = env.pool.acquire(1, CtMeta::new(0, 1)).map_err(e)?;
        env.ops.mult_scalar_sum(env.ctx, &mut env.ex, &env.sh, s1.get_mut().ok_or("shared")?, &terms, &cs).map_err(e)?;
        let mut acc = env.pool.acquire(1, CtMeta::new(0, 1)).map_err(e)?;
        let mut t = env.pool.acquire(1, CtMeta::new(0, 1)).map_err(e)?;
        for (j, &c) in cs.iter().enumerate() {
            let dst = if j == 0 { acc.get_mut() } else { t.get_mut() }.ok_or("shared")?;
            env.ops.mult_scalar(env.ctx, &mut env.ex, &env.sh, std::slice::from_mut(dst), &[&x], &[Complex::new(c, 0.0)]).map_err(e)?;
            if j > 0 {
                env.ops.add_inplace(env.ctx, &mut env.ex, &env.sh, std::slice::from_mut(acc.get_mut().ok_or("shared")?), &[&t]).map_err(e)?;
            }
        }
        let (w1, w2) = (words_of(env, &s1, 0)?, words_of(env, &acc, 0)?);
        let bad = diff(&w1, &w2);
        let mb = s1.meta.level != acc.meta.level || s1.meta.deg != acc.meta.deg || s1.meta.nf != acc.meta.nf;
        println!("[leaf] composite mult_scalar_sum (10 terms) == loop: words {bad}/{} meta L{} d{} vs L{} d{}", w1.len(), s1.meta.level, s1.meta.deg, acc.meta.level, acc.meta.deg);
        led.add("mult_scalar_sum", true, bad, w1.len(), 0.0, mb);
        let mut s2 = env.pool.acquire(1, CtMeta::new(0, 1)).map_err(e)?;
        env.ops.eval_linear_wsum(env.ctx, &mut env.ex, s2.get_mut().ok_or("shared")?, &terms, &cs).map_err(e)?;
        let bad = diff(&words_of(env, &s2, 0)?, &w1);
        println!("[leaf] composite eval_linear_wsum (10 terms) == mult_scalar_sum: words {bad}/{} (out L{} d{})", w1.len(), s2.meta.level, s2.meta.deg);
        led.add("eval_linear_wsum", true, bad, w1.len(), 0.0, s2.meta.level != s1.meta.level || s2.meta.deg != 2);
    }
    // complex scalar: re x + im (X^{N/2} x) words, and decrypted == (re + i im) z
    {
        let c = Complex::new(0.75, -1.25);
        let mut o = env.pool.acquire(1, CtMeta::new(0, 1)).map_err(e)?;
        env.ops.mult_scalar(env.ctx, &mut env.ex, &env.sh, std::slice::from_mut(o.get_mut().ok_or("shared")?), &[&x], &[c]).map_err(e)?;
        let mut r = env.pool.acquire(1, CtMeta::new(0, 1)).map_err(e)?;
        let mut im = env.pool.acquire(1, CtMeta::new(0, 1)).map_err(e)?;
        let mut im2 = env.pool.acquire(1, CtMeta::new(0, 1)).map_err(e)?;
        env.ops.mult_scalar(env.ctx, &mut env.ex, &env.sh, std::slice::from_mut(r.get_mut().ok_or("shared")?), &[&x], &[Complex::new(c.re, 0.0)]).map_err(e)?;
        env.ops.mult_scalar(env.ctx, &mut env.ex, &env.sh, std::slice::from_mut(im.get_mut().ok_or("shared")?), &[&x], &[Complex::new(c.im, 0.0)]).map_err(e)?;
        env.ops.mult_i(env.ctx, &mut env.ex, &env.sh, std::slice::from_mut(im2.get_mut().ok_or("shared")?), &[&im]).map_err(e)?;
        env.ops.add_inplace(env.ctx, &mut env.ex, &env.sh, std::slice::from_mut(r.get_mut().ok_or("shared")?), &[&im2]).map_err(e)?;
        let wo = words_of(env, &o, 0)?;
        let bad = diff(&wo, &words_of(env, &r, 0)?);
        let top = env.ctx.lay.nq - 1 - o.meta.level as usize;
        let ours = env.host.decrypt(&rows_of(env, &o.buf, 0, 0, top)?, &rows_of(env, &o.buf, 0, 1, top)?, None, o.meta.nf, h2.slots as usize);
        let z = env.host.decrypt(&host_rows(&h2, 0, n), &host_rows(&h2, 1, n), None, h2.noise_factor, h2.slots as usize);
        let want: Vec<Complex> = z.iter().map(|&v| v * c).collect();
        let rel = rel_err(&ours, &want);
        println!("[leaf] composite mult_scalar complex (0.75 - 1.25i): words vs re x + mult_i(im x) {bad}/{}, decrypted vs c z rel {rel:.2e}", wo.len());
        led.add("mult_scalar_complex", true, bad, wo.len(), rel, false);
    }
    // PtStore: set_slot_pt / mult_slot_pt (eager device encode on the miss) == v_3; mult_pt_sum of two terms == 2 v_3
    {
        let vals = read_c16(&env.pts.join("pt_1@26.values.npy"))?;
        env.store.set_slot_pt("w.pt_1", vals.clone(), vals.len(), Payload::Complex, 26, 1);
        let mut o = env.pool.acquire(1, CtMeta::new(0, 1)).map_err(e)?;
        env.store.mult_slot_pt(env.ctx, &mut env.ex, &mut env.ops, &env.sh, o.get_mut().ok_or("shared")?, &x, "w.pt_1").map_err(e)?;
        let bad = diff(&words_of(env, &o, 0)?, &scaled(&h3, 1, &q, n));
        println!("[leaf] composite set_slot_pt + mult_slot_pt (device encode) == v_3: words {bad}/{}", 2 * h3.c[0].len());
        led.add("mult_slot_pt", true, bad, 2 * h3.c[0].len(), 0.0, o.meta.level != 26 || o.meta.deg != 2);
        let mut o2 = env.pool.acquire(1, CtMeta::new(0, 1)).map_err(e)?;
        env.store.mult_pt_sum(env.ctx, &mut env.ex, &mut env.ops, &env.sh, o2.get_mut().ok_or("shared")?, &[&x, &x], &[vals.clone(), vals], h2.slots as usize, Payload::Complex).map_err(e)?;
        let bad = diff(&words_of(env, &o2, 0)?, &scaled(&h3, 2, &q, n));
        println!("[leaf] composite mult_pt_sum [v_2, v_2] x [pt_1, pt_1] == 2 v_3: words {bad}/{}", 2 * h3.c[0].len());
        led.add("mult_pt_sum", true, bad, 2 * h3.c[0].len(), 0.0, o2.meta.level != 26 || o2.meta.deg != 2);
        // coefficient rows: the constant polynomial 0.5 multiplies every slot by 0.5
        let mut coeffs = vec![0.0f64; n];
        coeffs[0] = 0.5;
        let mut o3 = env.pool.acquire(1, CtMeta::new(0, 1)).map_err(e)?;
        env.store.mult_coeff_sum(env.ctx, &mut env.ex, &mut env.ops, &env.sh, o3.get_mut().ok_or("shared")?, &[&x], &[coeffs]).map_err(e)?;
        let top = env.ctx.lay.nq - 1 - o3.meta.level as usize;
        let ours = env.host.decrypt(&rows_of(env, &o3.buf, 0, 0, top)?, &rows_of(env, &o3.buf, 0, 1, top)?, None, o3.meta.nf, h2.slots as usize);
        let z = env.host.decrypt(&host_rows(&h2, 0, n), &host_rows(&h2, 1, n), None, h2.noise_factor, h2.slots as usize);
        let want: Vec<Complex> = z.iter().map(|&v| v * 0.5).collect();
        let rel = rel_err(&ours, &want);
        println!("[leaf] composite mult_coeff_sum [v_2] x coeffs(0.5) decrypted vs 0.5 z: rel {rel:.2e}");
        led.add("mult_coeff_sum", false, 0, 0, rel, false);
    }
    Ok(())
}

/// Terms of the coefficient-row check (two launches of COEFF_MAXT: the accumulate path too).
const COEFF_TERMS: usize = 40;

/// `CoeffRowsDevice::sum` (float32 rows expanded on the device, fused into the sum) == `PtStore::mult_coeff_sum`
/// (the same rows encoded on the host as coefficient rows, `mult_pt` + `mult_add_many`): words and metadata (the
/// host path is checked by decryption elsewhere).
fn coeffrows_check(env: &mut Env<'_, '_>, dev: &Device, crd: &mut CoeffRowsDevice<'_, W>, rows: &mut pc_cuda::DBuf<'_, pc_cuda::Compute, f32>, led: &mut Ledger) -> Result<(), String> {
    let e = |x: pc_gpu::Error| x.to_string();
    let n = env.ctx.lay.n;
    let h2 = env.g0.ct("v_2");
    let mut x = env.pool.acquire(1, CtMeta::new(0, 1)).map_err(e)?;
    upload(env, x.get_mut().ok_or("shared")?, &h2)?;
    // a second distinct term ciphertext: x doubled (x + x)
    let mut x2 = env.pool.acquire(1, CtMeta::new(0, 1)).map_err(e)?;
    env.ops.add(env.ctx, &mut env.ex, &env.sh, std::slice::from_mut(x2.get_mut().ok_or("shared")?), &[&x], &[&x]).map_err(e)?;
    // weight-like rows: coefficients of magnitude ~1e-3 (a float32 row of a streamed linear), a few exact zeros
    let mut st = 0x2545_f491_4f6c_dd1du64;
    let mut host = vec![0f32; COEFF_TERMS * n];
    for v in host.iter_mut() {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        let u = (st >> 11) as f64 / (1u64 << 53) as f64;
        *v = if st % 97 == 0 { 0.0 } else { ((u - 0.5) * 2e-3) as f32 };
    }
    rows.upload(&env.ex.compute, &host).map_err(|x| x.to_string())?;
    for terms in [1usize, 7, COEFF_TERMS] {
        let cts: Vec<&Ct<'_, W>> = (0..terms).map(|i| if i % 3 == 1 { &*x2 } else { &*x }).collect();
        let ptrs: Vec<*const f32> = (0..terms).map(|i| unsafe { rows.dptr().add(i * n) }).collect();
        let mut o = env.pool.acquire(1, CtMeta::new(0, 1)).map_err(e)?;
        crd.sum(env.ctx, &env.ex.compute, &env.sh, o.get_mut().ok_or("shared")?, &cts, &ptrs).map_err(e)?;
        // the device rounds half to even (FIDESlib's llrint), pc-core's host encoder half away from zero: ties
        // (~0.07 % of float32 rows at a 2^53.7 scale) are pre-rounded so the host's round lands on the same integer
        let sc = CoeffRowsDevice::<W>::sum_meta(&env.sh, &[x.meta]).map_err(e)?.1;
        let coeffs: Vec<Vec<f64>> = (0..terms).map(|i| host[i * n..(i + 1) * n].iter().map(|&v| (v as f64 * sc).round_ties_even() / sc).collect()).collect();
        let mut r = env.pool.acquire(1, CtMeta::new(0, 1)).map_err(e)?;
        env.store.mult_coeff_sum(env.ctx, &mut env.ex, &mut env.ops, &env.sh, r.get_mut().ok_or("shared")?, &cts, &coeffs).map_err(e)?;
        let (wo, wr) = (words_of(env, &o, 0)?, words_of(env, &r, 0)?);
        let bad = diff(&wo, &wr);
        let mb = o.meta.level != r.meta.level || o.meta.deg != r.meta.deg || o.meta.nf != r.meta.nf;
        println!("[leaf] coeffrows device sum ({terms} terms) == PtStore::mult_coeff_sum (host rows): words {bad}/{} meta L{} d{} nf 2^{:.3} vs L{} d{} nf 2^{:.3}", wo.len(), o.meta.level, o.meta.deg, o.meta.nf.log2(), r.meta.level, r.meta.deg, r.meta.nf.log2());
        led.add("coeff_sum_device", true, bad, wo.len(), 0.0, mb);
    }
    // device time of one 32-term sum (the Llama decode's baby-step size) at x's level: events around 50 sums
    {
        let cts: Vec<&Ct<'_, W>> = (0..32).map(|i| if i % 3 == 1 { &*x2 } else { &*x }).collect();
        let ptrs: Vec<*const f32> = (0..32).map(|i| unsafe { rows.dptr().add(i * n) }).collect();
        let mut o = env.pool.acquire(1, CtMeta::new(0, 1)).map_err(e)?;
        let (e0, e1) = (pc_cuda::Event::timing(dev).map_err(|x| x.to_string())?, pc_cuda::Event::timing(dev).map_err(|x| x.to_string())?);
        let l = env.ctx.lay.nq - x.meta.level as usize;
        let mut words = vec![];
        for ch in [32usize, 16, 8, 4] {
            crd.chunk = ch;
            crd.sum(env.ctx, &env.ex.compute, &env.sh, o.get_mut().ok_or("shared")?, &cts, &ptrs).map_err(e)?;
            e0.record(&env.ex.compute).map_err(|x| x.to_string())?;
            for _ in 0..50 {
                crd.sum(env.ctx, &env.ex.compute, &env.sh, o.get_mut().ok_or("shared")?, &cts, &ptrs).map_err(e)?;
            }
            e1.record(&env.ex.compute).map_err(|x| x.to_string())?;
            e1.sync().map_err(|x| x.to_string())?;
            let ms = e0.elapsed_ms(&e1).map_err(|x| x.to_string())? as f64 / 50.0;
            words.push(words_of(env, &o, 0)?);
            println!("[leaf] coeffrows 32-term sum at {l} limbs, {ch} terms per launch: {ms:.3} ms");
        }
        let bad: usize = words.iter().map(|w| diff(w, &words[0])).sum();
        println!("[leaf] coeffrows chunkings agree: {bad} words differ");
        led.add("coeff_sum_chunking", true, bad, words[0].len() * (words.len() - 1), 0.0, false);
        crd.chunk = pc_kernels::COEFF_MAXT;
    }
    let ovf = crd.overflows(&env.ex.compute).map_err(e)?;
    println!("[leaf] coeffrows overflows {ovf}");
    led.add("coeff_sum_overflow", true, ovf as usize, 1, 0.0, false);
    Ok(())
}

/// The staging path: host encode on the rayon pool into the pinned ring, upload on the copy lane, join on compute
/// (+ NTT), rows == the oracle device rows.
fn stage_check(env: &mut Env<'_, '_>, cases: &[Case], led: &mut Ledger) -> Result<(), String> {
    use pc_gpu::pt_store::{StageItem, StageValues};
    let e = |x: pc_gpu::Error| x.to_string();
    let mut pts: Vec<(String, u32)> = cases
        .iter()
        .filter_map(|c| match &c.b {
            Some(Arg::Pt(n, l)) => Some((n.clone(), *l)),
            _ => None,
        })
        .collect();
    pts.sort();
    pts.dedup();
    pts.truncate(5);
    let mut items = Vec::new();
    for (name, level) in &pts {
        let base = env.pts.join(format!("{name}@{level}"));
        let vals = read_c16(Path::new(&format!("{}.values.npy", base.display())))?;
        let m: Value = serde_json::from_str(&std::fs::read_to_string(format!("{}.json", base.display())).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let slots = m["slots"].as_u64().unwrap_or(32768) as usize;
        // scale None = the level's scaling factor (what the oracle encoded with; the plaintext json's
        // noise_factor is not always that value)
        let _ = &m;
        items.push(StageItem { key: PtKey::new(format!("stage.{name}"), *level, PtForm::Eval), values: StageValues::Slots { values: vals, slots, payload: Payload::Complex }, deg: 1, scale: None });
    }
    let t = std::time::Instant::now();
    env.store.stage(&env.ex, &env.sh, &items).map_err(e)?;
    let issue = t.elapsed().as_secs_f64() * 1e3;
    let keys: Vec<PtKey> = items.iter().map(|i| i.key.clone()).collect();
    env.store.adopt(env.ctx, &env.ex, &keys).map_err(e)?;
    for ((name, level), key) in pts.iter().zip(&keys) {
        let base = env.pts.join(format!("{name}@{level}"));
        let dump = crate::dump::read_u32(Path::new(&format!("{}.q.bin", base.display())));
        let top = env.ctx.lay.nq - 1 - *level as usize;
        let pt = env.store.view(key).map_err(e)?;
        let rows = env.store_rows(&pt, top)?;
        // the staging worker is the host-FFT encoder: compare with the same encoder run eagerly
        let it = &items[keys.iter().position(|k| k == key).unwrap_or(0)];
        let StageValues::Slots { values, slots, payload } = &it.values else { return Err("stage item".into()) };
        let hk = PtKey::new(format!("hostref.{name}"), *level, PtForm::Eval);
        env.store.encode_with(env.ctx, &env.ex, &mut env.ops, &env.sh, EncoderKind::HostFft, hk.clone(), Payload2::Slots { values, slots: *slots, payload: *payload }, 1).map_err(e)?;
        let href = env.store_rows(&env.store.view(&hk).map_err(e)?, top)?;
        env.store.evict(&env.ex, &hk).map_err(e)?;
        let bad = rows.iter().zip(&href).filter(|(a, b)| a != b).count();
        let old = rows.iter().zip(&dump).filter(|(a, b)| a != b).count();
        println!("[leaf] stage {name}@{level}: staged rows vs eager host-FFT rows {bad}/{} (vs old-stack device rows {old}: FFT rounding, see encode)", rows.len());
        led.add("stage", true, bad, rows.len(), 0.0, false);
    }
    let ev = env.store.evict(&env.ex, &keys[0]).map_err(e)?;
    let gone = !env.store.has(&keys[0]);
    println!("[leaf] stage: {} items, host stage() {issue:.1} ms (ring of {}), evict {ev} -> has {}", items.len(), env.store.cfg.ring, !gone);
    led.add("stage", true, (!(ev && gone)) as usize, 1, 0.0, false);
    Ok(())
}
