//! `pc-check leaf` for the key-switching ops: every block-0 record of the g0 dump whose operands and
//! output were dumped is replayed through `pc_gpu::{rotate, relin, limbs}` with the oracle's keys (CPU-map `b` +
//! KSKA seed) and compared residue by residue with the dumped output.
//!
//!   ops: rotate, conjugate, rotate_sum, rotate_hoisted (runs of rotations of one input), rotate_and_sum (ladders
//!   recovered from the op stream: rotate_sum groups + rotate/add_inplace rungs), mult (ct x ct and square through
//!   `Ops::mult` / `Ops::square`, preludes included), relin (the same records through `Ops::tensor_for_relin` + `relin_staged`), merge_monomial
//!   (0 + X^{N/2} x against the dumped mult_i records), limbs (export_limbs of the bootstraps' in/out vs the dumped
//!   coefficient forms, import_limbs round trip).
//!   Every case runs per K (the input replicated K times, every copy compared) and per key source (Cached = `a`
//!   expanded and resident, Regen = `a` regenerated from the seed); mult/relin preludes run in `Ops::mult` / `Ops::tensor_for_relin`.
use pc_core::galois::{conj_galois, rotation_galois};
use pc_core::modmul::Barrett64;
use pc_core::params::{Chain, ChainSpec};
use pc_cuda::Device;
use pc_gpu::ks::{DevKey, KeyA, KeyGroup, KsKeyRef, KsPlans, KskSource};
use pc_core::keys::{KskMaterial, KskSeed};
use pc_gpu::keys::{import_material, DevKsk, KeyRole, KskLayout, KskShape};
use pc_gpu::limbs::{self, MonomialTable};
use pc_gpu::ops::{Ops, OpsCfg};
use pc_gpu::relin;
use pc_gpu::rotate::{self, KsCfg, RotKeys};
use pc_gpu::{Context, Ct, CtPool, Executor, MemCfg, MemPlan, PoolClass, PoolCt};
use crate::dump::{Dump, HostCt};
use crate::dump_g0::G0;
use pc_shadow::{BtsConfig, CtMeta, Quirks, Shadow, ShadowConfig};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

type W = u32;
type M = Barrett64;
type Res<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Debug)]
enum Kind {
    Rotate(i64),
    Conj,
    RotateSum(Vec<i64>),
    Hoisted(Vec<i64>, Vec<Option<String>>),
    Ladder { start: i64, stop: i64, groups: Vec<Vec<i64>> },
    Mult,
    Square,
    Monomial,
    Limbs,
}

#[derive(Clone, Debug)]
struct Case {
    op: &'static str,
    kind: Kind,
    ins: Vec<String>,
    out: String,
    /// the bts dump for `limbs` cases
    bts: bool,
    /// operands an earlier `mult` took as its first operand: FIDESlib rescales them IN PLACE there (FLEXIBLEAUTO
    /// `adjustForMult` on `this`; `square` works on a copy), so this record saw them rescaled, not as dumped
    realized: Vec<bool>,
}

fn steps_of(s: &str) -> Option<Vec<i64>> {
    let inner = s.strip_prefix("rot(")?.strip_suffix(')')?;
    inner.split(',').map(|x| x.trim().parse::<i64>().ok()).collect()
}

fn is_var(s: &str) -> bool {
    !(s.starts_with("pt_") || s.starts_with("const(") || s.starts_with("rot(") || s.starts_with("hint"))
}

fn strs(v: &Value) -> Vec<String> {
    v.as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default()
}

/// The ladder's radix groups as the C++ `rotate_and_sum` issues them (radix 3 = LADDER_RADIX_BITS of the dump).
fn ladder_groups(start: i64, stop: i64, radix: usize) -> Vec<Vec<i64>> {
    let sign = start.signum();
    let mut gaps = Vec::new();
    let mut g = start.abs();
    while g < stop {
        gaps.push(g);
        g *= 2;
    }
    let mut out = Vec::new();
    let mut i0 = 0;
    while i0 < gaps.len() {
        let n = radix.min(gaps.len() - i0);
        if n > 1 {
            out.push((1..(1i64 << n)).map(|j| sign * j * gaps[i0]).collect());
        } else {
            out.push(vec![sign * gaps[i0]]);
        }
        i0 += n;
    }
    out
}

fn discover(ops: &[Value], cts: &Dump, want: &BTreeSet<String>, max_per_op: usize) -> Vec<Case> {
    let has = |v: &str| cts.exists(&format!("{v}.json"));
    let blk: Vec<&Value> = ops.iter().filter(|o| o["step"].as_str().map(|s| s.starts_with("blk0")).unwrap_or(false) && o["op"] != "fx_input").collect();
    let mut cases = Vec::new();
    let mut count: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut push = |c: Case, cases: &mut Vec<Case>| {
        let n = count.entry(c.op).or_default();
        if *n < max_per_op {
            *n += 1;
            cases.push(c);
        }
    };
    let w = |s: &str| want.contains(s);
    for (j, o) in blk.iter().enumerate() {
        let op = o["op"].as_str().unwrap_or("");
        let ins = strs(&o["ins"]);
        let out = o["out"].as_str().unwrap_or("").to_string();
        let dumped = o["dumped"].as_bool().unwrap_or(false);
        let vars: Vec<String> = ins.iter().filter(|s| is_var(s)).cloned().collect();
        let all_in = !vars.is_empty() && vars.iter().all(|v| has(v));
        match op {
            "rotate" if w("rotate") && dumped && all_in => {
                if let Some(st) = steps_of(&ins[1]) {
                    push(Case { op: "rotate", kind: Kind::Rotate(st[0]), ins: vars.clone(), out: out.clone(), bts: false, realized: vec![] }, &mut cases);
                }
            }
            "conjugate" if w("conjugate") && dumped && all_in => push(Case { op: "conjugate", kind: Kind::Conj, ins: vars.clone(), out: out.clone(), bts: false, realized: vec![] }, &mut cases),
            "rotate_sum" if w("rotate_sum") && dumped && all_in => {
                if let Some(st) = steps_of(&ins[1]) {
                    push(Case { op: "rotate_sum", kind: Kind::RotateSum(st), ins: vars.clone(), out: out.clone(), bts: false, realized: vec![] }, &mut cases);
                }
            }
            "mult" | "square" if (w("mult") || w("relin")) && dumped && all_in && vars.len() == ins.len() => {
                let kind = if op == "square" { Kind::Square } else { Kind::Mult };
                let realized: Vec<bool> = vars
                    .iter()
                    .map(|v| blk[..j].iter().any(|p| p["op"] == "mult" && strs(&p["ins"]).first() == Some(v)))
                    .collect();
                for name in ["mult", "relin"] {
                    if w(name) {
                        push(Case { op: name, kind: kind.clone(), ins: vars.clone(), out: out.clone(), bts: false, realized: realized.clone() }, &mut cases);
                    }
                }
            }
            "mult_i" if w("merge_monomial") && dumped && all_in => push(Case { op: "merge_monomial", kind: Kind::Monomial, ins: vars.clone(), out: out.clone(), bts: false, realized: vec![] }, &mut cases),
            _ => {}
        }
        // hoisted runs: consecutive rotations of one input (the wrapper's rotate_hoisted records each output)
        if w("rotate_hoisted") && op == "rotate" && all_in && (j == 0 || !(blk[j - 1]["op"] == "rotate" && strs(&blk[j - 1]["ins"]).first() == ins.first())) {
            let mut steps = Vec::new();
            let mut outs = Vec::new();
            let mut t = j;
            while t < blk.len() && blk[t]["op"] == "rotate" && strs(&blk[t]["ins"]).first() == ins.first() {
                let ii = strs(&blk[t]["ins"]);
                if let Some(st) = steps_of(&ii[1]) {
                    steps.push(st[0]);
                    let ov = blk[t]["out"].as_str().unwrap_or("").to_string();
                    outs.push(if blk[t]["dumped"].as_bool().unwrap_or(false) && has(&ov) { Some(ov) } else { None });
                }
                t += 1;
            }
            if steps.len() >= 2 && outs.iter().any(|o| o.is_some()) {
                push(Case { op: "rotate_hoisted", kind: Kind::Hoisted(steps, outs), ins: vec![ins[0].clone()], out: String::new(), bts: false, realized: vec![] }, &mut cases);
            }
        }
        // ladders: a rotate_sum (or a rung) on a dumped value that is not itself the tail of a ladder step
        if w("rotate_and_sum") && (op == "rotate_sum" || op == "rotate") && all_in {
            let x = ins[0].clone();
            let prev_is_step = j > 0 && {
                let p = blk[j - 1];
                (p["op"] == "rotate_sum" || p["op"] == "add_inplace" || p["op"] == "add") && p["out"].as_str() == Some(&x)
            };
            if !prev_is_step {
                let mut acc = x.clone();
                let mut groups: Vec<Vec<i64>> = Vec::new();
                let mut t = j;
                loop {
                    let r = blk.get(t);
                    match r {
                        Some(r) if r["op"] == "rotate_sum" && strs(&r["ins"]).first() == Some(&acc) => {
                            groups.push(steps_of(&strs(&r["ins"])[1]).unwrap_or_default());
                            acc = r["out"].as_str().unwrap_or("").to_string();
                            t += 1;
                        }
                        Some(r) if r["op"] == "rotate" && strs(&r["ins"]).first() == Some(&acc) => {
                            let rot = r["out"].as_str().unwrap_or("").to_string();
                            let st = steps_of(&strs(&r["ins"])[1]).unwrap_or_default();
                            match blk.get(t + 1) {
                                Some(a) if (a["op"] == "add_inplace" || a["op"] == "add") && strs(&a["ins"]) == vec![acc.clone(), rot.clone()] => {
                                    groups.push(st);
                                    acc = a["out"].as_str().unwrap_or("").to_string();
                                    t += 2;
                                }
                                _ => break,
                            }
                        }
                        _ => break,
                    }
                }
                if groups.len() >= 2 && has(&acc) {
                    let first = &groups[0];
                    let start = first.iter().copied().min_by_key(|s| s.abs()).unwrap_or(0);
                    let last = groups.last().cloned().unwrap_or_default();
                    // the last group's top gap: steps j*h, j < 2^n -> the largest gap is h * 2^(n-1)
                    let h = last.iter().copied().min_by_key(|s| s.abs()).unwrap_or(1).abs();
                    let n = (last.len() + 1).trailing_zeros();
                    let stop = h << n;
                    if start != 0 && ladder_groups(start, stop, 3) == groups {
                        push(Case { op: "rotate_and_sum", kind: Kind::Ladder { start, stop, groups }, ins: vec![x], out: acc, bts: false, realized: vec![] }, &mut cases);
                    }
                }
            }
        }
    }
    cases
}

/// Ciphertext metadata of a dumped ct (FIDESlib top limb -> pc-shadow level).
fn meta_of(h: &HostCt, nq: usize) -> CtMeta {
    CtMeta { level: (nq - 1 - h.level) as u32, deg: h.noise_level as u8, nf: h.noise_factor, slots: h.slots as u32, tag: Default::default() }
}

struct Keys<'a> {
    keys: BTreeMap<u64, DevKey<'a, W>>,
    /// `--keys packed[:band] | dense[:band]`: the device key layout (`DevKsk`) instead of the local `DevKey`
    m5a: BTreeMap<u64, DevKsk<'a, W>>,
}
impl Keys<'_> {
    fn get(&self, g: u64) -> Option<&dyn KsKeyRef> {
        self.m5a.get(&g).map(|k| k as &dyn KsKeyRef).or_else(|| self.keys.get(&g).map(|k| k as &dyn KsKeyRef))
    }
    fn has(&self, g: u64) -> bool {
        self.m5a.contains_key(&g) || self.keys.contains_key(&g)
    }
}

/// Key storage under test: local dense keys, or the device key layout at a band (Q rows 0..=q_top kept).
#[derive(Clone, Copy, Debug)]
enum KeyMode {
    Local,
    M5a { layout_packed: bool, band: Option<usize> },
}

struct Groups<'a> {
    by_steps: BTreeMap<Vec<i64>, KeyGroup<'a>>,
}
impl RotKeys for Groups<'_> {
    fn group(&self, steps: &[i64]) -> Option<&KeyGroup<'_>> {
        self.by_steps.get(steps)
    }
}

fn key_name(g: u64, relin: bool) -> String {
    if relin {
        "mult".into()
    } else {
        g.to_string()
    }
}

fn upload(ct: &mut Ct<'_, W>, s: &pc_cuda::Stream<pc_cuda::Compute>, h: &HostCt, nq: usize) -> Res<()> {
    for c in 0..ct.k() {
        for part in 0..2 {
            ct.buf.upload_rows(s, c, part, &h.ids, &h.c[part])?;
        }
    }
    ct.meta = meta_of(h, nq);
    Ok(())
}

fn diff(ct: &Ct<'_, W>, s: &pc_cuda::Stream<pc_cuda::Compute>, h: &HostCt) -> Res<(usize, usize)> {
    let mut bad = 0;
    let mut tot = 0;
    for c in 0..ct.k() {
        for part in 0..2 {
            let got = ct.buf.download_rows(s, c, part, &h.ids)?;
            bad += got.iter().zip(&h.c[part]).filter(|(a, b)| a != b).count();
            tot += got.len();
        }
    }
    Ok((bad, tot))
}

#[derive(Default)]
struct Tally {
    by_op: BTreeMap<String, (usize, usize, usize)>, // cases, bad residues, residues
}
impl Tally {
    fn add(&mut self, what: &str, bad: usize, tot: usize) {
        let e = self.by_op.entry(what.to_string()).or_default();
        e.0 += 1;
        e.1 += bad;
        e.2 += tot;
    }
}

pub fn cmd_leaf(args: &[String]) -> ExitCode {
    match run(args) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            println!("[pc-check] leaf: error: {e}");
            ExitCode::from(2)
        }
    }
}

fn run(args: &[String]) -> Res<bool> {
    let mut dir = None;
    let mut want: BTreeSet<String> = BTreeSet::new();
    let mut ks_list = vec![1usize];
    let mut sources = vec![KskSource::Cached, KskSource::Regen];
    let mut max_per_op = 6usize;
    let mut list = false;
    let mut kmode = KeyMode::Local;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--ops" => {
                want = args.get(i + 1).ok_or("--ops needs a list")?.split(',').map(String::from).collect();
                i += 1;
            }
            "--k" => {
                ks_list = args.get(i + 1).ok_or("--k needs a list")?.split(',').map(|x| x.parse()).collect::<std::result::Result<_, _>>()?;
                i += 1;
            }
            "--source" => {
                sources = match args.get(i + 1).map(|s| s.as_str()) {
                    Some("cached") => vec![KskSource::Cached],
                    Some("regen") => vec![KskSource::Regen],
                    _ => vec![KskSource::Cached, KskSource::Regen],
                };
                i += 1;
            }
            "--list" => list = true,
            "--keys" => {
                let v = args.get(i + 1).ok_or("--keys local | packed[:q_top] | dense[:q_top]")?;
                let (kind, band) = v.split_once(':').map(|(a, b)| (a, b.parse::<usize>().ok())).unwrap_or((v.as_str(), None));
                kmode = match kind {
                    "packed" => KeyMode::M5a { layout_packed: true, band },
                    "dense" => KeyMode::M5a { layout_packed: false, band },
                    _ => KeyMode::Local,
                };
                i += 1;
            }
            "--max" => {
                max_per_op = args.get(i + 1).ok_or("--max needs a number")?.parse()?;
                i += 1;
            }
            d => dir = Some(d.to_string()),
        }
        i += 1;
    }
    let dir = dir.ok_or("usage: pc-check leaf --ops a,b <g0 dump> [--k 1,8] [--source cached|regen|both] [--max n]")?;
    let ks_known = ["rotate", "conjugate", "rotate_sum", "rotate_hoisted", "rotate_and_sum", "mult", "relin", "merge_monomial", "limbs"];
    let unknown: Vec<&String> = want.iter().filter(|o| !ks_known.contains(&o.as_str())).collect();
    if !unknown.is_empty() {
        println!("[pc-check] leaf: ops {unknown:?} are not key-switching ops (`pc-check leaf` runs the other leaf ops)");
    }
    let t0 = Instant::now();
    let g0 = G0::open(&dir)?;
    let pj = &g0.params;
    let chain = Chain::generate(&ChainSpec::n32())?;
    let nq = chain.q.len();
    let qd: Vec<u64> = pj["q"].as_array().ok_or("params.q")?.iter().filter_map(|x| x.as_u64()).collect();
    let pd: Vec<u64> = pj["p"].as_array().ok_or("params.p")?.iter().filter_map(|x| x.as_u64()).collect();
    if qd != chain.q || pd != chain.p {
        return Err("the dump's primes differ from the n32 preset".into());
    }
    let n = chain.n;
    let dnum = chain.dnum();
    let cts = g0.cts();
    let btsd = g0.bts();
    let ops = g0.ops()?;
    let mut cases = discover(&ops, &cts, &want, max_per_op);
    if want.contains("limbs") {
        for k in 1..=10 {
            if btsd.exists(&format!("{k}.in.c0coef.bin")) {
                for side in ["in", "out"] {
                    cases.push(Case { op: "limbs", kind: Kind::Limbs, ins: vec![format!("{k}.{side}")], out: format!("{k}.{side}"), bts: true, realized: vec![] });
                }
            }
        }
    }
    let mut per: BTreeMap<&str, usize> = BTreeMap::new();
    for c in &cases {
        *per.entry(c.op).or_default() += 1;
    }
    println!("[pc-check] leaf: {} cases {per:?}; K {ks_list:?}; sources {sources:?}", cases.len());

    // keys every case needs (galois element -> relin flag)
    let mut need: BTreeSet<(u64, bool)> = BTreeSet::new();
    let mut groups_needed: BTreeSet<Vec<i64>> = BTreeSet::new();
    let rg = |s: i64| rotation_galois(s, n).map_err(|e| e.to_string());
    for c in &cases {
        match &c.kind {
            Kind::Rotate(s) => {
                if rg(*s)? != 1 {
                    need.insert((rg(*s)?, false));
                    groups_needed.insert(vec![*s]);
                }
            }
            Kind::Conj => {
                need.insert((conj_galois(n)?, false));
            }
            Kind::RotateSum(st) => {
                for &s in st {
                    need.insert((rg(s)?, false));
                }
                groups_needed.insert(st.clone());
            }
            Kind::Hoisted(st, _) => {
                let nz: Vec<i64> = st.iter().copied().filter(|&s| rg(s).map(|g| g != 1).unwrap_or(false)).collect();
                for &s in &nz {
                    need.insert((rg(s)?, false));
                }
                groups_needed.insert(nz);
            }
            Kind::Ladder { groups, .. } => {
                for gr in groups {
                    for &s in gr {
                        need.insert((rg(s)?, false));
                    }
                    groups_needed.insert(gr.clone());
                }
            }
            Kind::Mult | Kind::Square => {
                need.insert((1, true));
            }
            _ => {}
        }
    }
    println!("[pc-check] leaf: {} keys, {} key groups", need.len(), groups_needed.len());
    if list {
        for c in &cases {
            println!("[pc-check]   {:<15} {:?} -> {} {:?}", c.op, c.ins, c.out, c.kind);
        }
        return Ok(true);
    }

    // memory plan
    let dev = Device::open()?;
    let kmax = 8;
    let cfg = MemCfg { k_max: kmax, pool: vec![PoolClass { k: 1, count: 8 }, PoolClass { k: kmax, count: 10 }], events: 16, scratch_cts: 2 };
    let mut plan = MemPlan::build::<W, M>(&chain, cfg)?;
    let lay = plan.lay;
    plan.reserve("ks.plans", KsPlans::bytes::<W>(&chain, kmax)?);
    let m5a_layout = |packed: bool| -> Res<KskLayout> { Ok(if packed { KskLayout::packed_for::<W>(&chain)? } else { KskLayout::Dense }) };
    let shape_of = |relin: bool, band: Option<usize>| -> Res<KskShape> {
        Ok(match band {
            Some(q) if !relin => KskShape::banded(&chain, q)?,
            _ => KskShape::full(&chain),
        })
    };
    match kmode {
        KeyMode::Local => plan.reserve("keys", need.len() * DevKey::<W>::bytes(&lay, dnum, true, true)),
        KeyMode::M5a { layout_packed, band } => {
            let mut b = 0;
            for &(_, relin) in &need {
                b += shape_of(relin, band)?.bytes::<W>(n, m5a_layout(layout_packed)?);
            }
            plan.reserve("keys", b);
        }
    }
    println!("[pc-check] leaf: key storage {kmode:?}");
    plan.reserve("ks.groups", groups_needed.iter().map(|g| KeyGroup::bytes(g.len().max(1), true)).sum::<usize>() + 2 * KeyGroup::bytes(1, true));
    plan.reserve("ks.regen_a", pc_gpu::ks::KsWork::<W>::regen_bytes(&lay, dnum));
    plan.reserve("limbs.monomial", MonomialTable::<W>::bytes(lay.nq, lay.n));
    Ops::<W>::reserve(&mut plan, &OpsCfg::default());
    println!("[pc-check] leaf: MemPlan {}", plan.report());
    let arena = plan.allocate(&dev)?;
    let mut ex = Executor::<W>::new(&dev, &arena, &plan)?;
    ex.ks.carve_regen(&arena, &lay)?;
    let ctx = Context::<W, M>::new(&arena, &ex.copy, chain.clone(), &plan)?;
    let pool = CtPool::<W>::new(&arena, &plan)?;
    let kplans = KsPlans::new(&arena, &ex.compute, &ctx, kmax)?;
    let mono = MonomialTable::new(&arena, &ex.compute, &ctx, n / 2)?;
    let mut ops = Ops::new(&arena, &dev, &ex.compute, &ctx, OpsCfg::default())?;
    // oracle keys: b from the dump, a expanded from the seed (Cached) and the seed kept (Regen)
    let tk = Instant::now();
    let mut keys = Keys { keys: BTreeMap::new(), m5a: BTreeMap::new() };
    for &(g, relin) in &need {
        let name = key_name(g, relin);
        if let KeyMode::M5a { layout_packed, band } = kmode {
            let k = &pj["keys"][&name];
            let seed: Vec<u32> = k["a_seed"].as_array().ok_or("a_seed")?.iter().filter_map(|x| x.as_u64().map(|v| v as u32)).collect();
            let seed: [u32; 8] = seed.try_into().map_err(|_| "seed must be 8 words")?;
            let moduli: Vec<u32> = k["b_moduli"][0].as_array().ok_or("b_moduli")?.iter().filter_map(|x| x.as_u64().map(|v| v as u32)).collect();
            let file = if relin { "keys/mult".to_string() } else { format!("keys/auto{name}") };
            let b = crate::dump::read_u32(&g0.dir.join(format!("{file}.b.bin")));
            let mat = KskMaterial::new(KskSeed(seed), dnum, n, moduli, b)?;
            let role = if relin {
                KeyRole::Relin
            } else if g == conj_galois(n)? {
                KeyRole::Conj
            } else {
                KeyRole::Rot { step: 0, galois: g }
            };
            let mut dk = DevKsk::carve(&arena, role, shape_of(relin, band)?, m5a_layout(layout_packed)?, n)?;
            import_material(&mut dk, &ex.compute, &chain, &mat, false)?;
            keys.m5a.insert(if relin { 1 } else { g }, dk);
            continue;
        }
        let k = &pj["keys"][&name];
        let seed: Vec<u32> = k["a_seed"].as_array().ok_or("a_seed")?.iter().filter_map(|x| x.as_u64().map(|v| v as u32)).collect();
        let seed: [u32; 8] = seed.try_into().map_err(|_| "seed must be 8 words")?;
        let file = if relin { "keys/mult".to_string() } else { format!("keys/auto{name}") };
        let b = crate::dump::read_u32(&g0.dir.join(format!("{file}.b.bin")));
        let dk = DevKey::new(&arena, &ex.compute, &ctx, &kplans, if relin { 1 } else { g }, dnum, &b, Some(seed), KeyA::Expand)?;
        keys.keys.insert(if relin { 1 } else { g }, dk);
    }
    ex.compute.sync()?;
    // the expansion gate: regenerated `a` == dumped dense `a` where the dump has one
    let mut t = Tally::default();
    for (&g, dk) in &keys.keys {
        let name = key_name(g, g == 1);
        if pj["keys"][&name]["has_a"].as_bool().unwrap_or(false) {
            let file = if g == 1 { "keys/mult".to_string() } else { format!("keys/auto{name}") };
            let a = crate::dump::read_u32(&g0.dir.join(format!("{file}.a.bin")));
            let got = dk.a.as_ref().ok_or("no a")?.download(&ex.compute)?;
            let bad = got.iter().zip(&a).filter(|(x, y)| x != y).count();
            println!("[pc-check] key {name}: expanded a vs dumped dense a: {bad} mismatches of {}", a.len());
            t.add("key_expand", bad, a.len());
        }
    }
    println!("[pc-check] leaf: keys loaded in {:.1} s", tk.elapsed().as_secs_f64());
    let regen_ptr = ex.ks.stage_ptrs();
    let mk_group = |steps: &[i64], extra: Option<u64>| -> Res<KeyGroup<'_>> {
        let mut refs: Vec<&dyn KsKeyRef> = Vec::new();
        for &s in steps {
            refs.push(keys.get(rg(s)?).ok_or("missing key")?);
        }
        if let Some(g) = extra {
            refs.push(keys.get(g).ok_or("missing key")?);
        }
        Ok(KeyGroup::new(&arena, &ex.compute, n, &refs, regen_ptr)?)
    };
    let mut groups = Groups { by_steps: BTreeMap::new() };
    for st in &groups_needed {
        if !st.is_empty() {
            groups.by_steps.insert(st.clone(), mk_group(st, None)?);
        }
    }
    let conj_group = if keys.has(conj_galois(n)?) { Some(mk_group(&[], Some(conj_galois(n)?))?) } else { None };
    let relin_group = if keys.has(1) { Some(mk_group(&[], Some(1))?) } else { None };
    ex.compute.sync()?;
    println!("[pc-check] leaf: setup {:.1} s; arena used {} of {} B", t0.elapsed().as_secs_f64(), arena.used_bytes(), arena.capacity());
    arena.freeze();

    let sh = Shadow::new(ShadowConfig { chain: ctx.shape, bts: BtsConfig::n32_shipped(), quirks: Quirks::none() }, Some(Arc::new(ctx.scale.clone())));
    for case in &cases {
        for &k in &ks_list {
            for &src in &sources {
                let srcs = match src {
                    KskSource::Cached => "cached",
                    KskSource::Regen => "regen",
                };
                // ops without a key switch run once
                if matches!(case.kind, Kind::Monomial | Kind::Limbs) && src == KskSource::Regen {
                    continue;
                }
                let kc = KsCfg { plans: &kplans, source: src };
                let r = run_case(&ctx, &mut ex, kc, &sh, &pool, &groups, conj_group.as_ref(), relin_group.as_ref(), &mono, &mut ops, &cts, &btsd, case, k, nq);
                match r {
                    Ok(v) => {
                        for (what, bad, tot) in v {
                            println!("[pc-check] {:<15} {:<28} k={k} {srcs:<6} {bad} mismatches of {tot}", case.op, what);
                            t.add(case.op, bad, tot);
                        }
                    }
                    Err(e) if e.to_string().contains("key band ends") => {
                        println!("[pc-check] {:<15} {:?} k={k} {srcs}: SKIP (above the key band: {e})", case.op, case.ins);
                    }
                    Err(e) => {
                        println!("[pc-check] {:<15} {:?} k={k} {srcs}: ERROR {e}", case.op, case.ins);
                        t.add(case.op, 1, 0);
                    }
                }
            }
        }
    }
    let mut bad = 0;
    println!("[pc-check] leaf summary (op: cases, mismatching residues / residues compared):");
    for (op, (c, b, tot)) in &t.by_op {
        println!("[pc-check]   {op:<15} {c:>4} runs  {b} / {tot}");
        bad += b;
    }
    println!("[pc-check] leaf: {} runs, {bad} mismatches ({:.1} s)", t.by_op.values().map(|v| v.0).sum::<usize>(), t0.elapsed().as_secs_f64());
    Ok(bad == 0)
}

fn acquire<'p, 'a>(pool: &'p CtPool<'a, W>, k: usize) -> Res<PoolCt<'p, 'a, W>> {
    Ok(pool.acquire(k, CtMeta::new(0, 1))?)
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    ctx: &Context<'_, W, M>,
    ex: &mut Executor<'_, W>,
    kc: KsCfg<'_, '_>,
    sh: &Shadow,
    pool: &CtPool<'_, W>,
    groups: &Groups<'_>,
    conj: Option<&KeyGroup<'_>>,
    relin_key: Option<&KeyGroup<'_>>,
    mono: &MonomialTable<'_, W>,
    ops: &mut Ops<'_, W>,
    cts: &Dump,
    btsd: &Dump,
    case: &Case,
    k: usize,
    nq: usize,
) -> Res<Vec<(String, usize, usize)>> {
    let hin = if case.bts { btsd.ct(&case.ins[0]) } else { cts.ct(&case.ins[0]) };
    let mut x = acquire(pool, k)?;
    upload(x.get_mut().ok_or("shared")?, &ex.compute, &hin, nq)?;
    let mut res = Vec::new();
    let label = format!("{} -> {}", case.ins.join(","), case.out);
    match &case.kind {
        Kind::Rotate(step) => {
            let mut o = acquire(pool, k)?;
            let grp = groups.group(&[*step]).ok_or("no group")?;
            rotate::rotate(ctx, ex, kc, sh, grp, &x, *step, o.get_mut().ok_or("shared")?)?;
            let (b, tt) = diff(&o, &ex.compute, &cts.ct(&case.out))?;
            res.push((format!("{label} rot {step}"), b, tt));
        }
        Kind::Conj => {
            let mut o = acquire(pool, k)?;
            rotate::conjugate(ctx, ex, kc, sh, conj.ok_or("no conj key")?, &x, o.get_mut().ok_or("shared")?)?;
            let (b, tt) = diff(&o, &ex.compute, &cts.ct(&case.out))?;
            res.push((label, b, tt));
        }
        Kind::RotateSum(st) => {
            let mut o = acquire(pool, k)?;
            rotate::rotate_sum(ctx, ex, kc, sh, groups.group(st).ok_or("no group")?, &x, st, o.get_mut().ok_or("shared")?)?;
            let (b, tt) = diff(&o, &ex.compute, &cts.ct(&case.out))?;
            res.push((format!("{label} {} steps", st.len()), b, tt));
        }
        Kind::Hoisted(st, outs) => {
            let nz: Vec<i64> = st.iter().copied().filter(|&v| rotation_galois(v, ctx.lay.n).map(|g| g != 1).unwrap_or(false)).collect();
            let grp = groups.group(&nz).ok_or("no group")?;
            let mut os: Vec<PoolCt<'_, '_, W>> = (0..st.len()).map(|_| acquire(pool, k)).collect::<Res<_>>()?;
            {
                let mut refs: Vec<&mut Ct<'_, W>> = os.iter_mut().map(|o| o.get_mut().ok_or("shared")).collect::<std::result::Result<_, _>>()?;
                rotate::rotate_hoisted(ctx, ex, kc, sh, grp, &x, st, &mut refs)?;
            }
            for (j, ov) in outs.iter().enumerate() {
                if let Some(ov) = ov {
                    let (b, tt) = diff(&os[j], &ex.compute, &cts.ct(ov))?;
                    res.push((format!("{} -> {ov} rot {} ({} hoisted)", case.ins[0], st[j], st.len()), b, tt));
                }
            }
        }
        Kind::Ladder { start, stop, groups: gr } => {
            let mut o = acquire(pool, k)?;
            let mut tmp = acquire(pool, k)?;
            rotate::rotate_and_sum(ctx, ex, kc, sh, groups, &x, *start, *stop, 3, o.get_mut().ok_or("shared")?, tmp.get_mut().ok_or("shared")?)?;
            let (b, tt) = diff(&o, &ex.compute, &cts.ct(&case.out))?;
            res.push((format!("{label} ladder({start},{stop}) {} groups", gr.len()), b, tt));
        }
        Kind::Mult | Kind::Square => {
            let key = relin_key.ok_or("no relin key")?;
            let sq = matches!(case.kind, Kind::Square);
            let hb = if sq { hin.clone_ct() } else { cts.ct(&case.ins[1]) };
            // the operands replicated K times; the ones the FIDESlib wrapper rescales in place are realized first,
            // then the product runs its own Recipe preludes (pc_gpu::ops::Ops::mult)
            let mut aa = acquire(pool, k)?;
            let mut bb = acquire(pool, k)?;
            upload(aa.get_mut().ok_or("shared")?, &ex.compute, &hin, nq)?;
            upload(bb.get_mut().ok_or("shared")?, &ex.compute, &hb, nq)?;
            for (i, c) in [&mut aa, &mut bb].into_iter().enumerate() {
                let c = c.get_mut().ok_or("shared")?;
                if case.realized.get(i).copied().unwrap_or(false) && c.meta.deg == 2 {
                    ops.realize(ctx, ex, sh, std::slice::from_mut(c))?;
                }
            }
            let mut o = acquire(pool, k)?;
            let npre = if sq { sh.square(&aa.meta)? } else { sh.mult(&aa.meta, &bb.meta)? }.pre.len();
            let om = o.get_mut().ok_or("shared")?;
            if case.op == "mult" {
                if sq {
                    ops.square(ctx, ex, kc, sh, key, om, &aa)?;
                } else {
                    ops.mult(ctx, ex, kc, sh, key, om, &aa, &bb)?;
                }
            } else {
                let (top, recipe) = ops.tensor_for_relin(ctx, ex, sh, om, &aa, if sq { None } else { Some(&bb) })?;
                relin::relin_staged(ctx, ex, kc, key, om, top)?;
                om.meta = recipe.out;
            }
            let (b, tt) = diff(&o, &ex.compute, &cts.ct(&case.out))?;
            let re: Vec<&str> = case.realized.iter().zip(&case.ins).filter(|(r, _)| **r).map(|(_, v)| v.as_str()).collect();
            res.push((format!("{label} pre {npre} realized {re:?}"), b, tt));
        }
        Kind::Monomial => {
            let mut o = acquire(pool, k)?;
            {
                let om = o.get_mut().ok_or("shared")?;
                om.buf.buf.zero(&ex.compute)?;
                om.meta = x.meta;
                limbs::merge_monomial(ctx, ex, mono, om, &x)?;
            }
            let (b, tt) = diff(&o, &ex.compute, &cts.ct(&case.out))?;
            res.push((format!("{label} (0 + X^N/2 x)"), b, tt));
        }
        Kind::Limbs => {
            let coef = hin.coef.as_ref().ok_or("no coefficient dump")?;
            let l = limbs::export_limbs(ctx, ex, &x, k - 1)?;
            let mut bad = 0;
            let mut tot = 0;
            for part in 0..2 {
                let flat: Vec<W> = l.c[part].iter().flatten().copied().collect();
                bad += flat.iter().zip(&coef[part]).filter(|(a, b)| a != b).count() + flat.len().abs_diff(coef[part].len());
                tot += coef[part].len();
            }
            res.push((format!("export {}", case.ins[0]), bad, tot));
            let mut y = acquire(pool, k)?;
            {
                let ym = y.get_mut().ok_or("shared")?;
                ym.meta = x.meta;
                limbs::import_limbs(ctx, ex, ym, 0, &l)?;
            }
            let ids: Vec<usize> = (0..=hin.level).collect();
            let mut b2 = 0;
            for part in 0..2 {
                let got = y.buf.download_rows(&ex.compute, 0, part, &ids)?;
                b2 += got.iter().zip(&hin.c[part]).filter(|(a, b)| a != b).count();
            }
            res.push((format!("import(export {}) == x", case.ins[0]), b2, 2 * hin.c[0].len()));
        }
    }
    Ok(res)
}

trait CloneCt {
    fn clone_ct(&self) -> HostCt;
}
impl CloneCt for HostCt {
    fn clone_ct(&self) -> HostCt {
        HostCt {
            level: self.level,
            noise_level: self.noise_level,
            noise_factor: self.noise_factor,
            slots: self.slots,
            ids: self.ids.clone(),
            pids: self.pids.clone(),
            c: self.c.clone(),
            p: self.p.clone(),
            coef: self.coef.clone(),
        }
    }
}
