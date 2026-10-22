//! `bind`: a compiled `Program` on a device runtime, once, at setup (before `Arena::freeze`).
//!
//! * every `VarId` -> its compiler-assigned slot, one full-layout ciphertext buffer each (`CtBuf::words(lay, 1)`),
//!   plus one scratch ciphertext for the in-place ops without an in-place kernel;
//! * every `PtRef` -> a `PtStore` key `(name, level, Eval)` (`resolve_pts` -> device row addresses);
//! * every rotation step list / conjugation / relinearization -> a setup-built `KeyGroup`;
//! * every refresh site -> its `Route`; every scalar -> an exact `f64`;
//! * every streamed weight row (`PtForm::CoeffStream`) -> its row in the block's weight frame (`StreamLayout`);
//! * the per-op argument table (`Step`): kind, slot indices, the compiled output metadata.
//!
//! Batched ops are bound member by member (separate slots). Details in docs/ARCHITECTURE.md "pc-program bind".
use crate::error::{Error, Result};
use crate::program::{Kind, Op, Origin, Program, Scalar};
use crate::replay::Routes;
use pc_cuda::{Arena, Compute, Stream};
use pc_gpu::bts::{BtsKeyRing, RouteKind};
use pc_gpu::encode_dev::PtForm as DevForm;
use pc_gpu::ks::{KeyGroup, KsKeyRef};
use pc_gpu::pt_store::{PtKey, PtStore};
use pc_gpu::{Ct, CtBuf, DevStrategy, DevWord, Layout};
use pc_shadow::{CtMeta, PtMeta, Route};
use std::collections::BTreeMap;
use std::ffi::c_void;

/// No operand.
pub const NONE: u32 = u32::MAX;

/// What one step runs (indices into the `Bound` tables).
#[derive(Clone, Debug, PartialEq)]
pub enum StepOp {
    Add,
    Sub,
    AddPt(u32),
    MultPt(u32),
    AddConst(f64),
    MultConst(f64),
    Mult,
    Square,
    Negate,
    /// Rotation by `k` through key group `g`.
    Rotate { k: i64, g: u32 },
    /// `a + sum rot(a, s)` over step list `l` (`Bound::step_lists`) through key group `g`.
    RotateSum { l: u32, g: u32 },
    Conjugate { g: u32 },
    MultI,
    Rescale,
    DropTo(u32),
    /// Refresh site `site` of the program on `route` (family `kind`).
    Bootstrap { site: u32, route: Route, kind: RouteKind },
    /// External finish `f` (`Bound::finishes`): a PCMM / CCMM op with several inputs and outputs, run by the
    /// runtime's `replay::FinishExec`.
    Finish { f: u32 },
    /// Streamed-weight sum `c` (`Bound::coeff_sums`), run by the runtime's `CoeffRowsDevice` on the rows of the
    /// block's weight frame (`Bound::stream_base`).
    CoeffSum { c: u32 },
}

/// A `Kind::CoeffSum` as bound: the term ciphertexts' slots and their rows' offsets in the weight frame (rows of N
/// floats).
#[derive(Clone, Debug, PartialEq)]
pub struct CoeffSumStep {
    pub ins: Vec<u32>,
    pub rows: Vec<u32>,
}

/// The weight frame of a bound block: every linear it streams (first-use order) at a row offset, whole (`rows` =
/// the linear's row count, whether or not the block reads every row); `total` rows.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StreamLayout {
    pub linears: Vec<(String, u32, u32)>,
    pub total: u32,
}

/// Row counts of the streamed linears the runtime holds (`<linear>` of the `<linear>#<row>` plaintext names).
pub trait StreamSource {
    fn rows(&self, linear: &str) -> Option<usize>;
}

/// `<linear>#<row>` -> (linear, row).
pub fn split_stream(name: &str) -> Option<(&str, u32)> {
    name.rsplit_once('#').and_then(|(l, k)| k.parse::<u32>().ok().map(|k| (l, k)))
}

/// The weight frame of `p`: its streamed linears in first-use order, back to back, whole. Err = the linears `src`
/// does not hold.
pub fn stream_layout(p: &Program, src: &dyn StreamSource) -> std::result::Result<StreamLayout, Vec<String>> {
    let mut out = StreamLayout::default();
    let mut missing: Vec<String> = vec![];
    for op in &p.ops {
        for m in op.members() {
            let Kind::CoeffSum { pts, .. } = &m.kind else { continue };
            for &id in pts {
                let Some((lin, _)) = p.pt_refs.get(id as usize).and_then(|r| split_stream(&r.name)) else { continue };
                if out.linears.iter().any(|x| x.0 == lin) || missing.iter().any(|x| x == lin) {
                    continue;
                }
                match src.rows(lin) {
                    Some(n) => {
                        out.linears.push((lin.to_string(), out.total, n as u32));
                        out.total += n as u32;
                    }
                    None => missing.push(lin.to_string()),
                }
            }
        }
    }
    if missing.is_empty() {
        Ok(out)
    } else {
        Err(missing)
    }
}

/// Limbs (top + 1) of the deepest-reaching streamed-weight sum of `p` (the expander's scratch rows), 0 = none.
pub fn coeff_rows_needed(p: &Program, shape: pc_shadow::ChainShape) -> usize {
    let mut r = 0;
    for op in &p.ops {
        for m in op.members() {
            if let Kind::CoeffSum { .. } = m.kind {
                if let Some(&v) = m.ins.first() {
                    if let Ok(t) = pc_shadow::meta::top_limb(shape, p.vars[v as usize].meta.level) {
                        r = r.max(t as usize + 1);
                    }
                }
            }
        }
    }
    r
}

/// A `Kind::Finish` as bound: input and output slots (op order: output m = the descriptor's output m), the
/// compiled landing of every output, the descriptor (v2 `args.external` JSON, or a calibration site name).
#[derive(Clone, Debug, PartialEq)]
pub struct FinishStep {
    pub ins: Vec<u32>,
    pub outs: Vec<u32>,
    pub out_metas: Vec<CtMeta>,
    pub desc: String,
}

/// One launch member: op, slots, the compiled output metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct Step {
    pub op: StepOp,
    pub out: u32,
    pub a: u32,
    pub b: u32,
    /// The output takes `a`'s slot.
    pub inplace: bool,
    /// Output var (diagnostics) and its compiled metadata.
    pub var: u32,
    pub meta: CtMeta,
    /// Compiled metadata of the operands in `a` / `b` (a dynamic step issued between two graph launches reads
    /// them: a launch does not move the host-side metadata of the slots it wrote).
    pub a_meta: Option<CtMeta>,
    pub b_meta: Option<CtMeta>,
}

/// Ciphertext slots shared by every bound block of a runtime (`Bound::new_shared`): blocks run one after the other
/// on one lane, so one slab of the largest block's slot count serves them all (`Bound::with_slab`); what crosses a
/// block boundary (the exit, KV states) is copied out by the driver.
pub struct Slab<'a, W: DevWord> {
    pub slots: Vec<Ct<'a, W>>,
    pub tmp: Option<Ct<'a, W>>,
}

impl<'a, W: DevWord> Slab<'a, W> {
    pub fn bytes(lay: &Layout, n: usize) -> usize {
        (n + 1) * Arena::bytes_for::<W>(CtBuf::<Compute, W>::words(lay, 1))
    }
    pub fn new(arena: &'a Arena, lay: Layout, n: usize) -> Result<Self> {
        let slots = (0..n).map(|_| Ok(Ct::new(CtMeta::new(0, 1), CtBuf::new(arena, "replay.slab", lay, 1)?))).collect::<Result<Vec<_>>>()?;
        Ok(Slab { slots, tmp: Some(Ct::new(CtMeta::new(0, 1), CtBuf::new(arena, "replay.slab", lay, 1)?)) })
    }
}

/// A bound plaintext operand: store key and, once resolved, its device rows.
#[derive(Clone, Debug)]
pub struct BoundPt {
    pub key: PtKey,
    pub meta: Option<PtMeta>,
    pub rows: *const c_void,
    /// A streamed weight row (read through the weight frame by a `CoeffSum`, never from the PtStore).
    pub stream: bool,
}

/// Exact scalars for the program's inexact (v1, 6-digit) literals, by graph node index (`Origin::Node`).
pub type ScalarTable = BTreeMap<u32, f64>;

/// What `bind` borrows from the runtime (setup phase: the arena is not frozen yet).
pub struct BindEnv<'e, 'a> {
    pub arena: &'a Arena,
    pub stream: &'e Stream<Compute>,
    pub lay: Layout,
    pub keys: &'e dyn BtsKeyRing,
    /// `Executor::ks.stage_ptrs()` (staging buffers for keys that need them).
    pub stage: Option<(*const c_void, *const c_void)>,
    pub scalars: &'e ScalarTable,
    /// The streamed linears' row counts (a program with a `CoeffSum` needs it).
    pub streams: Option<&'e dyn StreamSource>,
}

/// A program bound to one runtime: ciphertext slots, plaintext keys, key groups, routes, the step table, and its
/// captured graph (replay.rs). One `Bound` per (block, variant).
pub struct Bound<'a, W: DevWord> {
    pub label: String,
    /// Own slots (`Bound::new`), or empty for a block bound on a shared `Slab` (swapped in by `with_slab`).
    pub slots: Vec<Ct<'a, W>>,
    /// Scratch for in-place ops without an in-place kernel (copy, then out of place); the slab's while swapped in.
    pub tmp: Option<Ct<'a, W>>,
    /// Slots the program needs (`Program.mem`).
    pub n_slots: usize,
    /// Steps whose arguments the driver changes between runs (`set_scalar` / `set_rotation`): issued eagerly, the
    /// static segments between them replayed as graphs.
    pub dynamic: std::collections::BTreeSet<u32>,
    /// Key group of the rotations a dynamic rotate step may take (`Bound::new_shared(.., dyn_rot)`), and its steps.
    pub dyn_group: u32,
    pub dyn_steps: Vec<i64>,
    /// Per dynamic step: the operand and output metadata of its last eager issue (the compile leaves scales
    /// unknown; a capture and the replays between graph launches take these).
    pub dyn_metas: BTreeMap<u32, (Option<CtMeta>, Option<CtMeta>, CtMeta)>,
    pub steps: Vec<Step>,
    pub groups: Vec<KeyGroup<'a>>,
    /// Index of the relinearization group in `groups` (NONE when the block has no ct x ct product).
    pub relin: u32,
    pub step_lists: Vec<Vec<i64>>,
    pub pts: Vec<BoundPt>,
    /// External finishes (`StepOp::Finish`).
    pub finishes: Vec<FinishStep>,
    /// Streamed-weight sums (`StepOp::CoeffSum`) and the weight frame they read.
    pub coeff_sums: Vec<CoeffSumStep>,
    pub streams: StreamLayout,
    /// Device address of the frame holding this block's rows (`residency::StreamRing::adopt`), null until staged.
    pub stream_base: *const f32,
    /// (graph name, slot, metadata) of the block inputs, in `Program.inputs` order (the driver writes them).
    pub inputs: Vec<(String, u32, CtMeta)>,
    /// (graph name, slot, metadata) of `Program.outputs` (exit var first).
    pub outputs: Vec<(String, u32, CtMeta)>,
    /// Var names by id (diagnostics).
    pub names: Vec<String>,
    /// Ring dimension N (route families).
    pub ring_n: usize,
    /// Captured graphs of this block, one per plaintext-row layout it ran with (most recent last; a weight ring
    /// may place a block's rows in a few different slot sets over a token cycle).
    pub(crate) graphs: Vec<crate::replay::Captured>,
}

fn berr(block: &str, msg: impl Into<String>) -> Error {
    Error::Bind { block: block.to_string(), msg: msg.into() }
}

fn list<T: std::fmt::Display>(v: &[T]) -> String {
    let head: Vec<String> = v.iter().take(12).map(|x| x.to_string()).collect();
    if v.len() > 12 {
        format!("{} (+{} more)", head.join(", "), v.len() - 12)
    } else {
        head.join(", ")
    }
}

/// What a program's key groups are: relinearization, conjugation, rotation step lists (dedup, first use order).
fn key_needs(p: &Program) -> (bool, bool, Vec<Vec<i64>>) {
    let (mut relin, mut conj) = (false, false);
    let mut lists: Vec<Vec<i64>> = vec![];
    for op in &p.ops {
        for m in op.members() {
            match &m.kind {
                Kind::Mult | Kind::Square => relin = true,
                Kind::Conjugate => conj = true,
                Kind::Rotate(k) => {
                    if !lists.contains(&vec![*k]) {
                        lists.push(vec![*k]);
                    }
                }
                Kind::RotateSum(s) => {
                    if !lists.contains(s) {
                        lists.push(s.clone());
                    }
                }
                _ => {}
            }
        }
    }
    (relin, conj, lists)
}

impl<'a, W: DevWord> Bound<'a, W> {
    /// Arena bytes `bind` carves for `p` (MemPlan section "replay"): its slots + scratch, its key groups.
    pub fn bytes(p: &Program, lay: &Layout) -> usize {
        let ct = Arena::bytes_for::<W>(CtBuf::<Compute, W>::words(lay, 1));
        (p.mem.slot_bytes.len() + 1) * ct + Self::group_bytes(p, 0)
    }

    /// Arena bytes of `p`'s key groups (+ a dynamic-rotation group of `dyn_rot` steps): what `new_shared` carves.
    pub fn group_bytes(p: &Program, dyn_rot: usize) -> usize {
        let (relin, conj, lists) = key_needs(p);
        lists.iter().map(|l| KeyGroup::bytes(l.len().max(1), true)).sum::<usize>() + (relin as usize + conj as usize) * KeyGroup::bytes(1, true) + if dyn_rot > 0 { KeyGroup::bytes(dyn_rot, true) } else { 0 }
    }

    /// `new` without slots: the block runs on a shared `Slab` (`with_slab`). `dyn_rot`: rotation steps a dynamic
    /// rotate step may be set to (`set_rotation`; one extra key group).
    pub fn new_shared<M: DevStrategy<W>>(p: &Program, env: &BindEnv<'_, 'a>, routes: &Routes<'_, 'a, W, M>, dyn_rot: &[i64]) -> Result<Self> {
        let mut b = Self::build(p, env, routes, false)?;
        let half = (env.lay.n / 2) as i64;
        let steps: Vec<i64> = dyn_rot.iter().copied().filter(|s| s.rem_euclid(half) != 0).collect();
        if !steps.is_empty() {
            let mut refs: Vec<&dyn KsKeyRef> = vec![];
            let mut missing = vec![];
            for &s in &steps {
                match env.keys.rot(s) {
                    Ok(k) => refs.push(k),
                    Err(_) => missing.push(s),
                }
            }
            if !missing.is_empty() {
                return Err(berr(&b.label, format!("dynamic rotation key(s) missing from the key ring: {}", list(&missing))));
            }
            b.dyn_group = b.groups.len() as u32;
            b.groups.push(KeyGroup::new(env.arena, env.stream, env.lay.n, &refs, env.stage)?);
            b.dyn_steps = steps;
        }
        Ok(b)
    }

    /// Bind `p` (see the module doc). `routes` must hold every route family the program's refresh sites use;
    /// plaintexts are resolved later (`resolve_pts`), so a weight ring can stage them after setup.
    pub fn new<M: DevStrategy<W>>(p: &Program, env: &BindEnv<'_, 'a>, routes: &Routes<'_, 'a, W, M>) -> Result<Self> {
        Self::build(p, env, routes, true)
    }

    fn build<M: DevStrategy<W>>(p: &Program, env: &BindEnv<'_, 'a>, routes: &Routes<'_, 'a, W, M>, carve: bool) -> Result<Self> {
        let block = p.block.as_str();
        let lay = env.lay;
        let n_slots = p.mem.slot_bytes.len();
        let slot_of = |v: u32| -> Result<u32> {
            p.vars.get(v as usize).and_then(|x| x.slot).ok_or_else(|| berr(block, format!("var {v} has no slot (alloc not run?)")))
        };
        // key groups
        let (need_relin, need_conj, lists) = key_needs(p);
        let mut groups: Vec<KeyGroup<'a>> = vec![];
        let mut missing: Vec<String> = vec![];
        let mut relin = NONE;
        if need_relin {
            match env.keys.relin() {
                Ok(k) => {
                    groups.push(KeyGroup::new(env.arena, env.stream, lay.n, &[k], env.stage)?);
                    relin = 0;
                }
                Err(_) => missing.push("relin".into()),
            }
        }
        let mut conj = NONE;
        if need_conj {
            match env.keys.conj() {
                Ok(k) => {
                    conj = groups.len() as u32;
                    groups.push(KeyGroup::new(env.arena, env.stream, lay.n, &[k], env.stage)?);
                }
                Err(_) => missing.push("conj".into()),
            }
        }
        let half = (lay.n / 2) as i64;
        let mut list_group: Vec<u32> = vec![];
        for l in &lists {
            let mut refs: Vec<&dyn KsKeyRef> = vec![];
            for &s in l.iter().filter(|s| s.rem_euclid(half) != 0) {
                match env.keys.rot(s) {
                    Ok(k) => refs.push(k),
                    Err(_) => missing.push(format!("rot {s}")),
                }
            }
            if refs.is_empty() {
                // every step is 0 mod N/2: the identity (rotate copies, rotate_sum refuses at run time)
                list_group.push(NONE);
            } else {
                list_group.push(groups.len() as u32);
                groups.push(KeyGroup::new(env.arena, env.stream, lay.n, &refs, env.stage)?);
            }
        }
        missing.dedup();
        if !missing.is_empty() {
            return Err(berr(block, format!("{} key(s) missing from the key ring: {}", missing.len(), list(&missing))));
        }
        let list_idx = |l: &Vec<i64>| -> u32 { lists.iter().position(|x| x == l).unwrap_or(0) as u32 };
        // plaintext keys (resolved later)
        // a block's own plaintexts (`pt_N`, encoded at use) are per-block names: namespaced, so a weight ring can hold
        // several blocks; masks and named slot plaintexts keep their shared names
        let pts: Vec<BoundPt> = p
            .pt_refs
            .iter()
            .map(|r| {
                // a graph's `pt_N` is numbered per block whatever its form (v2 names cached masks `pt_N` too)
                let stream = r.form == crate::program::PtForm::CoeffStream;
                let name = if !stream && (r.form == crate::program::PtForm::AtUse || r.name.starts_with("pt_")) { format!("{block}/{}", r.name) } else { r.name.clone() };
                BoundPt { key: PtKey::new(name, r.level, DevForm::Eval), meta: None, rows: std::ptr::null(), stream }
            })
            .collect();
        // steps
        let mut steps = vec![];
        let mut no_scalar: Vec<String> = vec![];
        let mut no_route: Vec<String> = vec![];
        let mut alias: Vec<String> = vec![];
        let mut finishes: Vec<FinishStep> = vec![];
        let mut coeff_sums: Vec<CoeffSumStep> = vec![];
        let streams = match env.streams {
            Some(src) => stream_layout(p, src).map_err(|missing| berr(block, format!("{} streamed linear(s) the runtime does not hold (StreamSource): {}", missing.len(), list(&missing))))?,
            None if p.ops.iter().any(|o| o.members().iter().any(|m| matches!(m.kind, Kind::CoeffSum { .. }))) => {
                return Err(berr(block, "the program streams weights (coeff_sum) and the runtime has no StreamSource"));
            }
            None => StreamLayout::default(),
        };
        let scalar = |m: &Op, c: &Scalar, no_scalar: &mut Vec<String>| -> f64 {
            if c.exact {
                return c.value;
            }
            if let Origin::Node(i) = m.origin {
                if let Some(&v) = env.scalars.get(&i) {
                    // the table holds the graph's scalar; a `sub` of a constant is lowered to add(-c)
                    return if (v - c.value).abs() <= (v + c.value).abs() { v } else { -v };
                }
            }
            no_scalar.push(format!("{:?}: {}", m.origin, c.value));
            c.value
        };
        for op in &p.ops {
            for m in op.members() {
                let a = m.ins.first().map(|&v| slot_of(v)).transpose()?.unwrap_or(NONE);
                let b = m.ins.get(1).map(|&v| slot_of(v)).transpose()?.unwrap_or(NONE);
                if let Kind::Finish { desc, .. } = &m.kind {
                    let desc = desc.clone().ok_or_else(|| berr(block, format!("finish {:?} has no descriptor (v2 args.external)", m.origin)))?;
                    let ins = m.ins.iter().map(|&v| slot_of(v)).collect::<Result<Vec<_>>>()?;
                    let outs = m.outs.iter().map(|&v| slot_of(v)).collect::<Result<Vec<_>>>()?;
                    if outs.is_empty() {
                        return Err(berr(block, format!("finish {:?} without outputs", m.origin)));
                    }
                    let out_metas = m.outs.iter().map(|&v| p.vars[v as usize].meta).collect();
                    let var = m.outs[0];
                    finishes.push(FinishStep { ins: ins.clone(), outs: outs.clone(), out_metas, desc });
                    let vm = |k: usize| m.ins.get(k).map(|&v| p.vars[v as usize].meta);
                    // outputs may take dying inputs' slots: the service reads every input before writing an output
                    steps.push(Step { op: StepOp::Finish { f: finishes.len() as u32 - 1 }, out: outs[0], a: ins.first().copied().unwrap_or(NONE), b: NONE, inplace: false, var, meta: p.vars[var as usize].meta, a_meta: vm(0), b_meta: None });
                    continue;
                }
                if let Kind::CoeffSum { pts: ids, .. } = &m.kind {
                    let ins = m.ins.iter().map(|&v| slot_of(v)).collect::<Result<Vec<_>>>()?;
                    let var = *m.outs.first().ok_or_else(|| berr(block, format!("coeff_sum {:?} without an output", m.origin)))?;
                    let out = slot_of(var)?;
                    if ins.contains(&out) {
                        alias.push(format!("{:?}: coeff_sum output slot {out} is one of its terms", m.origin));
                    }
                    let mut rows = Vec::with_capacity(ids.len());
                    for &id in ids {
                        let r = p.pt_refs.get(id as usize).ok_or_else(|| berr(block, format!("plaintext {id} out of range")))?;
                        let (lin, row) = split_stream(&r.name).ok_or_else(|| berr(block, format!("stream plaintext {:?} is not <linear>#<row>", r.name)))?;
                        let (_, off, n) = streams.linears.iter().find(|x| x.0 == lin).ok_or_else(|| berr(block, format!("{lin} missing from the weight frame")))?;
                        if row >= *n {
                            return Err(berr(block, format!("{}: row {row} of a {n}-row streamed linear", r.name)));
                        }
                        rows.push(off + row);
                    }
                    coeff_sums.push(CoeffSumStep { ins: ins.clone(), rows });
                    let vm = |k: usize| m.ins.get(k).map(|&v| p.vars[v as usize].meta);
                    steps.push(Step { op: StepOp::CoeffSum { c: coeff_sums.len() as u32 - 1 }, out, a: ins[0], b: NONE, inplace: false, var, meta: p.vars[var as usize].meta, a_meta: vm(0), b_meta: None });
                    continue;
                }
                if m.outs.len() != 1 {
                    return Err(berr(block, format!("{} op with {} outputs ({:?}): PCMM/CCMM finishes replay through the matmul services only", m.kind.name(), m.outs.len(), m.origin)));
                }
                let var = m.outs[0];
                let out = slot_of(var)?;
                let sop = match &m.kind {
                    Kind::Add => StepOp::Add,
                    Kind::Sub => StepOp::Sub,
                    Kind::AddPt(i) => StepOp::AddPt(*i),
                    Kind::MultPt(i) => StepOp::MultPt(*i),
                    Kind::AddConst(c) => StepOp::AddConst(scalar(m, c, &mut no_scalar)),
                    Kind::MultConst(c) => StepOp::MultConst(scalar(m, c, &mut no_scalar)),
                    Kind::Mult => StepOp::Mult,
                    Kind::Square => StepOp::Square,
                    Kind::Negate => StepOp::Negate,
                    Kind::Rotate(k) => StepOp::Rotate { k: *k, g: list_group[list_idx(&vec![*k]) as usize] },
                    Kind::RotateSum(s) => StepOp::RotateSum { l: list_idx(s), g: list_group[list_idx(s) as usize] },
                    Kind::Conjugate => StepOp::Conjugate { g: conj },
                    Kind::MultI => StepOp::MultI,
                    Kind::Rescale => StepOp::Rescale,
                    Kind::DropTo(l) => StepOp::DropTo(*l),
                    Kind::Bootstrap(site) => {
                        let s = p.bts_sites.get(*site as usize).ok_or_else(|| berr(block, format!("bootstrap site {site} out of range")))?;
                        let kind = RouteKind::of(&s.route, lay.n);
                        if routes.get(kind).is_none() {
                            no_route.push(format!("{kind:?} (site {} on {})", site, s.var));
                        }
                        StepOp::Bootstrap { site: *site, route: s.route, kind }
                    }
                    Kind::Finish { .. } => return Err(berr(block, "finish reached the single-output lowering")),
                    Kind::CoeffSum { .. } => return Err(berr(block, "coeff_sum reached the single-output lowering")),
                    Kind::Batched(_) => return Err(berr(block, "nested batch")),
                };
                let inplace = m.inplace;
                if inplace && out != a {
                    alias.push(format!("{:?}: in place but out slot {out} != a slot {a}", m.origin));
                }
                if !inplace && a != NONE && out == a {
                    alias.push(format!("{:?}: out of place but out slot {out} == a slot", m.origin));
                }
                if b != NONE && out == b && !(inplace && a == b) {
                    alias.push(format!("{:?}: out slot {out} == b slot", m.origin));
                }
                let vm = |k: usize| m.ins.get(k).map(|&v| p.vars[v as usize].meta);
                steps.push(Step { op: sop, out, a, b, inplace, var, meta: p.vars[var as usize].meta, a_meta: vm(0), b_meta: vm(1) });
            }
        }
        if !no_scalar.is_empty() {
            return Err(berr(block, format!("{} inexact v1 scalar(s) with no exact value in the driver's table: {}", no_scalar.len(), list(&no_scalar))));
        }
        if !no_route.is_empty() {
            return Err(berr(block, format!("route(s) not held by the runtime: {}", list(&no_route))));
        }
        if !alias.is_empty() {
            return Err(berr(block, format!("slot aliasing the replay cannot run: {}", list(&alias))));
        }
        let (slots, tmp) = if carve {
            let slots = (0..n_slots).map(|_| Ok(Ct::new(CtMeta::new(0, 1), CtBuf::new(env.arena, "replay.slot", lay, 1)?))).collect::<Result<Vec<_>>>()?;
            (slots, Some(Ct::new(CtMeta::new(0, 1), CtBuf::new(env.arena, "replay.slot", lay, 1)?)))
        } else {
            (vec![], None)
        };
        let io = |vs: &[u32]| -> Result<Vec<(String, u32, CtMeta)>> { vs.iter().map(|&v| Ok((p.vars[v as usize].name.clone(), slot_of(v)?, p.vars[v as usize].meta))).collect() };
        Ok(Bound {
            label: block.to_string(),
            slots,
            tmp,
            n_slots,
            dynamic: Default::default(),
            dyn_group: NONE,
            dyn_steps: vec![],
            dyn_metas: BTreeMap::new(),
            steps,
            groups,
            relin,
            step_lists: lists,
            pts,
            finishes,
            coeff_sums,
            streams,
            stream_base: std::ptr::null(),
            inputs: io(&p.inputs)?,
            outputs: io(&p.outputs)?,
            names: p.vars.iter().map(|v| v.name.clone()).collect(),
            ring_n: lay.n,
            graphs: vec![],
        })
    }

    /// Resolve every plaintext key to its resident device rows (missing = one error listing them). Returns whether
    /// any address changed since the last call (a captured graph baked the old ones: replay re-captures).
    pub fn resolve_pts(&mut self, store: &PtStore<'_, '_, W>) -> Result<bool> {
        let mut missing: Vec<String> = vec![];
        let mut moved = false;
        for bp in self.pts.iter_mut().filter(|p| !p.stream) {
            match store.view(&bp.key) {
                Ok(pt) => {
                    moved |= pt.rows != bp.rows;
                    bp.rows = pt.rows;
                    bp.meta = Some(pt.meta);
                }
                Err(_) => missing.push(format!("{}@{}", bp.key.name, bp.key.level)),
            }
        }
        if !missing.is_empty() {
            missing.sort();
            missing.dedup();
            return Err(berr(&self.label, format!("{} plaintext(s) not resident in the PtStore: {}", missing.len(), list(&missing))));
        }
        Ok(moved)
    }

    /// The distinct plaintext keys of the block (staging list for the weight ring).
    pub fn pt_keys(&self) -> Vec<PtKey> {
        let mut v: Vec<PtKey> = vec![];
        for p in self.pts.iter().filter(|p| !p.stream) {
            if !v.contains(&p.key) {
                v.push(p.key.clone());
            }
        }
        v
    }

    /// The ciphertext a block input is written into (the driver encrypts / uploads here before `run`); its
    /// metadata is set to the compiled input metadata.
    pub fn input_mut(&mut self, i: usize) -> Result<&mut Ct<'a, W>> {
        let (_, s, m) = self.inputs.get(i).cloned().ok_or_else(|| berr(&self.label, format!("no input {i}")))?;
        let label = self.label.clone();
        let c = self.slots.get_mut(s as usize).ok_or_else(|| berr(&label, "slots not present (a shared block outside with_slab)"))?;
        c.meta = m;
        Ok(c)
    }

    /// Output `i` (0 = the exit var) after a run.
    pub fn output(&self, i: usize) -> Result<&Ct<'a, W>> {
        let (_, s, _) = self.outputs.get(i).ok_or_else(|| berr(&self.label, format!("no output {i}")))?;
        self.slots.get(*s as usize).ok_or_else(|| berr(&self.label, "slots not present (a shared block outside with_slab)"))
    }

    /// Re-route refresh site `site` (a driver decision the graph does not carry, e.g. a deliberate refresh of a
    /// real payload: perseus `bootstrap_real`). Same route family only: the runtime's route table was checked at
    /// bind. Drops nothing: call before the first run (a captured graph keeps the old route).
    pub fn set_site_route(&mut self, site: u32, route: Route) -> Result<usize> {
        let mut n = 0;
        for st in self.steps.iter_mut() {
            if let StepOp::Bootstrap { site: s, route: r, kind } = &mut st.op {
                if *s == site {
                    let k = RouteKind::of(&route, self.ring_n);
                    if k != *kind {
                        return Err(berr(&self.label, format!("site {site}: {route:?} is route family {k:?}, bound as {kind:?}")));
                    }
                    *r = route;
                    n += 1;
                }
            }
        }
        if !self.graphs.is_empty() {
            return Err(berr(&self.label, "set_site_route after a capture"));
        }
        Ok(n)
    }

    /// Captured graphs held.
    pub fn n_graphs(&self) -> usize {
        self.graphs.len()
    }

    /// Run `f` with `slab`'s slots as this block's slots (a block bound by `new_shared`); the slab gets them back
    /// whatever `f` returns.
    pub fn with_slab<T>(&mut self, slab: &mut Slab<'a, W>, f: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        if !self.slots.is_empty() || self.tmp.is_some() {
            return Err(berr(&self.label, "with_slab on a block with its own slots"));
        }
        if slab.slots.len() < self.n_slots || slab.tmp.is_none() {
            return Err(berr(&self.label, format!("slab of {} slots, the block needs {}", slab.slots.len(), self.n_slots)));
        }
        std::mem::swap(&mut self.slots, &mut slab.slots);
        std::mem::swap(&mut self.tmp, &mut slab.tmp);
        let r = f(self);
        std::mem::swap(&mut self.slots, &mut slab.slots);
        std::mem::swap(&mut self.tmp, &mut slab.tmp);
        r
    }

    /// The steps producing graph name `name` (a var `name` or its `~pre` / `~raw` states).
    fn steps_of(&self, name: &str) -> Vec<usize> {
        (0..self.steps.len()).filter(|&i| self.names[self.steps[i].var as usize].split('~').next() == Some(name)).collect()
    }

    fn make_dynamic(&mut self, i: usize) -> Result<()> {
        if !self.dynamic.contains(&(i as u32)) {
            if !self.graphs.is_empty() {
                return Err(berr(&self.label, format!("step {i} made dynamic after a graph capture (declare dynamic steps before the first run)")));
            }
            self.dynamic.insert(i as u32);
        }
        Ok(())
    }

    /// Set the scalar of the constant op producing `name` (a per-position constant); the step becomes dynamic.
    pub fn set_scalar(&mut self, name: &str, value: f64) -> Result<()> {
        let idx: Vec<usize> = self.steps_of(name).into_iter().filter(|&i| matches!(self.steps[i].op, StepOp::AddConst(_) | StepOp::MultConst(_))).collect();
        if idx.len() != 1 {
            return Err(berr(&self.label, format!("set_scalar {name}: {} constant steps produce it", idx.len())));
        }
        let i = idx[0];
        self.make_dynamic(i)?;
        match &mut self.steps[i].op {
            StepOp::AddConst(c) | StepOp::MultConst(c) => *c = value,
            _ => {}
        }
        Ok(())
    }

    /// Set the rotation of the rotate op producing `name` to `k` (0 mod N/2 = a copy; else one of the steps the
    /// block was bound with in `dyn_rot`); the step becomes dynamic.
    pub fn set_rotation(&mut self, name: &str, k: i64) -> Result<()> {
        let idx: Vec<usize> = self.steps_of(name).into_iter().filter(|&i| matches!(self.steps[i].op, StepOp::Rotate { .. })).collect();
        if idx.len() != 1 {
            return Err(berr(&self.label, format!("set_rotation {name}: {} rotate steps produce it", idx.len())));
        }
        let half = (self.ring_n / 2) as i64;
        let g = if k.rem_euclid(half) == 0 {
            NONE
        } else if self.dyn_steps.contains(&k) {
            self.dyn_group
        } else {
            return Err(berr(&self.label, format!("set_rotation {name}: step {k} is not among the dynamic rotation keys {:?}", self.dyn_steps)));
        };
        let i = idx[0];
        self.make_dynamic(i)?;
        self.steps[i].op = StepOp::Rotate { k, g };
        Ok(())
    }

    /// Declare the step producing `name` dynamic without changing it (its arguments will change later).
    pub fn mark_dynamic(&mut self, name: &str) -> Result<()> {
        let idx = self.steps_of(name);
        if idx.is_empty() {
            return Err(berr(&self.label, format!("mark_dynamic {name}: no step produces it")));
        }
        for i in idx {
            self.make_dynamic(i)?;
        }
        Ok(())
    }
}

impl Program {
    /// `Bound::new`.
    pub fn bind<'a, W: DevWord, M: DevStrategy<W>>(&self, env: &BindEnv<'_, 'a>, routes: &Routes<'_, 'a, W, M>, store: &PtStore<'_, '_, W>) -> Result<Bound<'a, W>> {
        let mut b = Bound::new(self, env, routes)?;
        b.resolve_pts(store)?;
        Ok(b)
    }
}
