//! The planner's simulation (`perseus/plan/placer/sim.py::simulate`) on top of the `Shadow` rules.
//!
//! `Shadow::step` maps one captured graph node to its rule (products -> `mult`/`square`/`mult_pt`/`mult_scalar`,
//! sums -> `add`/`sub`/`add_pt`/`add_scalar`, `negate`, `level_reduce`, hints, refresh landings, pass-through ops)
//! and is what `sim.py` calls. `simulate` is the planner bookkeeping around it (seeds, refreshed
//! overrides, realize anchors, the per-node budget), ported line for line so node results are identical; the
//! planner runs the C++ rules (`Quirks::planner()` is empty).
use crate::error::{Error, Result};
use crate::graph::{is_add_family, is_literal_input, is_mult_family, is_plaintext_name, Graph, Node};
use crate::meta::{effective_level, CtMeta};
use crate::quirks::Quirk;
use crate::rules::{PtMeta, Shadow};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Per-node context the planner supplies for the ops whose outcome is a plan decision.
#[derive(Clone, Debug, Default)]
pub struct StepArgs {
    /// Hint: the plan's pinned decision (overrides the threshold).
    pub hint_force: Option<bool>,
    /// Hint: the output is a placed refresh site (fires whatever the threshold says).
    pub placed: bool,
    /// Hint: where a fired refresh lands (the refresh spec); None = the sim's default.
    pub landing: Option<(u32, u8)>,
    /// Hint: the site is sparse-routed (the dense StC-first fallback does not apply).
    pub sparse: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StepOut {
    pub out: CtMeta,
    pub hint_fired: Option<bool>,
}

/// StC-first fallback: (deepest effective input level, landing level, landing degree).
pub type DenseFallback = (u32, u32, u8);

impl Shadow {
    /// One captured node through the rules (planner mode reads the captured levels where the rule needs a target
    /// the v1 graph does not record). `ins` holds the state of `node.cipher_inputs`, in order.
    pub fn step(&self, node: &Node, ins: &[CtMeta], args: &StepArgs, dense_fallback: Option<DenseFallback>) -> Result<StepOut> {
        let bl = self.cfg.bts.bootstrap_level;
        let u = self.unit();
        let fresh = CtMeta::new(bl, 1);
        if let Some(thr) = node.hint_level {
            let a = ins.first().copied().unwrap_or(fresh);
            let lv = a.level;
            let mut fired = (lv as f64) > thr;
            if let Some(f) = args.hint_force {
                fired = f;
            }
            if fired || args.placed {
                let (mut level, mut deg) = match args.landing {
                    Some(x) => x,
                    None => {
                        let l = self.bootstrap(&a, &crate::route::Route::dense())?.out;
                        (l.level, l.deg)
                    }
                };
                if let Some((entry, fl, fd)) = dense_fallback {
                    if !args.sparse && a.effective_level(u) > entry {
                        (level, deg) = (fl, fd);
                    }
                }
                return Ok(StepOut { out: CtMeta { level, deg, ..a }, hint_fired: Some(fired) });
            }
            return Ok(StepOut { out: a, hint_fired: Some(false) });
        }
        if ins.is_empty() {
            if node.op == "level_reduce" {
                let level = node.output_level.map_or(Ok(bl), level_of)?;
                return Ok(StepOut { out: CtMeta::new(level, node.output_deg.unwrap_or(1) as u8), hint_fired: None });
            }
            if node.op == "fold_bootstrap" {
                let level = node.output_level.map_or(Ok(bl + u), level_of)?;
                return Ok(StepOut { out: CtMeta::new(level, 2), hint_fired: None });
            }
            return Err(Error::Rule { op: "step", msg: format!("node {} ({}) has no cipher input", node.idx, node.op) });
        }
        let a = ins[0];
        let has_pt = node.inputs.iter().any(|x| is_plaintext_name(x));
        let out = if is_mult_family(&node.op) {
            if node.op.starts_with("square") {
                self.square(&a)?.out
            } else if ins.len() >= 2 {
                let mut acc = self.mult(&a, &ins[1])?.out;
                for b in &ins[2..] {
                    // sim.py: max effective level over every cipher input
                    let e = acc.level.max(b.effective_level(u));
                    acc = CtMeta { level: e, deg: 2, ..acc };
                }
                acc
            } else if has_pt {
                let pt = self.pt_input(node, &a);
                self.mult_pt(&a, &pt)?.out
            } else {
                let c = node.inputs.iter().find_map(|x| const_value(x)).unwrap_or(1.0);
                self.mult_scalar(&a, c)?.out
            }
        } else if is_add_family(&node.op) {
            let sub = node.op.starts_with("sub");
            if ins.len() >= 2 {
                if ins.len() > 2 {
                    return Err(Error::Rule { op: "step", msg: format!("{} with {} cipher inputs", node.op, ins.len()) });
                }
                if sub {
                    self.sub(&a, &ins[1])?.out
                } else {
                    self.add(&a, &ins[1])?.out
                }
            } else if has_pt {
                if u == 1 {
                    self.add_pt_unit1(node, &a)
                } else {
                    let pt = self.pt_input(node, &a);
                    self.add_pt(&a, &pt)?.out
                }
            } else {
                // ct +- scalar: no level change
                a
            }
        } else if node.op == "negate" || node.op == "negate_inplace" {
            self.negate(&a)?.out
        } else if node.op == "level_reduce" {
            let target = match node.output_level {
                Some(l) => level_of(l)?,
                None => a.level,
            };
            self.drop_to_level(&a, target).out
        } else if node.op == "fold_bootstrap" {
            let level = node.output_level.map_or(Ok(bl + u), level_of)?;
            CtMeta { level, deg: 2, ..a }
        } else {
            // rotate, rotate_sum, conjugate, mult_i, clone, level_hint, ...: level and degree pass through
            a
        };
        Ok(StepOut { out, hint_fired: None })
    }

    /// The plaintext operand of a ct-pt node. Under `PtLevelIgnored` the planner never reads it (level 0 = never
    /// binding); otherwise the level the compile encodes it at: a `pt_` / `mask.` plaintext is re-encoded like the
    /// ciphertext it meets (its effective level, degree 1), whatever level the capture recorded.
    fn pt_input(&self, node: &Node, a: &CtMeta) -> PtMeta {
        let level = if self.has(Quirk::PtLevelIgnored) || !node.inputs.iter().any(|x| is_plaintext_name(x)) {
            0
        } else {
            a.effective_level(self.unit())
        };
        PtMeta { level, deg: 1, nf: 0.0, tag: Default::default() }
    }

    /// sim.py's unit-1 chain rule for ct + pt: a plaintext captured deeper than the ciphertext realizes it.
    fn add_pt_unit1(&self, node: &Node, a: &CtMeta) -> CtMeta {
        let (mut ct_cap, mut pt_cap): (Option<f64>, Option<f64>) = (None, None);
        for (x, lv) in node.inputs.iter().zip(&node.input_levels) {
            let Some(lv) = *lv else { continue };
            if is_plaintext_name(x) {
                pt_cap = Some(pt_cap.map_or(lv, |p| p.max(lv)));
            } else if !is_literal_input(x) {
                ct_cap = Some(ct_cap.map_or(lv, |c| c.max(lv)));
            }
        }
        match (pt_cap, ct_cap) {
            (Some(p), Some(c)) if p > c => CtMeta { level: a.effective_level(self.unit()), deg: 1, ..*a },
            _ => *a,
        }
    }
}

fn const_value(x: &str) -> Option<f64> {
    x.strip_prefix("const(")?.strip_suffix(')')?.parse().ok()
}

fn level_of(x: f64) -> Result<u32> {
    if x < 0.0 || x.fract() != 0.0 {
        return Err(Error::Range { what: "level", msg: format!("captured level {x} is not a level") });
    }
    Ok(x as u32)
}

/// Everything `sim.py::simulate` takes besides the graph, in absolute levels.
#[derive(Clone, Debug, Default)]
pub struct SimInputs {
    /// Budget L = max_level - bootstrap_level, and the guard margin (one prime less on guarded/terminal nodes).
    pub budget_l: f64,
    pub guard_margin: bool,
    /// (level, deg) of every graph input.
    pub seeds: HashMap<String, (u32, u8)>,
    /// Landing (level, deg) of every refreshed var (placed sites and hint outputs).
    pub refreshed: HashMap<String, (u32, u8)>,
    pub sparse_refreshed: HashSet<String>,
    pub placed: HashSet<String>,
    pub realized: HashSet<String>,
    pub hint_force: HashMap<String, bool>,
    pub deliberate_clamp0: bool,
    /// PLAN_DELIB_DEG=captured.
    pub delib_deg_captured: bool,
    /// PLAN_STC_FIRST_KV_LANDING (absolute level, deg).
    pub stc_first_kv_landing: Option<(u32, u8)>,
    pub dense_fallback: Option<DenseFallback>,
    /// Per-step captured bootstrap landing offsets; None = derived from the graph (`step_bts_offsets`).
    pub step_offsets: Option<HashMap<String, f64>>,
}

#[derive(Clone, Debug, Default)]
pub struct SimResult {
    /// Final state of every var (inputs, outputs after overrides).
    pub state: HashMap<String, CtMeta>,
    /// Each node's rule output before any refresh override (consumed depth, as sim.py's node_out), and degree.
    pub node_out: BTreeMap<usize, f64>,
    pub node_deg: BTreeMap<usize, u8>,
    pub hint_fired: BTreeMap<usize, bool>,
    pub over_budget: Vec<(usize, f64)>,
}

impl SimResult {
    pub fn consumed(&self, v: &str, bootstrap_level: u32) -> Option<f64> {
        self.state.get(v).map(|m| m.level as f64 - bootstrap_level as f64)
    }
}

/// sim.py `step_bts_offsets`.
pub fn step_bts_offsets(g: &Graph, bootstrap_level: u32) -> HashMap<String, f64> {
    let mut out: HashMap<String, f64> = HashMap::new();
    for n in &g.nodes {
        if (n.is_deliberate_bts() || n.is_auto_bts()) && n.output_level.is_some() {
            let v = n.output_level.unwrap_or(0.0) - bootstrap_level as f64;
            let e = out.entry(n.step.clone()).or_insert(0.0);
            *e = e.max(v);
        }
    }
    out
}

const GUARDED: [&str; 10] =
    ["mult", "mult_inplace", "square", "square_inplace", "sub", "sub_inplace", "sub_ct", "sub_inplace_ct", "negate", "negate_inplace"];

/// `sim.py::simulate`.
pub fn simulate(shadow: &Shadow, g: &Graph, inp: &SimInputs) -> Result<SimResult> {
    let bl = shadow.cfg.bts.bootstrap_level;
    let blf = bl as f64;
    let unit = g.level_unit as u32;
    if unit != shadow.unit() {
        return Err(Error::Rule { op: "simulate", msg: format!("graph unit {unit} != chain unit {}", shadow.unit()) });
    }
    let derived;
    let offs = match &inp.step_offsets {
        Some(o) => o,
        None => {
            derived = step_bts_offsets(g, bl);
            &derived
        }
    };
    let fresh = CtMeta::new(bl, 1);
    let mut st: HashMap<String, CtMeta> = HashMap::new();
    for v in &g.inputs {
        let (l, d) = inp.seeds.get(v).copied().ok_or_else(|| Error::Rule { op: "simulate", msg: format!("no seed for input {v}") })?;
        st.insert(v.clone(), CtMeta::new(l, d));
    }
    for (v, &(l, d)) in &inp.refreshed {
        st.insert(v.clone(), CtMeta::new(l, d));
    }
    let terminal = g.nodes.iter().rev().find(|n| !n.output.is_empty() && !n.is_deliberate_bts()).map(|n| n.idx);
    let mut res = SimResult::default();
    for n in &g.nodes {
        if n.is_deliberate_bts() {
            if !n.output.is_empty() {
                let mut base = match n.output_level {
                    Some(ol) => ol - blf,
                    None => offs.get(&n.step).copied().unwrap_or(0.0).max(0.0),
                };
                if inp.deliberate_clamp0 && base + (unit as f64) < 0.0 {
                    base = 0.0;
                }
                let mut deg = match (inp.delib_deg_captured, n.output_deg) {
                    (true, Some(d)) => d as u8,
                    _ => n.output_deg.unwrap_or(2) as u8,
                };
                let mut level = level_of(blf + base)?;
                if let Some((l, d)) = inp.stc_first_kv_landing {
                    (level, deg) = (l, d);
                }
                st.insert(n.output.clone(), CtMeta::new(level, deg));
            }
            continue;
        }
        let ins: Vec<CtMeta> = n.cipher_inputs.iter().map(|v| st.get(v).copied().unwrap_or(fresh)).collect();
        let args = StepArgs {
            hint_force: inp.hint_force.get(&n.output).copied(),
            placed: inp.placed.contains(&n.output),
            landing: inp.refreshed.get(&n.output).copied(),
            sparse: inp.sparse_refreshed.contains(&n.output),
        };
        let so = shadow.step(n, &ins, &args, inp.dense_fallback)?;
        if let Some(f) = so.hint_fired {
            res.hint_fired.insert(n.idx, f);
        }
        let (c_level, d) = (so.out.level, so.out.deg);
        let c = c_level as f64 - blf;
        if !n.output.is_empty() {
            let mut m = so.out;
            if n.hint_level.is_none() {
                if let Some(&(l, dd)) = inp.refreshed.get(&n.output) {
                    m = CtMeta::new(l, dd);
                    if let Some((entry, fl, fd)) = inp.dense_fallback {
                        if !inp.sparse_refreshed.contains(&n.output) && effective_level(c_level, d, unit) > entry {
                            m = CtMeta::new(fl, fd);
                        }
                    }
                }
            }
            if inp.realized.contains(&n.output) && m.deg == 2 {
                m.level += unit;
                m.deg = 1;
            }
            st.insert(n.output.clone(), m);
        }
        res.node_out.insert(n.idx, c);
        res.node_deg.insert(n.idx, d);
        // Budget.node_budget
        let mut b = inp.budget_l;
        if let Some(cap) = n.lvl_cap {
            b = b.min((cap - blf).max(0.0) + n.cost as f64);
        }
        let refreshed_out = !n.output.is_empty() && inp.refreshed.contains_key(&n.output);
        if inp.guard_margin && !refreshed_out && (GUARDED.contains(&n.op.as_str()) || Some(n.idx) == terminal) {
            b = b.min(inp.budget_l - 1.0);
        }
        if c > b {
            res.over_budget.push((n.idx, c - b));
        }
    }
    res.state = st;
    Ok(res)
}
