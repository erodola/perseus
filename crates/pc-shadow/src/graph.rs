//! graph.json (v1) loader: the captured op list of one block, in the planner's topological order.
//! A port of `perseus/plan/placer/ir.py` (`Graph.from_nodes`, literals, level pins, depth costs) and
//! `place.py::erase_reactive_bootstraps`, so node indices match the planner's exactly. v2 fields (`args`,
//! full `tag`, `shadow`) are additive and land as `#[serde(default)]` fields of `RawNode`.
use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};
use std::path::Path;

pub const MULT_FAMILY: [&str; 4] = ["mult", "mult_inplace", "square", "square_inplace"];
pub const ADD_FAMILY: [&str; 6] = ["add", "add_inplace", "sub", "sub_inplace", "sub_ct", "sub_inplace_ct"];

pub fn is_mult_family(op: &str) -> bool {
    MULT_FAMILY.contains(&op)
}
pub fn is_add_family(op: &str) -> bool {
    ADD_FAMILY.contains(&op)
}

/// Table depth costs in CKKS levels (ir.py DEPTH_COSTS).
fn depth_cost(op: &str) -> i64 {
    match op {
        "mult" | "mult_inplace" | "square" | "square_inplace" => 1,
        _ => 0,
    }
}

pub fn is_literal_input(name: &str) -> bool {
    name.starts_with("rot(")
        || name.starts_with("const(")
        || name == "ct_null"
        || name == "pt_null"
        || name.starts_with("hint_lev(")
        || name.starts_with("lvl_cap(")
}

pub fn is_plaintext_name(name: &str) -> bool {
    name.starts_with("pt_") || name.starts_with("mask.")
}

pub fn is_kv_cache_read(var: &str) -> bool {
    var.contains("cache.k") || var.contains("cache.v")
}

/// A Python float literal of the shape `-?\d+(\.\d+)?`.
fn parse_plain_number(s: &str) -> Option<f64> {
    let body = s.strip_prefix('-').unwrap_or(s);
    let (int, frac) = match body.split_once('.') {
        Some((i, f)) => (i, Some(f)),
        None => (body, None),
    };
    if int.is_empty() || !int.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    if let Some(f) = frac {
        if f.is_empty() || !f.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
    }
    s.parse().ok()
}

/// ir.py `parse_lvl_suffix`: the level pinned by a `-lvl=<n>` name suffix.
pub fn parse_lvl_suffix(name: &str) -> Option<f64> {
    let i = name.rfind("-lvl=")?;
    parse_plain_number(&name[i + 5..])
}

/// ir.py `_parse_token`: the value of the first `prefix(<float>)` input.
fn parse_token(inputs: &[String], prefix: &str) -> Option<f64> {
    for x in inputs {
        if x.starts_with(prefix) && x.ends_with(')') {
            if let Ok(v) = x[prefix.len()..x.len() - 1].trim().parse::<f64>() {
                return Some(v);
            }
        }
    }
    None
}

fn num(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        _ => None,
    }
}

/// One node as stored in graph.json (v1 fields; everything else is ignored).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RawNode {
    #[serde(default)]
    pub op_type: String,
    #[serde(default)]
    pub inputs: Vec<Value>,
    #[serde(default)]
    pub input_levels: Vec<Value>,
    #[serde(default)]
    pub output: Value,
    #[serde(default)]
    pub output_level: Value,
    #[serde(default)]
    pub output_noise_level: Value,
    #[serde(default)]
    pub step: Value,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RawGraph {
    #[serde(default)]
    pub version: Value,
    #[serde(default)]
    pub nodes: Vec<RawNode>,
}

fn s_of(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// One captured op with the planner's derived fields.
#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub idx: usize,
    pub op: String,
    pub inputs: Vec<String>,
    pub input_levels: Vec<Option<f64>>,
    pub output: String,
    pub output_level: Option<f64>,
    pub output_deg: Option<i64>,
    pub step: String,
    pub cipher_inputs: Vec<String>,
    pub hint_level: Option<f64>,
    pub lvl_cap: Option<f64>,
    pub cost: i64,
    pub cost_forced: bool,
}

impl Node {
    pub fn is_deliberate_bts(&self) -> bool {
        self.op == "deliberate_bootstrap"
    }
    pub fn is_auto_bts(&self) -> bool {
        self.op == "auto_bootstrap"
    }
}

#[derive(Clone, Debug)]
pub struct Graph {
    pub nodes: Vec<Node>,
    /// output var -> node index (the last producer in topological order).
    pub producer_of: HashMap<String, usize>,
    /// cipher vars consumed and never produced (sorted).
    pub inputs: Vec<String>,
    pub level_unit: i64,
}

#[derive(Clone, Debug)]
struct Draft {
    op: String,
    inputs: Vec<String>,
    input_levels: Vec<Option<f64>>,
    output: String,
    output_level: Option<f64>,
    output_deg: Option<i64>,
    step: String,
}

/// Loader options: the level unit and PLAN_DELIB_LANDING_REMAP (captured deliberate landings re-mapped).
#[derive(Clone, Debug, Default)]
pub struct LoadOpts {
    pub level_unit: i64,
    pub delib_landing_remap: BTreeMap<i64, f64>,
}

impl LoadOpts {
    /// PLAN_DELIB_LANDING_REMAP syntax: `old:new,...`.
    pub fn parse_remap(spec: &str) -> Result<BTreeMap<i64, f64>> {
        let mut m = BTreeMap::new();
        for tok in spec.split(',') {
            if let Some((a, b)) = tok.split_once(':') {
                let a: f64 = a.trim().parse().map_err(|_| Error::Graph(format!("bad remap key {a}")))?;
                let b: f64 = b.trim().parse().map_err(|_| Error::Graph(format!("bad remap value {b}")))?;
                m.insert(a as i64, b);
            }
        }
        Ok(m)
    }
}

impl Graph {
    pub fn load(path: impl AsRef<Path>, opts: &LoadOpts) -> Result<Graph> {
        let raw: RawGraph = serde_json::from_slice(&std::fs::read(path.as_ref())?)?;
        Graph::from_raw(&raw.nodes, opts)
    }

    /// `ir.py::Graph.from_nodes`.
    pub fn from_raw(raw: &[RawNode], opts: &LoadOpts) -> Result<Graph> {
        let drafts: Vec<Draft> = raw
            .iter()
            .map(|n| {
                let inputs: Vec<String> = n.inputs.iter().map(s_of).collect();
                let input_levels = (0..inputs.len())
                    .map(|i| n.input_levels.get(i).and_then(num).filter(|&x| x >= 0.0))
                    .collect();
                let ol = num(&n.output_level).filter(|&x| x >= 0.0);
                let ol = match ol {
                    Some(x) if n.op_type == "deliberate_bootstrap" && !opts.delib_landing_remap.is_empty() => {
                        // remap keys are compared as floats in Python; captured levels are integral
                        if x.fract() == 0.0 {
                            Some(*opts.delib_landing_remap.get(&(x as i64)).unwrap_or(&x))
                        } else {
                            Some(x)
                        }
                    }
                    other => other,
                };
                Draft {
                    op: n.op_type.clone(),
                    inputs,
                    input_levels,
                    output: s_of(&n.output),
                    output_level: ol,
                    output_deg: num(&n.output_noise_level).map(|x| x as i64),
                    step: s_of(&n.step),
                }
            })
            .collect();
        Graph::from_drafts(drafts, opts.level_unit)
    }

    fn from_drafts(mut drafts: Vec<Draft>, unit: i64) -> Result<Graph> {
        if drafts.is_empty() {
            return Err(Error::Graph("graph has no nodes".into()));
        }
        for d in drafts.iter_mut() {
            for i in 0..d.inputs.len() {
                if let Some(exp) = parse_lvl_suffix(&d.inputs[i]) {
                    if let Some(lv) = d.input_levels[i] {
                        if exp < lv {
                            return Err(Error::Graph(format!(
                                "input pin {} declares level {exp} below the captured {lv} (op {} -> {}): pins may only deepen",
                                d.inputs[i], d.op, d.output
                            )));
                        }
                    }
                    d.input_levels[i] = Some(exp);
                }
            }
        }
        let mut out_to_i: HashMap<&str, usize> = HashMap::new();
        for (i, d) in drafts.iter().enumerate() {
            if !d.output.is_empty() {
                out_to_i.insert(d.output.as_str(), i);
            }
        }
        let n = drafts.len();
        let mut indeg = vec![0usize; n];
        let mut deps: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut cipher_ins_of: Vec<Vec<String>> = Vec::with_capacity(n);
        for (i, d) in drafts.iter().enumerate() {
            let cins: Vec<String> =
                d.inputs.iter().filter(|x| !is_literal_input(x) && !is_plaintext_name(x)).cloned().collect();
            for v in &cins {
                if let Some(&j) = out_to_i.get(v.as_str()) {
                    deps[j].push(i);
                    indeg[i] += 1;
                }
            }
            cipher_ins_of.push(cins);
        }
        let mut heap: BinaryHeap<Reverse<usize>> = (0..n).filter(|&i| indeg[i] == 0).map(Reverse).collect();
        let mut order = Vec::with_capacity(n);
        while let Some(Reverse(i)) = heap.pop() {
            order.push(i);
            let mut ds = deps[i].clone();
            ds.sort_unstable();
            for j in ds {
                indeg[j] -= 1;
                if indeg[j] == 0 {
                    heap.push(Reverse(j));
                }
            }
        }
        if order.len() != n {
            return Err(Error::Graph(format!("graph has a cycle ({} nodes unreachable)", n - order.len())));
        }
        let mut mult_lazy_out: HashMap<&str, f64> = HashMap::new();
        for d in &drafts {
            if is_mult_family(&d.op) && !d.output.is_empty() {
                if let Some(ol) = d.output_level {
                    mult_lazy_out.insert(d.output.as_str(), ol);
                }
            }
        }
        let mut nodes = Vec::with_capacity(n);
        for (pos, &i) in order.iter().enumerate() {
            let d = &drafts[i];
            let cins = &cipher_ins_of[i];
            let mut base = depth_cost(&d.op) * unit;
            let mut forced = false;
            if base == 0 && d.op != "auto_bootstrap" && d.op != "deliberate_bootstrap" {
                let known: Vec<f64> = d
                    .inputs
                    .iter()
                    .zip(&d.input_levels)
                    .filter(|(v, lv)| !is_literal_input(v) && !is_plaintext_name(v) && lv.is_some())
                    .map(|(_, lv)| lv.unwrap_or(0.0))
                    .collect();
                if !known.is_empty() && known.len() == cins.len() {
                    if let Some(ol) = d.output_level {
                        let max_in = known.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        if ol > max_in {
                            let lazy = d.inputs.iter().zip(&d.input_levels).any(|(v, lv)| {
                                mult_lazy_out.get(v.as_str()).is_some_and(|&m| *lv == Some(max_in) && m == max_in)
                            });
                            if !lazy {
                                base = if unit != 1 { (ol - max_in) as i64 } else { 1 };
                                forced = true;
                            }
                        }
                    }
                }
            }
            nodes.push(Node {
                idx: pos,
                op: d.op.clone(),
                inputs: d.inputs.clone(),
                input_levels: d.input_levels.clone(),
                output: d.output.clone(),
                output_level: d.output_level,
                output_deg: d.output_deg,
                step: d.step.clone(),
                cipher_inputs: cins.clone(),
                hint_level: parse_token(&d.inputs, "hint_lev("),
                lvl_cap: parse_token(&d.inputs, "lvl_cap("),
                cost: base,
                cost_forced: forced,
            });
        }
        let mut producer_of = HashMap::new();
        for nd in &nodes {
            if !nd.output.is_empty() {
                producer_of.insert(nd.output.clone(), nd.idx);
            }
        }
        let mut consumed: HashSet<&str> = HashSet::new();
        for nd in &nodes {
            for v in &nd.cipher_inputs {
                consumed.insert(v.as_str());
            }
        }
        let mut inputs: Vec<String> =
            consumed.into_iter().filter(|v| !producer_of.contains_key(*v)).map(str::to_string).collect();
        inputs.sort();
        Ok(Graph { nodes, producer_of, inputs, level_unit: unit })
    }

    /// `place.py::erase_reactive_bootstraps`: drop every auto_bootstrap and rewire its consumers to its input
    /// (input levels follow the resolved producer's captured output level).
    pub fn erase_reactive_bootstraps(&self, opts: &LoadOpts) -> Result<(Graph, usize)> {
        let mut remap: HashMap<&str, &str> = HashMap::new();
        let mut kept: Vec<&Node> = Vec::new();
        for nd in &self.nodes {
            if nd.is_auto_bts() && !nd.cipher_inputs.is_empty() && !nd.output.is_empty() {
                remap.insert(nd.output.as_str(), nd.cipher_inputs[0].as_str());
                continue;
            }
            kept.push(nd);
        }
        let resolve = |v: &str| -> String {
            let mut seen: HashSet<&str> = HashSet::new();
            let mut v = v;
            while let Some(&nx) = remap.get(v) {
                if !seen.insert(v) {
                    break;
                }
                v = nx;
            }
            v.to_string()
        };
        let mut out_level_of: HashMap<&str, Option<f64>> = HashMap::new();
        for nd in &kept {
            if !nd.output.is_empty() {
                out_level_of.insert(nd.output.as_str(), nd.output_level);
            }
        }
        let mut drafts = Vec::with_capacity(kept.len());
        for nd in &kept {
            let mut new_in = Vec::with_capacity(nd.inputs.len());
            let mut new_lv = Vec::with_capacity(nd.inputs.len());
            for (x, lv) in nd.inputs.iter().zip(&nd.input_levels) {
                if !is_literal_input(x) && !is_plaintext_name(x) {
                    let r = resolve(x);
                    let rl = out_level_of.get(r.as_str()).copied().flatten();
                    new_lv.push(if &r != x && rl.is_some() { rl } else { *lv });
                    new_in.push(r);
                } else {
                    new_in.push(x.clone());
                    new_lv.push(*lv);
                }
            }
            // the raw round trip writes the (already remapped) output level back; from_nodes remaps it again
            let ol = match nd.output_level {
                Some(x) if nd.is_deliberate_bts() && !opts.delib_landing_remap.is_empty() && x.fract() == 0.0 => {
                    Some(*opts.delib_landing_remap.get(&(x as i64)).unwrap_or(&x))
                }
                other => other,
            };
            drafts.push(Draft {
                op: nd.op.clone(),
                inputs: new_in,
                input_levels: new_lv,
                output: nd.output.clone(),
                output_level: ol,
                output_deg: nd.output_deg,
                step: nd.step.clone(),
            });
        }
        let g = Graph::from_drafts(drafts, self.level_unit)?;
        Ok((g, remap.len()))
    }
}
