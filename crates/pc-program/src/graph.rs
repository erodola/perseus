//! graph.json v1/v2 for the compiler: the planner's loader (`pc_shadow::graph`, a port of ir.py: topological order,
//! level pins, `erase_reactive_bootstraps`) plus the v2 per-node fields (`args`, `tag`, `shadow`) it ignores.
//! v1 graphs (every shipped graph today) carry none of them: the compiler then reads the literals of `inputs`
//! (`rot(..)`, `const(..)`, `hint_lev(..)`, `lvl_cap(..)`) and the captured output level/degree.
use crate::error::{json, Error, Result};
use pc_shadow::graph::{Graph, LoadOpts, RawNode};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

/// The v2 `shadow` record of a node (what the fakecap Shadow computed at capture).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ShadowRec {
    #[serde(default)]
    pub level: Option<u32>,
    #[serde(default)]
    pub deg: Option<u8>,
    #[serde(default)]
    pub nf: Option<f64>,
}

/// One graph.json node: the v1 fields the loader reads plus the v2 additions.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GNode {
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
    /// v2: composite-op arguments (rotate_sum steps, pt descriptor, exact scalar, hint, bootstrap route, external
    /// finish descriptor).
    #[serde(default)]
    pub args: Option<Value>,
    /// v2: the full pack tag.
    #[serde(default)]
    pub tag: Option<Value>,
    /// v2: (level, deg, nf) the capture Shadow computed.
    #[serde(default)]
    pub shadow: Option<ShadowRec>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GraphFile {
    #[serde(default)]
    pub version: Value,
    #[serde(default)]
    pub nodes: Vec<GNode>,
}

/// The v2 extras of one node, keyed by the node's output var (outputs are unique: the wrapper's var counter).
#[derive(Clone, Debug, Default)]
pub struct Extra {
    pub args: Option<Value>,
    pub tag: Option<Value>,
    pub shadow: Option<ShadowRec>,
}

/// A loaded block graph: the planner's node order with the captured auto_bootstraps erased.
#[derive(Clone, Debug)]
pub struct BlockGraph {
    pub name: String,
    pub version: u32,
    /// All nodes as captured (planner order), auto_bootstraps included.
    pub n_nodes: usize,
    pub n_auto: usize,
    /// After `erase_reactive_bootstraps`.
    pub graph: Graph,
    pub extra: HashMap<String, Extra>,
    /// The deepest captured refresh landing (level, deg) over every bootstrap node, autos
    /// included (`planner.py::fed_back_entry`: where a token fed back by generation enters the block).
    pub fed_back_entry: Option<(u32, u8)>,
}

fn s_of(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

impl BlockGraph {
    pub fn load(path: impl AsRef<Path>, name: &str, opts: &LoadOpts) -> Result<BlockGraph> {
        let gf: GraphFile = json(path.as_ref())?;
        BlockGraph::from_file(gf, name, opts)
    }

    pub fn from_file(gf: GraphFile, name: &str, opts: &LoadOpts) -> Result<BlockGraph> {
        let version = gf.version.as_u64().unwrap_or(1) as u32;
        if version > 2 {
            return Err(Error::Graph { block: name.into(), msg: format!("graph.json version {version} (known: 1, 2)") });
        }
        let mut extra = HashMap::new();
        let mut fed_back_entry: Option<(f64, u8)> = None;
        for n in &gf.nodes {
            if n.op_type.ends_with("bootstrap") {
                if let Some(l) = n.output_level.as_f64() {
                    if fed_back_entry.map_or(true, |(b, _)| l > b) {
                        fed_back_entry = Some((l, n.output_noise_level.as_u64().unwrap_or(1) as u8));
                    }
                }
            }
        }
        let fed_back_entry = fed_back_entry.map(|(l, d)| (l as u32, d));
        let raw: Vec<RawNode> = gf
            .nodes
            .into_iter()
            .map(|n| {
                if n.args.is_some() || n.tag.is_some() || n.shadow.is_some() {
                    extra.insert(s_of(&n.output), Extra { args: n.args, tag: n.tag, shadow: n.shadow });
                }
                RawNode {
                    op_type: n.op_type,
                    inputs: n.inputs,
                    input_levels: n.input_levels,
                    output: n.output,
                    output_level: n.output_level,
                    output_noise_level: n.output_noise_level,
                    step: n.step,
                }
            })
            .collect();
        let n_nodes = raw.len();
        let g0 = Graph::from_raw(&raw, opts)?;
        let (graph, n_auto) = g0.erase_reactive_bootstraps(opts)?;
        Ok(BlockGraph { name: name.to_string(), version, n_nodes, n_auto, graph, extra, fed_back_entry })
    }

    pub fn args(&self, output: &str) -> Option<&Value> {
        self.extra.get(output).and_then(|e| e.args.as_ref())
    }
}

/// `capture_env.json` of a graph dir: `{env: {NAME: value|null}, session: {...}}`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CaptureEnv {
    #[serde(default)]
    pub env: std::collections::BTreeMap<String, Value>,
    #[serde(default)]
    pub session: Value,
}
