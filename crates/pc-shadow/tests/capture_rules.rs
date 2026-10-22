//! The C++ rules (no quirks) against the captured graphs: every node whose cipher inputs were produced inside the
//! block (so their captured level and degree are known) is replayed through the matching `Shadow` rule and must
//! land on the captured (output_level, output_noise_level). Plaintext operands take their captured input level at
//! degree 1. Refreshes, hints and level moves carry run-time decisions and are skipped.
use pc_shadow::graph::{is_add_family, is_literal_input, is_mult_family, is_plaintext_name, RawGraph};
use pc_shadow::meta::ChainShape;
use pc_shadow::route::BtsConfig;
use pc_shadow::rules::{PtMeta, Shadow, ShadowConfig};
use pc_shadow::{CtMeta, Quirks};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};

const GRAPH_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../graphs");
const GRAPHS: [&str; 3] =
    ["gpt2_decode", "gpt2_prefill128", "llama31_8b_prefill128"];

fn s(v: &Value) -> String {
    v.as_str().map(str::to_string).unwrap_or_default()
}

#[test]
fn exact_rules_reproduce_captures() {
    if let Some((n, bad)) = replay(Quirks::none(), true) {
        println!("[capture] {n} nodes replayed, {bad} differ from the capture");
        assert!(n > 0);
        assert_eq!(bad, 0);
    }
}

/// The quirk sets against the same captures (informational: how far each mirror is from the C++ capture).
#[test]
fn quirk_sets_vs_captures() {
    for (name, q) in [("planner", Quirks::planner()), ("fake", Quirks::fake()), ("fakecap", Quirks::fakecap())] {
        if let Some((n, bad)) = replay(q, false) {
            println!("[capture] quirks {name}: {n} nodes, {bad} differ");
        }
    }
}

fn replay(quirks: Quirks, verbose: bool) -> Option<(usize, usize)> {
    let shadow = Shadow::new(ShadowConfig { chain: ChainShape::N32, bts: BtsConfig::n32_shipped(), quirks }, None);
    let mut total: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut shown = 0;
    for gname in GRAPHS {
        let dir = std::path::Path::new(GRAPH_ROOT).join(gname);
        if !dir.is_dir() {
            // the Llama capture ships as a release artifact, not in the repository
            println!("SKIP: {} not readable", dir.display());
            continue;
        }
        let mut blocks: Vec<_> = std::fs::read_dir(&dir).unwrap().flatten().map(|e| e.path().join("graph.json")).filter(|p| p.is_file()).collect();
        blocks.sort();
        for gf in blocks {
            let raw: RawGraph = serde_json::from_slice(&std::fs::read(&gf).unwrap()).unwrap();
            let mut state: HashMap<String, CtMeta> = HashMap::new();
            for n in &raw.nodes {
                let out = s(&n.output);
                let (Some(ol), Some(od)) = (n.output_level.as_u64(), n.output_noise_level.as_u64()) else { continue };
                let captured = CtMeta::new(ol as u32, od as u8);
                let inputs: Vec<String> = n.inputs.iter().map(s).collect();
                let lv: Vec<Option<u64>> = (0..inputs.len()).map(|i| n.input_levels.get(i).and_then(Value::as_u64)).collect();
                let cins: Vec<&String> = inputs.iter().filter(|x| !is_literal_input(x) && !is_plaintext_name(x)).collect();
                let ins: Option<Vec<CtMeta>> = cins.iter().map(|v| state.get(*v).copied()).collect();
                let pt = inputs.iter().zip(&lv).find(|(x, _)| is_plaintext_name(x)).and_then(|(_, l)| *l);
                let op = n.op_type.as_str();
                let fam = if is_mult_family(op) {
                    "mult"
                } else if is_add_family(op) {
                    "add"
                } else {
                    op
                };
                let got = match (ins, fam) {
                    (Some(ins), "mult") if ins.len() == 2 => Some(shadow.mult(&ins[0], &ins[1]).map(|r| r.out)),
                    (Some(ins), "mult") if op.starts_with("square") => Some(shadow.square(&ins[0]).map(|r| r.out)),
                    (Some(ins), "mult") if ins.len() == 1 => Some(match pt {
                        Some(pl) => shadow.mult_pt(&ins[0], &PtMeta { level: pl as u32, deg: 1, nf: 0.0, tag: Default::default() }).map(|r| r.out),
                        None => shadow.mult_scalar(&ins[0], 1.0).map(|r| r.out),
                    }),
                    (Some(ins), "add") if ins.len() == 2 => Some(if op.starts_with("sub") { shadow.sub(&ins[0], &ins[1]) } else { shadow.add(&ins[0], &ins[1]) }.map(|r| r.out)),
                    (Some(ins), "add") if ins.len() == 1 => Some(match pt {
                        Some(pl) => shadow.add_pt(&ins[0], &PtMeta { level: pl as u32, deg: 1, nf: 0.0, tag: Default::default() }).map(|r| r.out),
                        None => Ok(ins[0]),
                    }),
                    (Some(ins), "negate" | "negate_inplace") => Some(shadow.negate(&ins[0]).map(|r| r.out)),
                    (Some(ins), "rotate" | "rotate_sum" | "conjugate" | "mult_i" | "clone" | "level_hint") if !ins.is_empty() => Some(Ok(ins[0])),
                    _ => None,
                };
                if let Some(got) = got {
                    let e = total.entry(format!("{fam}{}", if pt.is_some() { "+pt" } else { "" })).or_default();
                    e.0 += 1;
                    let ok = matches!(&got, Ok(m) if (m.level, m.deg) == (captured.level, captured.deg));
                    if !ok {
                        e.1 += 1;
                        if verbose && shown < 12 {
                            shown += 1;
                            let insd: Vec<_> = cins.iter().map(|v| state.get(*v).map(|m| (m.level, m.deg))).collect();
                            println!("  {gname} {op} {inputs:?} ins {insd:?} pt {pt:?}: captured ({ol}, {od}) rule {:?}", got.map(|m| (m.level, m.deg)));
                        }
                    }
                }
                if !out.is_empty() {
                    state.insert(out, captured);
                }
            }
        }
    }
    let (mut n, mut bad) = (0, 0);
    for (k, (a, b)) in &total {
        if verbose {
            println!("[capture] {k:12} {a:6} nodes, {b} differ");
        }
        n += a;
        bad += b;
    }
    Some((n, bad))
}
