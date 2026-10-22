//! v2 graphs (fakecap: `args.pt.kind = slot` plaintexts registered at one level) compile against the shipped decode
//! plan with 0 check errors: slot plaintexts are encoded at the consumer's level, and `bootstrap_real`
//! (`args.route.real`) routes the KV push refresh real. Skips when the oracle's v2 capture is absent.
use pc_program::compile::{compile_dir, CompileOpts};
use pc_program::program::SiteKind;
use std::path::Path;

const V2: &str = "/data02/users/mithrillo/projects/perseus-oracle/fx/dumps/m7b/gpt2_decode_v2";

#[test]
fn v2_decode_graph_compiles() {
    let g = Path::new(V2);
    if !g.is_dir() {
        eprintln!("skip: {V2} absent");
        return;
    }
    let plan = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bootstrap_placements/gpt2_decode");
    let dc = compile_dir(&plan, g, &CompileOpts { keep_going: true, ..CompileOpts::default() }).unwrap();
    let mut n = 0;
    for (label, r) in dc.programs {
        let p = r.unwrap_or_else(|e| panic!("{label}: {e}"));
        assert!(p.check.errors.is_empty(), "{label}: {} errors, first {:?}", p.check.errors.len(), p.check.errors.first());
        n += 1;
        if label == "block_0" {
            // this capture has no args.route.real record: the KV push stays on the complex route there
            assert!(p.bts_sites.iter().any(|s| s.kind == SiteKind::Deliberate));
        }
    }
    assert_eq!(n, 15);
}
