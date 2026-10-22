//! Loader for the g0 oracle dump (GPT-2 decode token 0, block 0; old-stack `examples/gpt2_from_primitives/fx_dump.py`,
//! layout in the dump's MANIFEST.md). g0 ciphertexts use the d3 per-ciphertext format (`dump.rs`), so `G0::cts()` /
//! `G0::bts()` hand out a `Dump` rooted at the subdirectory.
//!
//! `validate` checks every file against the parameters (row counts from the JSON metadata, N from params.json, npy
//! headers) without reading the payloads.
use crate::dump::Dump;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub enum G0Error {
    Io(PathBuf, std::io::Error),
    Json(PathBuf, serde_json::Error),
    Field(PathBuf, &'static str),
}

impl fmt::Display for G0Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            G0Error::Io(p, e) => write!(f, "{}: {e}", p.display()),
            G0Error::Json(p, e) => write!(f, "{}: bad JSON: {e}", p.display()),
            G0Error::Field(p, k) => write!(f, "{}: missing or mistyped field `{k}`", p.display()),
        }
    }
}

impl std::error::Error for G0Error {}

pub struct G0 {
    pub dir: PathBuf,
    pub params: Value,
    pub n: usize,
}

/// What `validate` found: per category (file count, bytes), and every invalid file with the reason.
#[derive(Default)]
pub struct Inventory {
    pub files: usize,
    pub cats: BTreeMap<&'static str, (usize, u64)>,
    pub counts: BTreeMap<&'static str, usize>,
    pub errors: Vec<String>,
}

impl Inventory {
    fn file(&mut self, cat: &'static str, bytes: u64) {
        self.files += 1;
        let e = self.cats.entry(cat).or_default();
        e.0 += 1;
        e.1 += bytes;
    }
    pub fn report(&self) -> String {
        let mut s = String::new();
        for (k, v) in &self.counts {
            s += &format!("[g0] {k}: {v}\n");
        }
        let mut tot = 0u64;
        for (k, (n, b)) in &self.cats {
            s += &format!("[g0] files {k:>8}: {n:6} files {:9.3} GB\n", *b as f64 / 1e9);
            tot += b;
        }
        s += &format!("[g0] total: {} files {:.3} GB\n", self.files, tot as f64 / 1e9);
        s
    }
}

fn read_json(p: &Path) -> Result<Value, G0Error> {
    let s = std::fs::read_to_string(p).map_err(|e| G0Error::Io(p.to_path_buf(), e))?;
    serde_json::from_str(&s).map_err(|e| G0Error::Json(p.to_path_buf(), e))
}

/// (dtype descr, shape, header length) of a .npy file.
pub fn npy_header(p: &Path) -> Result<(String, Vec<usize>, usize), String> {
    use std::io::Read;
    let mut f = std::fs::File::open(p).map_err(|e| format!("{}: {e}", p.display()))?;
    let mut pre = [0u8; 10];
    f.read_exact(&mut pre).map_err(|e| format!("{}: {e}", p.display()))?;
    if &pre[..6] != b"\x93NUMPY" {
        return Err(format!("{}: not an npy file", p.display()));
    }
    let (hl, off) = if pre[6] == 1 {
        (u16::from_le_bytes([pre[8], pre[9]]) as usize, 10)
    } else {
        let mut x = [0u8; 2];
        f.read_exact(&mut x).map_err(|e| format!("{}: {e}", p.display()))?;
        (u32::from_le_bytes([pre[8], pre[9], x[0], x[1]]) as usize, 12)
    };
    let mut h = vec![0u8; hl];
    f.read_exact(&mut h).map_err(|e| format!("{}: {e}", p.display()))?;
    let h = String::from_utf8_lossy(&h).to_string();
    let descr = h.split("'descr':").nth(1).and_then(|r| r.split('\'').nth(1)).unwrap_or("").to_string();
    let shape_s = h.split("'shape':").nth(1).and_then(|r| r.split('(').nth(1)).and_then(|r| r.split(')').next()).unwrap_or("");
    let shape = shape_s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
    Ok((descr, shape, off + hl))
}

fn dtype_bytes(d: &str) -> Option<u64> {
    Some(match d.trim_start_matches(['<', '|', '=']) {
        "i1" | "u1" => 1,
        "u4" | "i4" | "f4" => 4,
        "f8" | "u8" | "i8" => 8,
        "c16" => 16,
        _ => return None,
    })
}

fn ids_len(v: &Value) -> Option<usize> {
    v.as_array().map(|a| a.len())
}

impl G0 {
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, G0Error> {
        let dir = dir.as_ref().to_path_buf();
        let pp = dir.join("params.json");
        let params = read_json(&pp)?;
        let n = params["N"].as_u64().ok_or(G0Error::Field(pp.clone(), "N"))? as usize;
        Ok(G0 { dir, params, n })
    }
    /// The block-0 vars, as a d3-style dump (`.ct(var)`).
    pub fn cts(&self) -> Dump {
        Dump::open(self.dir.join("cts"))
    }
    /// The bootstraps (`.ct("<k>.in")`, `.ct("<k>.out")`, `.ct("<k>.<stage>")`).
    pub fn bts(&self) -> Dump {
        Dump::open(self.dir.join("bts"))
    }
    pub fn ops(&self) -> Result<Vec<Value>, G0Error> {
        let p = self.dir.join("ops.jsonl");
        let s = std::fs::read_to_string(&p).map_err(|e| G0Error::Io(p.clone(), e))?;
        s.lines().map(|l| serde_json::from_str(l).map_err(|e| G0Error::Json(p.clone(), e))).collect()
    }

    fn check_size(&self, inv: &mut Inventory, cat: &'static str, p: &Path, want: u64) {
        match std::fs::metadata(p) {
            Ok(m) if m.len() == want => inv.file(cat, m.len()),
            Ok(m) => {
                inv.file(cat, m.len());
                inv.errors.push(format!("{}: {} bytes, expected {want}", p.display(), m.len()));
            }
            Err(e) => inv.errors.push(format!("{}: {e}", p.display())),
        }
    }

    fn check_npy(&self, inv: &mut Inventory, cat: &'static str, p: &Path, want_len: Option<usize>) {
        match npy_header(p) {
            Ok((d, shape, off)) => {
                let el: usize = shape.iter().product();
                let Some(b) = dtype_bytes(&d) else {
                    inv.errors.push(format!("{}: unexpected dtype {d}", p.display()));
                    return;
                };
                if let Some(w) = want_len {
                    if el != w {
                        inv.errors.push(format!("{}: {el} elements, expected {w}", p.display()));
                    }
                }
                self.check_size(inv, cat, p, off as u64 + el as u64 * b);
            }
            Err(e) => inv.errors.push(e),
        }
    }

    /// A ciphertext `<base>.json` + its row files: Q rows (ids) x N per part, special rows when listed, the
    /// coefficient form when recorded, the decrypted slots when present.
    fn check_ct(&self, inv: &mut Inventory, cat: &'static str, base: &Path) {
        // names carry dots (`bts/3.in`, `pts/pt_7@30`): append, never `with_extension`
        let jp = PathBuf::from(format!("{}.json", base.display()));
        let m = match read_json(&jp) {
            Ok(m) => m,
            Err(e) => {
                inv.errors.push(e.to_string());
                return;
            }
        };
        inv.file(cat, std::fs::metadata(&jp).map(|m| m.len()).unwrap_or(0));
        let n = self.n as u64;
        let b = base.to_string_lossy().to_string();
        for part in ["c0", "c1"] {
            let Some(nq) = ids_len(&m[part]["ids"]) else {
                inv.errors.push(format!("{}: no {part}.ids", jp.display()));
                return;
            };
            self.check_size(inv, cat, Path::new(&format!("{b}.{part}q.bin")), nq as u64 * n * 4);
            if let Some(np) = ids_len(&m[part]["special_ids"]) {
                self.check_size(inv, cat, Path::new(&format!("{b}.{part}p.bin")), np as u64 * n * 4);
            }
            if let Some(cm) = ids_len(&m["coef_moduli"]) {
                self.check_size(inv, cat, Path::new(&format!("{b}.{part}coef.bin")), cm as u64 * n * 4);
            }
        }
        let sp = PathBuf::from(format!("{b}.slots.npy"));
        if sp.exists() {
            let slots = m["slots"].as_u64().map(|x| x as usize);
            self.check_npy(inv, cat, &sp, slots);
        }
    }

    /// Plaintext rows: `ids` limbs x the stored width (compact plaintexts store fewer than N values per limb).
    fn check_pt_rows(&self, inv: &mut Inventory, cat: &'static str, c0: &Value, q: &Path, p: &Path) {
        let w = |k: &str| c0[k].as_array().map(|a| a.iter().filter_map(|x| x.as_u64()).max().unwrap_or(0).min(self.n as u64));
        match (ids_len(&c0["ids"]), w("q_sizes")) {
            (Some(nq), Some(wq)) => self.check_size(inv, cat, q, nq as u64 * wq * 4),
            _ => inv.errors.push(format!("{}: no c0.ids / q_sizes", q.display())),
        }
        if let (Some(np), Some(wp)) = (ids_len(&c0["special_ids"]), w("p_sizes")) {
            self.check_size(inv, cat, p, np as u64 * wp * 4);
        }
    }

    pub fn validate(&self) -> Inventory {
        let mut inv = Inventory::default();
        let pj = &self.params;
        let n = self.n as u64;
        inv.counts.insert("N", self.n);
        inv.counts.insert("L (top Q limb index)", pj["L"].as_u64().unwrap_or(0) as usize);
        inv.counts.insert("K (special primes)", pj["K"].as_u64().unwrap_or(0) as usize);
        inv.counts.insert("dnum", pj["dnum"].as_u64().unwrap_or(0) as usize);
        inv.file("params", std::fs::metadata(self.dir.join("params.json")).map(|m| m.len()).unwrap_or(0));
        let man = self.dir.join("MANIFEST.md");
        if man.exists() {
            inv.file("params", std::fs::metadata(&man).map(|m| m.len()).unwrap_or(0));
        } else {
            inv.errors.push(format!("{}: missing", man.display()));
        }
        self.check_npy(&mut inv, "params", &self.dir.join("sk.npy"), Some(self.n));

        // keys: OpenFHE CPU-map keys (b; dense a when has_a), device encapsulation keys
        if let Some(keys) = pj["keys"].as_object() {
            inv.counts.insert("keys (CPU map, incl. relin)", keys.len());
            let mut seeded = 0;
            for (name, k) in keys {
                let file = if name == "mult" { "mult".to_string() } else { format!("auto{name}") };
                let shape: Vec<u64> = k["shape"].as_array().map(|a| a.iter().filter_map(|x| x.as_u64()).collect()).unwrap_or_default();
                if shape.len() != 3 || shape[2] != n {
                    inv.errors.push(format!("key {name}: shape {shape:?} (expected [dnum, moduli, N])"));
                    continue;
                }
                let bytes = shape.iter().product::<u64>() * 4;
                self.check_size(&mut inv, "keys", &self.dir.join(format!("keys/{file}.b.bin")), bytes);
                if k["has_a"].as_bool() == Some(true) {
                    self.check_size(&mut inv, "keys", &self.dir.join(format!("keys/{file}.a.bin")), bytes);
                }
                match k["a_seed"].as_array().map(|a| a.len()) {
                    Some(8) => seeded += 1,
                    Some(0) if k["has_a"].as_bool() == Some(true) => {}
                    _ => inv.errors.push(format!("key {name}: no 8-word KSKA seed and no dense `a`")),
                }
                if let Some(mods) = k["b_moduli"].as_array() {
                    if mods.len() as u64 != shape[0] || mods.iter().any(|d| d.as_array().map(|x| x.len() as u64) != Some(shape[1])) {
                        inv.errors.push(format!("key {name}: b_moduli does not match shape {shape:?}"));
                    }
                }
            }
            inv.counts.insert("keys with KSKA seed", seeded);
        } else {
            inv.errors.push("params.json: no `keys`".into());
        }
        if let Some(dev) = pj["encaps_device"].as_object() {
            let mut nk = 0;
            for (fname, k) in dev.iter().filter(|(k, _)| k.starts_with("dev_")) {
                nk += 1;
                for part in ["a", "b"] {
                    if let Some(shape) = k[part]["shape"].as_array() {
                        let bytes = shape.iter().filter_map(|x| x.as_u64()).product::<u64>() * 4;
                        self.check_size(&mut inv, "keys", &self.dir.join(format!("keys/{fname}.{part}.bin")), bytes);
                    } else if let Some(same) = k[part]["same_as"].as_str() {
                        // another route's device copy of the same key: stored once
                        if !self.dir.join(format!("keys/{same}.bin")).exists() {
                            inv.errors.push(format!("{fname}.{part}: same_as {same}, which is missing"));
                        }
                    } else if part == "a" && k["a"]["released"].as_bool() == Some(true) && k["a"]["seed"].as_array().map(|s| s.len()) != Some(8) {
                        inv.errors.push(format!("{fname}: released `a` without a seed"));
                    }
                }
            }
            inv.counts.insert("device encapsulation keys", nk);
        }

        // block-0 vars
        let cts = self.dir.join("cts");
        let mut ncts = 0;
        if let Ok(rd) = std::fs::read_dir(&cts) {
            let mut names: Vec<String> = rd.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().to_string()).filter(|f| f.ends_with(".json")).collect();
            names.sort();
            for f in names {
                ncts += 1;
                self.check_ct(&mut inv, "cts", &cts.join(&f[..f.len() - 5]));
            }
        }
        inv.counts.insert("block-0 ciphertexts", ncts);

        // plaintexts
        let pts = self.dir.join("pts");
        let mut npts = 0;
        if let Ok(rd) = std::fs::read_dir(&pts) {
            for f in rd.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().to_string()).filter(|f| f.ends_with(".json")) {
                npts += 1;
                let base = pts.join(&f[..f.len() - 5]);
                let jp = pts.join(&f);
                match read_json(&jp) {
                    Ok(m) => {
                        inv.file("pts", std::fs::metadata(&jp).map(|m| m.len()).unwrap_or(0));
                        let b = base.to_string_lossy().to_string();
                        if m["device"].as_bool() == Some(true) {
                            self.check_pt_rows(&mut inv, "pts", &m["c0"], Path::new(&format!("{b}.q.bin")), Path::new(&format!("{b}.p.bin")));
                        }
                        self.check_npy(&mut inv, "pts", Path::new(&format!("{b}.values.npy")), None);
                    }
                    Err(e) => inv.errors.push(e.to_string()),
                }
            }
        }
        inv.counts.insert("plaintexts", npts);

        // bootstraps
        let bd = self.dir.join("bts");
        let mut nb = 0;
        let mut nstage = 0;
        if let Ok(rd) = std::fs::read_dir(&bd) {
            let mut files: Vec<String> = rd.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().to_string()).collect();
            files.sort();
            for f in files.iter().filter(|f| f.ends_with(".json") && !f[..f.len() - 5].contains('.')) {
                nb += 1;
                let k = &f[..f.len() - 5];
                let jp = bd.join(f);
                let m = match read_json(&jp) {
                    Ok(m) => m,
                    Err(e) => {
                        inv.errors.push(e.to_string());
                        continue;
                    }
                };
                inv.file("bts", std::fs::metadata(&jp).map(|m| m.len()).unwrap_or(0));
                for io in ["in", "out"] {
                    self.check_ct(&mut inv, "bts", &bd.join(format!("{k}.{io}")));
                }
                for st in m["stages"].as_array().into_iter().flatten().filter_map(|x| x.as_str()) {
                    nstage += 1;
                    self.check_ct(&mut inv, "bts", &bd.join(format!("{k}.{st}")));
                }
            }
        }
        inv.counts.insert("bootstraps (dumped)", nb);
        inv.counts.insert("bootstrap stage outputs", nstage);

        // per-route CtS plaintexts
        if let Some(routes) = pj["boot_routes"].as_object() {
            let mut np = 0;
            for (sl, pre) in routes {
                for (k, st) in pre["CtS"].as_array().into_iter().flatten().enumerate() {
                    for (j, pt) in st["pts"].as_array().into_iter().flatten().enumerate() {
                        np += 1;
                        let d = self.dir.join(format!("routes/{sl}"));
                        self.check_pt_rows(&mut inv, "routes", &pt["c0"], &d.join(format!("cts{k}.pt{j}.q.bin")), &d.join(format!("cts{k}.pt{j}.p.bin")));
                    }
                }
            }
            inv.counts.insert("bootstrap routes", routes.len());
            inv.counts.insert("route CtS plaintexts", np);
        }

        match self.ops() {
            Ok(ops) => {
                inv.file("params", std::fs::metadata(self.dir.join("ops.jsonl")).map(|m| m.len()).unwrap_or(0));
                inv.counts.insert("ops recorded (whole run)", ops.len());
                let blk0 = ops.iter().filter(|o| o["step"].as_str().is_some_and(|s| s.starts_with("blk0"))).count();
                inv.counts.insert("ops recorded (block 0)", blk0);
                let dumped: Vec<&Value> = ops.iter().filter(|o| o["dumped"].as_bool() == Some(true)).collect();
                inv.counts.insert("ops with dumped output", dumped.len());
                for o in dumped {
                    let v = o["out"].as_str().unwrap_or("");
                    if !cts.join(format!("{v}.json")).exists() {
                        inv.errors.push(format!("ops.jsonl: {v} marked dumped, cts/{v}.json missing"));
                    }
                }
            }
            Err(e) => inv.errors.push(e.to_string()),
        }
        inv
    }
}
