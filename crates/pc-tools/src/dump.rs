//! Loader for the reference dumps (tools/dump_finish.py): JSON manifests + raw little-endian u32 arrays.
use serde_json::Value;
use std::path::{Path, PathBuf};

pub struct Dump {
    pub dir: PathBuf,
}

impl Dump {
    pub fn open(dir: impl AsRef<Path>) -> Self {
        Dump { dir: dir.as_ref().to_path_buf() }
    }
    pub fn json(&self, name: &str) -> Value {
        let p = self.dir.join(name);
        let s = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        serde_json::from_str(&s).unwrap()
    }
    pub fn u32s(&self, name: &str) -> Vec<u32> {
        read_u32(&self.dir.join(name))
    }
    pub fn exists(&self, name: &str) -> bool {
        self.dir.join(name).exists()
    }
}

pub fn read_u32(p: &Path) -> Vec<u32> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    assert!(b.len() % 4 == 0);
    let mut v = vec![0u32; b.len() / 4];
    for (i, c) in b.chunks_exact(4).enumerate() {
        v[i] = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
    }
    v
}

pub fn write_u32(p: &Path, v: &[u32]) {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    std::fs::write(p, b).unwrap();
}

pub fn as_u64v(v: &Value) -> Vec<u64> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect()
}
pub fn as_i64v(v: &Value) -> Vec<i64> {
    v.as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect()
}
pub fn as_f64v(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

/// A ciphertext from the dump: eval-form rows (Q ids, optional special ids), metadata.
pub struct HostCt {
    pub level: usize,
    pub noise_level: i64,
    pub noise_factor: f64,
    pub slots: i64,
    pub ids: Vec<usize>,
    pub pids: Vec<usize>,
    pub c: [Vec<u32>; 2],
    pub p: [Vec<u32>; 2],
    pub coef: Option<[Vec<u32>; 2]>,
}

impl Dump {
    pub fn ct(&self, name: &str) -> HostCt {
        let m = self.json(&format!("{name}.json"));
        let ids: Vec<usize> = m["c0"]["ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
        let pids: Vec<usize> = m["c0"]
            .get("special_ids")
            .map(|v| v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect())
            .unwrap_or_default();
        let ld = |s: &str| if self.exists(s) { self.u32s(s) } else { Vec::new() };
        let coef = if self.exists(&format!("{name}.c0coef.bin")) {
            Some([self.u32s(&format!("{name}.c0coef.bin")), self.u32s(&format!("{name}.c1coef.bin"))])
        } else {
            None
        };
        HostCt {
            level: m["level"].as_u64().unwrap() as usize,
            noise_level: m["noise_level"].as_i64().unwrap(),
            noise_factor: m["noise_factor"].as_f64().unwrap(),
            slots: m["slots"].as_i64().unwrap(),
            ids,
            pids,
            c: [self.u32s(&format!("{name}.c0q.bin")), self.u32s(&format!("{name}.c1q.bin"))],
            p: [ld(&format!("{name}.c0p.bin")), ld(&format!("{name}.c1p.bin"))],
            coef,
        }
    }
}
