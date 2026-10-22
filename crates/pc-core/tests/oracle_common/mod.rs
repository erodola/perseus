//! Test-only helpers for the d3-dump gates: d3 dump access, a reference NTT (OpenFHE bit-reversed order), exact CRT,
//! and a complex FFT for the canonical embedding. Primes / roots are read from d3/params.json.
#![allow(dead_code)]
use pc_core::bts_setup::C64;
use serde_json::Value;
use std::path::{Path, PathBuf};

pub const D3: &str = "/data02/users/mithrillo/projects/perseus-oracle/fx/dumps/d3";

pub fn d3() -> PathBuf {
    let p = PathBuf::from(D3);
    assert!(p.join("params.json").exists(), "oracle dump missing: {D3}/params.json");
    p
}

pub fn params() -> Value {
    let s = std::fs::read_to_string(d3().join("params.json")).expect("read d3/params.json");
    serde_json::from_str(&s).expect("parse d3/params.json")
}

pub fn read_u32(p: &Path) -> Vec<u32> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    assert!(b.len() % 4 == 0);
    b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// Read `count` u32 words at word offset `off`.
pub fn read_u32_at(p: &Path, off: usize, count: usize) -> Vec<u32> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    f.seek(SeekFrom::Start(off as u64 * 4)).unwrap();
    let mut b = vec![0u8; count * 4];
    f.read_exact(&mut b).unwrap();
    b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

pub fn read_npy_i8(p: &Path) -> Vec<i8> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    assert_eq!(&b[..6], b"\x93NUMPY");
    let hl = u16::from_le_bytes([b[8], b[9]]) as usize;
    let hdr = std::str::from_utf8(&b[10..10 + hl]).unwrap();
    assert!(hdr.contains("'|i1'"), "expected int8 npy: {hdr}");
    b[10 + hl..].iter().map(|&x| x as i8).collect()
}

pub fn u64v(v: &Value) -> Vec<u64> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect()
}
pub fn i64v(v: &Value) -> Vec<i64> {
    v.as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect()
}
pub fn f64v(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

// ---- modular arithmetic -------------------------------------------------------------------------------------------

/// a * b mod p for p < 2^32 (the product fits u64).
#[inline]
pub fn mulmod(a: u64, b: u64, p: u64) -> u64 {
    debug_assert!(p < (1 << 32) && a < p && b < p);
    a * b % p
}
pub fn powmod(mut a: u64, mut e: u64, p: u64) -> u64 {
    let mut r = 1u64 % p;
    a %= p;
    while e > 0 {
        if e & 1 == 1 {
            r = mulmod(r, a, p);
        }
        a = mulmod(a, a, p);
        e >>= 1;
    }
    r
}
pub fn invmod(a: u64, p: u64) -> u64 {
    powmod(a, p - 2, p)
}
pub fn brev(x: usize, bits: u32) -> usize {
    x.reverse_bits() >> (usize::BITS - bits)
}

/// Twiddle tables of one prime: psi^brev(i) and psi^-brev(i).
pub struct NttTables {
    pub p: u64,
    pub n: usize,
    tw: Vec<u64>,
    itw: Vec<u64>,
    ninv: u64,
}

impl NttTables {
    pub fn new(p: u64, psi: u64, n: usize) -> Self {
        assert_eq!(powmod(psi, n as u64, p), p - 1, "psi is not a primitive 2N-th root mod {p}");
        let logn = n.trailing_zeros();
        let ipsi = invmod(psi, p);
        let tw = (0..n).map(|i| powmod(psi, brev(i, logn) as u64, p)).collect();
        let itw = (0..n).map(|i| powmod(ipsi, brev(i, logn) as u64, p)).collect();
        NttTables { p, n, tw, itw, ninv: invmod(n as u64, p) }
    }

    /// OpenFHE ForwardTransformToBitReverse: out[i] = a(psi^(2 brev(i) + 1)).
    pub fn fwd(&self, a: &mut [u32]) {
        let (n, p) = (self.n, self.p);
        let mut t = n;
        let mut m = 1;
        while m < n {
            t /= 2;
            for i in 0..m {
                let s = self.tw[m + i];
                let j1 = 2 * i * t;
                for j in j1..j1 + t {
                    let u = a[j] as u64;
                    let v = mulmod(a[j + t] as u64, s, p);
                    a[j] = ((u + v) % p) as u32;
                    a[j + t] = ((u + p - v) % p) as u32;
                }
            }
            m *= 2;
        }
    }

    /// OpenFHE InverseTransformFromBitReverse.
    pub fn inv(&self, a: &mut [u32]) {
        let (n, p) = (self.n, self.p);
        let mut t = 1;
        let mut m = n / 2;
        while m >= 1 {
            let mut j1 = 0;
            for i in 0..m {
                let s = self.itw[m + i];
                for j in j1..j1 + t {
                    let u = a[j] as u64;
                    let v = a[j + t] as u64;
                    a[j] = ((u + v) % p) as u32;
                    a[j + t] = mulmod((u + p - v) % p, s, p) as u32;
                }
                j1 += 2 * t;
            }
            t *= 2;
            m /= 2;
        }
        for x in a.iter_mut() {
            *x = mulmod(*x as u64, self.ninv, p) as u32;
        }
    }
}

/// Centered value of a residue.
pub fn centered(v: u32, p: u32) -> i64 {
    if v as u64 > p as u64 / 2 {
        v as i64 - p as i64
    } else {
        v as i64
    }
}

/// Exact CRT (Garner) of residues over primes whose product < 2^126, centered, as f64.
pub fn crt_centered(res: &[u32], primes: &[u32]) -> f64 {
    let mut x: u128 = res[0] as u128;
    let mut prod: u128 = primes[0] as u128;
    for k in 1..primes.len() {
        let p = primes[k] as u128;
        let xm = (x % p) as u64;
        let inv = invmod((prod % p) as u64, p as u64) as u128;
        let t = ((res[k] as u128 + p - xm as u128) % p) * inv % p;
        x += t * prod;
        prod *= p;
    }
    if x > prod / 2 {
        -((prod - x) as f64)
    } else {
        x as f64
    }
}

// ---- complex FFT / canonical embedding ----------------------------------------------------------------------------

/// In-place radix-2 FFT, sign -1 (forward: X_k = sum x_j e^(-2 pi i jk/n)) or +1.
pub fn fft(a: &mut [C64], sign: f64) {
    let n = a.len();
    let logn = n.trailing_zeros();
    for i in 0..n {
        let j = brev(i, logn);
        if j > i {
            a.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = sign * 2.0 * std::f64::consts::PI / len as f64;
        let half = len / 2;
        let w: Vec<C64> = (0..half).map(|k| C64::cis(ang * k as f64)).collect();
        for st in (0..n).step_by(len) {
            for k in 0..half {
                let u = a[st + k];
                let v = a[st + k + half] * w[k];
                a[st + k] = u + v;
                a[st + k + half] = u - v;
            }
        }
        len <<= 1;
    }
}

/// Canonical embedding (OpenFHE slot convention): slot j = m(zeta^(5^j)), zeta = e^(i pi / N), j < N/2.
pub fn decode_slots(m: &[f64]) -> Vec<C64> {
    let n = m.len();
    let mut a = vec![C64::ZERO; 2 * n];
    for i in 0..n {
        a[i] = C64::new(m[i], 0.0);
    }
    // value at zeta^t = sum_i m_i e^(i pi i t / N) = inverse-sign DFT of length 2N at index t
    fft(&mut a, 1.0);
    let m2 = 2 * n;
    let mut g = 1usize;
    (0..n / 2)
        .map(|_| {
            let v = a[g];
            g = g * 5 % m2;
            v
        })
        .collect()
}

/// Inverse canonical embedding of a full slot vector z (N/2 complex) to real coefficients (not rounded).
pub fn encode_slots(z: &[C64], n: usize) -> Vec<f64> {
    assert_eq!(z.len(), n / 2);
    let m2 = 2 * n;
    let mut v = vec![C64::ZERO; m2];
    let mut g = 1usize;
    for zj in z {
        v[g] = *zj;
        v[m2 - g] = zj.conj();
        g = g * 5 % m2;
    }
    // m_i = (1/N) sum_{t odd} V_t zeta^(-i t)
    fft(&mut v, -1.0);
    (0..n).map(|i| v[i].re / n as f64).collect()
}

/// Find a prime q = 1 mod 2n below 2^28 and a primitive 2n-th root of unity.
pub fn ntt_prime(n: usize) -> (u64, u64) {
    let m = 2 * n as u64;
    let mut q = ((1u64 << 28) / m) * m + 1;
    loop {
        q -= m;
        if is_prime(q) {
            break;
        }
    }
    for x in 2..q {
        let r = powmod(x, (q - 1) / m, q);
        if powmod(r, n as u64, q) == q - 1 {
            return (q, r);
        }
    }
    unreachable!()
}

fn is_prime(q: u64) -> bool {
    if q < 2 {
        return false;
    }
    let mut d = 2;
    while d * d <= q {
        if q % d == 0 {
            return false;
        }
        d += 1;
    }
    true
}
