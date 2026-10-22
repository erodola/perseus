//! Field kinds and their legacy-env spelling. The bridge must round-trip every value a shipped plan stamps
//! (`capture_env`, `runtime=`) byte for byte, so the parsers are strict: a flag is "0" or "1", an int is what
//! `i64::to_string` prints back, a float any finite spelling (compared by value), and anything else is refused
//! with the knob named.
use crate::error::{Error, Result};

pub type Flag = bool;
pub type Int = i64;
pub type Float = f64;
pub type Str = String;
/// A comma-separated list (`SPARSE_BTS_SLOTS=512,1`); the empty string is the empty list.
pub type List = Vec<String>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Kind {
    Flag,
    Int,
    Float,
    Str,
    List,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Flag => "flag",
            Kind::Int => "int",
            Kind::Float => "float",
            Kind::Str => "str",
            Kind::List => "list",
        }
    }
}

pub trait Value: Sized {
    fn to_env(&self) -> String;
    fn from_env(knob: &'static str, s: &str) -> Result<Self>;
}

fn bad(knob: &'static str, s: &str, want: &str) -> Error {
    Error::Value { knob, value: s.to_string(), want: want.to_string() }
}

impl Value for bool {
    fn to_env(&self) -> String {
        if *self { "1" } else { "0" }.to_string()
    }
    fn from_env(knob: &'static str, s: &str) -> Result<bool> {
        match s {
            "1" => Ok(true),
            "0" => Ok(false),
            _ => Err(bad(knob, s, "a flag: \"0\" or \"1\"")),
        }
    }
}

impl Value for i64 {
    fn to_env(&self) -> String {
        self.to_string()
    }
    fn from_env(knob: &'static str, s: &str) -> Result<i64> {
        match s.parse::<i64>() {
            Ok(v) if v.to_string() == s => Ok(v),
            _ => Err(bad(knob, s, "an integer in canonical form")),
        }
    }
}

impl Value for f64 {
    fn to_env(&self) -> String {
        // integral values print without a fraction ("32", the way the env dicts spell them); others shortest-roundtrip
        if self.fract() == 0.0 && self.abs() < 1e15 {
            format!("{}", *self as i64)
        } else {
            format!("{self}")
        }
    }
    fn from_env(knob: &'static str, s: &str) -> Result<f64> {
        // any finite spelling ("2e-5", "2e-05", "0.00002" are one value): floats compare by value, the env side
        // prints the canonical form (`to_env`), so a round trip is equal by value, not byte for byte
        match s.trim().parse::<f64>() {
            Ok(v) if v.is_finite() && s == s.trim() => Ok(v),
            _ => Err(bad(knob, s, "a finite number (\"32\", \"0.5\", \"2e-5\")")),
        }
    }
}

impl Value for String {
    fn to_env(&self) -> String {
        self.clone()
    }
    fn from_env(_knob: &'static str, s: &str) -> Result<String> {
        Ok(s.to_string())
    }
}

impl Value for Vec<String> {
    fn to_env(&self) -> String {
        self.join(",")
    }
    fn from_env(knob: &'static str, s: &str) -> Result<Vec<String>> {
        if s.is_empty() {
            return Ok(vec![]);
        }
        let v: Vec<String> = s.split(',').map(str::to_string).collect();
        if v.iter().any(|x| x.is_empty() || x.trim() != x) {
            return Err(bad(knob, s, "a comma-separated list without empty or padded items"));
        }
        Ok(v)
    }
}
