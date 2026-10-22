//! Pack tags: what the slots of a ciphertext hold (period, support, field), propagated op by op so a refresh can
//! be routed sparse. A 1:1 port of the tag algebra at the top of `perseus/impl/fakecap.py` (itself the Python twin of
//! `include/packing/pack_tag.h`); the test vectors in `tests/data/tag_vectors.json` are produced by that Python and
//! must match exactly. Python integer semantics are kept where they matter: `%` is floor-mod (`rem_euclid`), `//`
//! on non-negative operands, products in i128.
//!
//! The one non-value field is `conj_of`: the original stores the ciphertext OBJECT a conjugate came from and
//! compares identities (`is`). Here it is an opaque ciphertext id chosen by the caller.
use serde::{Deserialize, Serialize};

pub const EMPTY: i32 = 0;
pub const AP: i32 = 1;
pub const DENSE: i32 = 2;

pub const F_UNKNOWN: i32 = 0;
pub const F_REAL: i32 = 1;
pub const F_IMAG: i32 = 2;
pub const F_COMPLEX: i32 = 3;

/// Which slots are live: nothing, everything, or `count` blocks of `width` slots every `stride` from `offset`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Support {
    pub kind: i32,
    pub offset: i64,
    pub stride: i64,
    pub count: i64,
    pub width: i64,
}

impl Default for Support {
    fn default() -> Self {
        Support::dense()
    }
}

impl Support {
    pub fn new(kind: i32, offset: i64, stride: i64, count: i64, width: i64) -> Support {
        Support { kind, offset, stride, count, width }
    }
    pub fn empty() -> Support {
        Support::new(EMPTY, 0, 1, 0, 1)
    }
    pub fn dense() -> Support {
        Support::new(DENSE, 0, 1, 0, 1)
    }
    pub fn ap(off: i64, st: i64, cnt: i64) -> Support {
        if cnt <= 0 {
            return Support::empty();
        }
        if st <= 1 && off == 0 {
            return Support::dense();
        }
        Support::new(AP, off, st, cnt, 1)
    }
    pub fn block(off: i64, st: i64, w: i64, cnt: i64) -> Support {
        if cnt <= 0 || w <= 0 {
            return Support::empty();
        }
        if w == 1 {
            return Support::ap(off, st, cnt);
        }
        if w >= st {
            return Support::dense();
        }
        Support::new(AP, off, st, cnt, w)
    }
    pub fn is_dense(&self) -> bool {
        self.kind == DENSE
    }
    pub fn is_empty(&self) -> bool {
        self.kind == EMPTY
    }
}

/// Opaque identity of the ciphertext a conjugate was taken from (fakecap's `conj_of` object).
pub type CtId = u64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tag {
    pub slots: i64,
    pub period: i64,
    pub support: Support,
    pub field: i32,
    pub conj_of: Option<CtId>,
}

impl Default for Tag {
    /// The unknown tag (fakecap `Tag()`): no slot count, no period, dense support.
    fn default() -> Self {
        Tag { slots: 0, period: 0, support: Support::dense(), field: F_UNKNOWN, conj_of: None }
    }
}

fn gcd(a: i64, b: i64) -> i64 {
    // math.gcd: non-negative, gcd(0, b) = |b|
    let (mut a, mut b) = (a.unsigned_abs(), b.unsigned_abs());
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a as i64
}

/// Python floor division (operands of either sign).
fn floordiv(a: i64, b: i64) -> i64 {
    a.div_euclid(b) - if b < 0 && a.rem_euclid(b) != 0 { 1 } else { 0 }
}

impl Tag {
    pub fn new(slots: i64, period: i64, support: Support) -> Tag {
        Tag { slots, period, support, field: F_UNKNOWN, conj_of: None }
    }
    pub fn known(&self) -> bool {
        self.slots > 0 && self.period > 0
    }
    pub fn periodic_at(&self, s: i64) -> bool {
        self.period > 0 && s.rem_euclid(self.period) == 0
    }
    /// Whether folding the slots onto `s` (sum of the S/s copies) maps no two live slots onto one.
    pub fn fold_collision_free_at(&self, s: i64) -> bool {
        let sp = &self.support;
        if sp.is_empty() {
            return true;
        }
        if sp.is_dense() {
            return s >= self.slots;
        }
        if sp.width > 1 {
            if sp.stride > 0 && s.rem_euclid(sp.stride) == 0 {
                return sp.count <= floordiv(s, sp.stride) && sp.width <= sp.stride;
            }
            return false;
        }
        sp.count <= floordiv(s, gcd(sp.stride, s))
    }
    /// The smallest power-of-two slot count a refresh of this value can run at.
    pub fn min_routable_s(&self) -> i64 {
        let mut s = 1i64;
        while s < self.slots {
            if self.periodic_at(s) || self.fold_collision_free_at(s) {
                return s;
            }
            s <<= 1;
        }
        self.slots
    }
}

pub fn top(slots: i64) -> Tag {
    Tag::new(slots, slots, Support::dense())
}

pub fn lcm_capped(a: i64, b: i64, cap: i64) -> i64 {
    if a <= 0 || b <= 0 {
        return cap;
    }
    let l = (a as i128) * (b as i128) / gcd(a, b) as i128;
    if l >= cap as i128 {
        cap
    } else {
        l as i64
    }
}

pub fn union(a: Support, b: Support) -> Support {
    if a.is_empty() {
        return b;
    }
    if b.is_empty() {
        return a;
    }
    if a.is_dense() || b.is_dense() {
        return Support::dense();
    }
    if a.stride == b.stride && a.offset == b.offset && a.width == b.width {
        return Support::block(a.offset, a.stride, a.width, a.count.max(b.count));
    }
    if a.stride == b.stride {
        let lo = a.offset.min(b.offset);
        let hi = (a.offset + a.width).max(b.offset + b.width);
        if hi - lo <= a.stride {
            return Support::block(lo, a.stride, hi - lo, a.count.max(b.count));
        }
    }
    Support::dense()
}

pub fn intersect(a: Support, b: Support) -> Support {
    if a.is_empty() || b.is_empty() {
        return Support::empty();
    }
    if a.is_dense() {
        return b;
    }
    if b.is_dense() {
        return a;
    }
    if a.stride == b.stride && a.offset == b.offset {
        return Support::block(a.offset, a.stride, a.width.min(b.width), a.count.min(b.count));
    }
    if (a.count as i128) * (a.width as i128) <= (b.count as i128) * (b.width as i128) {
        a
    } else {
        b
    }
}

pub fn f_add(a: i32, b: i32) -> i32 {
    if a == F_UNKNOWN || b == F_UNKNOWN {
        return F_UNKNOWN;
    }
    if a == b {
        a
    } else {
        F_COMPLEX
    }
}

pub fn f_mult(a: i32, b: i32) -> i32 {
    if a == F_UNKNOWN || b == F_UNKNOWN {
        return F_UNKNOWN;
    }
    if a == F_COMPLEX || b == F_COMPLEX {
        return F_COMPLEX;
    }
    if a == b {
        F_REAL
    } else {
        F_IMAG
    }
}

/// A copy with `field` replaced and the conjugate link dropped.
pub fn with_field(t: &Tag, f: i32) -> Tag {
    Tag { slots: t.slots, period: t.period, support: t.support, field: f, conj_of: None }
}

pub fn t_mult_scalar(x: &Tag, c: f64) -> Tag {
    if !x.known() {
        return with_field(&Tag::default(), x.field);
    }
    if c == 0.0 {
        return with_field(&Tag::new(x.slots, 1, Support::empty()), x.field);
    }
    with_field(x, x.field)
}

pub fn t_add_scalar(x: &Tag, c: f64) -> Tag {
    let f = if c == 0.0 { x.field } else { f_add(x.field, F_REAL) };
    if !x.known() {
        return with_field(&Tag::default(), f);
    }
    if c == 0.0 {
        return with_field(x, f);
    }
    with_field(&Tag::new(x.slots, x.period, Support::dense()), f)
}

pub fn t_square(x: &Tag) -> Tag {
    with_field(x, f_mult(x.field, x.field))
}

pub fn t_add(a: &Tag, b: &Tag) -> Tag {
    let f = f_add(a.field, b.field);
    if !a.known() || !b.known() {
        return with_field(&Tag::default(), f);
    }
    with_field(&Tag::new(a.slots, lcm_capped(a.period, b.period, a.slots), union(a.support, b.support)), f)
}

pub fn t_mult(a: &Tag, b: &Tag) -> Tag {
    let f = f_mult(a.field, b.field);
    if !a.known() && !b.known() {
        return with_field(&Tag::default(), f);
    }
    if !a.known() {
        return with_field(&Tag::new(b.slots, b.slots, b.support), f);
    }
    if !b.known() {
        return with_field(&Tag::new(a.slots, a.slots, a.support), f);
    }
    with_field(&Tag::new(a.slots, lcm_capped(a.period, b.period, a.slots), intersect(a.support, b.support)), f)
}

pub fn t_mult_i(x: &Tag) -> Tag {
    let f = match x.field {
        F_REAL => F_IMAG,
        F_IMAG => F_REAL,
        other => other,
    };
    with_field(x, f)
}

pub fn t_rotate(x: &Tag, k: i64) -> Tag {
    if !x.known() {
        return with_field(&Tag::default(), x.field);
    }
    let mut r = with_field(x, x.field);
    if x.support.kind == AP {
        let off = (x.support.offset - k).rem_euclid(x.slots);
        r.support = Support::block(off, x.support.stride, x.support.width, x.support.count);
    }
    r
}

pub fn t_conjugate(x: &Tag) -> Tag {
    if !x.known() {
        return with_field(&Tag::default(), x.field);
    }
    let mut r = with_field(x, x.field);
    if x.support.kind == AP {
        r.support = Support::dense();
    }
    r
}

/// A refresh at `s` slots (0 = dense) of a ciphertext with `slots` slots: a sparse route emits s-periodic data.
pub fn t_bootstrap(x: &Tag, s: i64, slots: i64) -> Tag {
    if s <= 0 {
        return with_field(x, x.field);
    }
    if slots <= 0 {
        return with_field(&Tag::default(), x.field);
    }
    with_field(&Tag::new(slots, s.min(slots), Support::dense()), x.field)
}

/// x + conj(x) is real, x - conj(x) imaginary: `t` (the sum's tag) gets that field when one operand is the other's
/// conjugate. `a`, `b` are the operands' ids, `ta`, `tb` their tags.
pub fn conj_pair(mut t: Tag, a: CtId, b: CtId, ta: &Tag, tb: &Tag, sub: bool) -> Tag {
    if tb.conj_of == Some(a) || ta.conj_of == Some(b) {
        t.field = if sub { F_IMAG } else { F_REAL };
    }
    t
}

/// tag_plaintext(analyze_packing(values)) from the measured signature of a plaintext.
pub fn from_signature(slots: i64, period: i64, live: i64, stride: i64, window: i64, offset: i64) -> Tag {
    let mut t = Tag::new(slots, if period > 0 { period } else { slots }, Support::dense());
    t.support = if live <= 0 {
        Support::empty()
    } else if live >= slots {
        Support::dense()
    } else if stride > 1 {
        Support::ap(offset, stride, live)
    } else {
        Support::ap(offset, 1, if window > offset { window - offset } else { live })
    };
    t
}

/// The `tag_reduce(stride)` overlay (fakecap CapInf.tag_reduce): the caller asserts the value is `stride`-periodic.
pub fn t_reduce(x: &Tag, slots: i64, stride: i64) -> Tag {
    let p = if stride <= 1 { 1 } else { stride.min(slots) };
    with_field(&Tag::new(slots, p, Support::dense()), x.field)
}
