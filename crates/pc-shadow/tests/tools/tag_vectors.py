"""Pack-tag test vectors: random cases evaluated by the ORIGINAL perseus/impl/fakecap.py tag algebra.

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/data02/users/mithrillo/projects/perseus \
        /data02/users/mithrillo/projects/perseus/.venv/bin/python crates/pc-shadow/tests/tools/tag_vectors.py \
        crates/pc-shadow/tests/data/tag_vectors.json [n=10000] [seed=20261018]

Encoding: Support = [kind, offset, stride, count, width]; Tag = [slots, period, Support, field, conj_of] where
conj_of is a ciphertext id (int) or null. Each case is {"f": name, "args": [...], "out": ...}.
"""
from __future__ import annotations

import json
import random
import sys

from perseus.impl import fakecap as fc


class Ct:
    """A ciphertext identity for conj_pair (`is` comparisons in the original)."""
    __slots__ = ("id",)

    def __init__(self, i):
        self.id = i


CTS = [Ct(i) for i in range(4)]


def enc_s(s): return [s.kind, s.offset, s.stride, s.count, s.width]


def enc_t(t): return [t.slots, t.period, enc_s(t.support), t.field, None if t.conj_of is None else t.conj_of.id]


def rnd_support(r: random.Random, slots: int):
    k = r.random()
    off = r.randrange(0, max(slots, 1) + 3)
    st = r.choice([0, 1, 2, 3, 4, 8, 16, 64, 512, 1024, 4096, r.randrange(1, 9000)])
    cnt = r.choice([0, 1, 2, 3, 7, 8, 64, 100, 512, r.randrange(0, 40000)])
    w = r.choice([0, 1, 1, 2, 3, 4, 8, 16, 64, r.randrange(1, 600)])
    if k < 0.1:
        return fc.Support.empty()
    if k < 0.25:
        return fc.Support.dense()
    if k < 0.45:
        return fc.Support.ap(off, st, cnt)
    if k < 0.7:
        return fc.Support.block(off, st, w, cnt)
    # raw, unnormalized (any field combination the constructors would not produce)
    return fc.Support(r.choice([fc.EMPTY, fc.AP, fc.DENSE, fc.AP]), off, st, cnt, w)


def rnd_tag(r: random.Random, slots: int | None = None):
    if r.random() < 0.08:
        t = fc.Tag()                                   # unknown
    else:
        if slots is None:
            slots = 1 << r.randrange(0, 16)
        p = r.random()
        if p < 0.6:
            period = 1 << r.randrange(0, slots.bit_length())
        elif p < 0.8:
            period = slots
        elif p < 0.9:
            period = 0
        else:
            period = r.randrange(1, 3 * slots + 2)
        t = fc.Tag(slots, period, rnd_support(r, slots))
    t.field = r.randrange(0, 4)
    if r.random() < 0.15:
        t.conj_of = r.choice(CTS)
    return t


def rnd_scalar(r):
    return r.choice([0.0, 1.0, -1.0, 0.5, 2.5, -3.75, 1e-300, r.uniform(-10, 10)])


def main() -> None:
    out = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 20261018
    r = random.Random(seed)
    cases = []
    fns = ["support_ap", "support_block", "union", "intersect", "f_add", "f_mult", "lcm_capped", "with_field",
           "t_mult_scalar", "t_add_scalar", "t_square", "t_add", "t_mult", "t_mult_i", "t_rotate", "t_conjugate",
           "t_bootstrap", "conj_pair", "from_signature", "min_routable_s", "fold_collision_free_at", "periodic_at",
           "known", "top"]
    for i in range(n):
        f = fns[i % len(fns)] if i < 4 * len(fns) else r.choice(fns)
        same_slots = r.random() < 0.7
        if f == "support_ap":
            a = [r.randrange(-3, 40000), r.randrange(-1, 9000), r.randrange(-2, 40000)]
            o = enc_s(fc.Support.ap(*a))
        elif f == "support_block":
            a = [r.randrange(-3, 40000), r.randrange(-1, 9000), r.randrange(-1, 600), r.randrange(-2, 40000)]
            o = enc_s(fc.Support.block(*a))
        elif f in ("union", "intersect"):
            sl = 1 << r.randrange(0, 16)
            x, y = rnd_support(r, sl), rnd_support(r, sl)
            if r.random() < 0.3:                     # same stride, overlapping offsets: the merge branches
                y = fc.Support(fc.AP, r.randrange(0, 8), x.stride, r.randrange(0, 100), r.randrange(1, 9))
                x.kind = fc.AP
            a = [enc_s(x), enc_s(y)]
            o = enc_s(fc._union(x, y) if f == "union" else fc._intersect(x, y))
        elif f in ("f_add", "f_mult"):
            a = [r.randrange(0, 4), r.randrange(0, 4)]
            o = (fc._f_add if f == "f_add" else fc._f_mult)(*a)
        elif f == "lcm_capped":
            a = [r.randrange(-2, 70000), r.randrange(-2, 70000), r.choice([1, 2, 512, 32768, r.randrange(1, 70000)])]
            o = fc._lcm_capped(*a)
        elif f == "top":
            a = [1 << r.randrange(0, 16)]
            o = enc_t(fc.top(*a))
        elif f in ("t_add", "t_mult"):
            x = rnd_tag(r)
            y = rnd_tag(r, x.slots if (same_slots and x.slots) else None)
            a = [enc_t(x), enc_t(y)]
            o = enc_t((fc.t_add if f == "t_add" else fc.t_mult)(x, y))
        elif f == "conj_pair":
            x = rnd_tag(r)
            y = rnd_tag(r, x.slots if (same_slots and x.slots) else None)
            ia, ib = r.randrange(0, 4), r.randrange(0, 4)
            sub = r.random() < 0.5
            t = fc.t_add(x, y)
            a = [enc_t(t), ia, ib, enc_t(x), enc_t(y), sub]
            o = enc_t(fc.conj_pair(t, CTS[ia], CTS[ib], x, y, sub))
        elif f == "from_signature":
            sl = 1 << r.randrange(0, 16)
            live = r.choice([0, 1, 2, sl, sl + 1, r.randrange(0, sl + 2)])
            a = [sl, r.choice([0, 1, sl, 1 << r.randrange(0, 16)]), live, r.choice([0, 1, 2, 3, 64, r.randrange(0, sl + 1)]),
                 r.randrange(0, sl + 2), r.randrange(0, sl + 1)]
            o = enc_t(fc.from_signature(*a))
        elif f in ("min_routable_s",):
            x = rnd_tag(r)
            a = [enc_t(x)]
            o = x.min_routable_s()
        elif f in ("fold_collision_free_at", "periodic_at"):
            x = rnd_tag(r)
            s = 1 << r.randrange(0, 17) if r.random() < 0.8 else r.randrange(1, 70000)
            a = [enc_t(x), s]
            o = getattr(x, f)(s)
        elif f == "known":
            x = rnd_tag(r)
            a = [enc_t(x)]
            o = x.known()
        elif f in ("t_mult_scalar", "t_add_scalar"):
            x = rnd_tag(r)
            c = rnd_scalar(r)
            a = [enc_t(x), c]
            o = enc_t(getattr(fc, f)(x, c))
        elif f == "with_field":
            x = rnd_tag(r)
            a = [enc_t(x), r.randrange(0, 4)]
            o = enc_t(fc.with_field(x, a[1]))
        elif f == "t_rotate":
            x = rnd_tag(r)
            k = r.choice([0, 1, -1, 64, -2048, 32767, -32768, r.randrange(-100000, 100000)])
            a = [enc_t(x), k]
            o = enc_t(fc.t_rotate(x, k))
        elif f == "t_bootstrap":
            x = rnd_tag(r)
            a = [enc_t(x), r.choice([0, 1, 256, 512, 32768, -1]), r.choice([0, 32768, 512, -1, 16])]
            o = enc_t(fc.t_bootstrap(x, *a[1:]))
        else:                                          # unary: t_square, t_mult_i, t_conjugate
            x = rnd_tag(r)
            a = [enc_t(x)]
            o = enc_t(getattr(fc, f)(x))
        cases.append({"f": f, "args": a, "out": o})
    with open(out, "w") as fh:
        json.dump({"seed": seed, "n": n, "cases": cases}, fh, separators=(",", ":"))
    print(f"{n} cases -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
