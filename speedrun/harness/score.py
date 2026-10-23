"""Score an entry against the current record from their runs (runner/run.py's run.json files). Standard library only.

    python3 -I harness/score.py --track {prefill,decode} --entry RUN.json... [--record RUN.json...] [--json OUT]
                                [--official]

Quality, on the entry's runs: the mean KL of every scored request (non-warm-up) must be under the track's bar
(protocol.TRACKS: Prefill-128 1e-2, Decode-8 1e-3), by a one-sided t-test at p < 0.01, and the CutMax argmax must
agree with the logits' own on at least 95 % of the scored positions (a near tie may flip). Speed, with --record: the
runs are paired in order (record and entry interleaved on one GPU, a pair on the same prompts, which is checked); with
r_i = ln(t_record,i / t_entry,i), the entry is a new record when mean(r) >= ln(1 + delta), delta = max(1 %, 3 standard
errors), and a one-sided paired t-test gives p < 0.01. A run's time is the mean score of its scored requests.
--official refuses any run not on the hidden test set (the hash in prompts/test.sha256).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protocol import TRACKS  # noqa: E402

P_MAX, ARGMAX_MIN, DELTA_MIN = 0.01, 0.95, 0.01
TEST_SHA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts", "test.sha256")


def _betacf(a, b, x):
    """The continued fraction of the regularized incomplete beta function (Lentz)."""
    tiny, c, d = 1e-300, 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        for num in (m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
                    -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))):
            d = 1.0 + num * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + num / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < 1e-15:
            break
    return h


def t_sf(t, df):
    """P(T > t) for Student's t with df degrees of freedom: I_x(df/2, 1/2) / 2 at x = df / (df + t^2)."""
    if t == 0:
        return 0.5
    a, b, x = df / 2, 0.5, df / (df + t * t)
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    ib = front * _betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1.0 - front * _betacf(b, a, 1.0 - x) / b
    return 0.5 * ib if t > 0 else 1.0 - 0.5 * ib


def mean_se(xs):
    n = len(xs)
    m = sum(xs) / n
    return m, (math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n) if n > 1 else float("inf"))


def scored(run):
    reqs = [r for r in run["result"].get("requests", []) if not r["warmup"]]
    if run.get("void") or run["result"].get("error") or not reqs:
        raise ValueError(f"run {run.get('run')}: void, failed or without scored requests")
    return reqs


def prompts(run):
    """What a run was scored on: its prompt set's hash and the prompt of each scored request."""
    return run["result"].get("prompts_sha256"), [r.get("prompt") for r in scored(run)]


def official(runs):
    """Official runs: on the hidden test set, all by one unmodified harness commit."""
    with open(TEST_SHA, encoding="utf-8") as f:
        want = f.read().split()[0]
    other = [run.get("run") for run in runs if run["result"].get("prompts_sha256") != want]
    if other:
        raise ValueError(f"runs {other} are not on the hidden test set ({TEST_SHA})")
    harness = sorted({str(run.get("harness")) for run in runs})
    if len(harness) != 1 or not re.fullmatch(r"[0-9a-f]{40}", harness[0]):
        raise ValueError(f"the runs come from harness commits {harness}: one, without local changes")


def quality(track, runs):
    t = TRACKS[track]
    kls, agree = [], []
    for run in runs:
        for r in scored(run):
            kls.append(r["kl_scored_mean"])
            agree += [r["argmax_ok"][j] for j in t.scored]
    m, se = mean_se(kls)
    p = t_sf((t.kl_max - m) / se, len(kls) - 1) if len(kls) > 1 and se > 0 else (0.0 if m < t.kl_max else 1.0)
    share = sum(agree) / len(agree)
    return {"requests": len(kls), "kl_mean": m, "kl_max": max(kls), "kl_bar": t.kl_max, "p": p,
            "argmax_agreement": share, "pass": p < P_MAX and share >= ARGMAX_MIN}


def speed(entry, record):
    if len(entry) != len(record) or len(entry) < 2:
        raise ValueError(f"{len(record)} record and {len(entry)} entry runs: pair at least 2 of each")
    for i, (e, r) in enumerate(zip(entry, record)):
        if prompts(e) != prompts(r):
            raise ValueError(f"pair {i}: the record ran prompts {prompts(r)}, the entry {prompts(e)}")
    run_t = lambda run: sum(r["score"] for r in scored(run)) / len(scored(run))
    te, tr = [run_t(r) for r in entry], [run_t(r) for r in record]
    rs = [math.log(a / b) for a, b in zip(tr, te)]
    m, se = mean_se(rs)
    delta = max(DELTA_MIN, 3 * se) if math.isfinite(se) else float("inf")
    p = t_sf(m / se, len(rs) - 1) if se > 0 else (0.0 if m > 0 else 1.0)
    return {"pairs": len(rs), "entry_s": sum(te) / len(te), "record_s": sum(tr) / len(tr),
            "speedup": math.exp(m), "delta": delta, "p": p, "new_record": m >= math.log1p(delta) and p < P_MAX}


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--track", choices=TRACKS, required=True)
    a.add_argument("--entry", nargs="+", required=True)
    a.add_argument("--record", nargs="*", default=[])
    a.add_argument("--json")
    a.add_argument("--official", action="store_true", help="every run must be on the hidden test set")
    a = a.parse_args()
    load = lambda ps: [r for r in (json.load(open(p, encoding="utf-8")) for p in ps) if r.get("track") == a.track]
    entry = load(a.entry)
    if a.official:
        official(entry + load(a.record))
    out = {"track": a.track, "quality": quality(a.track, entry)}
    if a.record:
        out["speed"] = speed(entry, load(a.record))
    q = out["quality"]
    print(f"quality: {q['requests']} requests, mean KL {q['kl_mean']:.3g} (bar {q['kl_bar']:g}, p {q['p']:.2g}), "
          f"argmax agreement {q['argmax_agreement']:.1%}: {'PASS' if q['pass'] else 'FAIL'}")
    if "speed" in out:
        s = out["speed"]
        print(f"speed: {s['pairs']} pairs, entry {s['entry_s']:.4g} s vs record {s['record_s']:.4g} s, "
              f"x{s['speedup']:.3f} (delta {s['delta']:.1%}, p {s['p']:.2g}): "
              f"{'NEW RECORD' if s['new_record'] else 'not a record'}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
    return 0 if q["pass"] and out.get("speed", {}).get("new_record", True) else 1


if __name__ == "__main__":
    sys.exit(main())
