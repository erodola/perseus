"""Compare two plans field by field, from git objects only: what an entry's regenerated plan changes.

    python3 -I speedrun/runner/plan_diff.py REV_A PLAN_DIR_A REV_B PLAN_DIR_B

Every block's placement JSON of plan A against plan B's (a file only one of them has is listed), and their recipes
(PLAN_CMD.txt) line by line. Fields under `summary` are the planner's report on its plan (solve times, histograms,
predicted errors) and are only counted; every other differing field is printed. Exit 0 when nothing but `summary`
differs. Standard library only.
"""
from __future__ import annotations

import difflib
import json
import subprocess
import sys


def git(*args) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def files(rev: str, plan: str) -> set[str]:
    return set(git("ls-tree", "--name-only", f"{rev}:{plan}").split())


def diff(a, b, path=""):
    """(path, a, b) for every leaf where a and b differ."""
    if type(a) is not type(b):
        yield path, a, b
    elif isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                yield f"{path}.{k}", a.get(k, "<absent>"), b.get(k, "<absent>")
            else:
                yield from diff(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list) and len(a) != len(b):
        yield path, f"{len(a)} items", f"{len(b)} items"
    elif isinstance(a, list):
        for i, (x, y) in enumerate(zip(a, b)):
            yield from diff(x, y, f"{path}[{i}]")
    elif a != b:
        yield path, a, b


def main() -> int:
    rev_a, plan_a, rev_b, plan_b = sys.argv[1:5]
    fa, fb = files(rev_a, plan_a), files(rev_b, plan_b)
    real = 0
    for name in sorted(fa ^ fb):
        print(f"{name}: only in {'A' if name in fa else 'B'}")
        real += 1
    for name in sorted(fa & fb):
        a, b = git("show", f"{rev_a}:{plan_a}/{name}"), git("show", f"{rev_b}:{plan_b}/{name}")
        if not name.endswith(".json"):
            lines = [ln for ln in difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=0)
                     if ln[:1] in "+-" and ln[:3] not in ("+++", "---")]
            for ln in lines:
                print(f"{name}: {ln[:200]}")
            continue
        found = list(diff(json.loads(a), json.loads(b)))
        plan = [d for d in found if not d[0].startswith(".summary")]
        real += len(plan)
        if found:
            print(f"{name}: {len(plan)} plan fields differ, {len(found) - len(plan)} summary fields")
        for path, x, y in plan:
            print(f"   {path}: {str(x)[:80]} -> {str(y)[:80]}")
    print(f"plan fields that differ: {real}")
    return 1 if real else 0


if __name__ == "__main__":
    sys.exit(main())
