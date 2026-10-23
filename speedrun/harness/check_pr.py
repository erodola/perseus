"""The check every Perseus PR gets before a maintainer is asked to review it (.github/workflows/speedrun-pr-check.yml).

    python3 speedrun/harness/check_pr.py --base BASE_SHA --head HEAD_REF [--author LOGIN]

It reads the PR through git only (`git diff`, `git show`), never running its code. Every PR leaves the speedrun's
harness, rules and records and the repository's automation alone (PROTECTED), unless a maintainer wrote it. A PR that
adds a record folder `speedrun/records/<YYYY-MM-DD>_<Name>/` is an entry: its change to Perseus plus that one folder
(README.md with a Disclosures section, submission.json) and, for an entry with its own key or ciphertext formats, at
most one new client `speedrun/harness/<name>_client.py`. The client runs the commit its client file pins, so an entry
that changes the client's code brings its own client, pinned to a commit of the PR with the head's client code.
submission.json must have the template's fields, and its CKKS parameters must reach 128-bit security on the HE
standard's table (ternary secret, classic estimates): log QP within the bound of the ring dimension. The sparse keys
it declares are listed for the maintainers' Lattice Estimator run, which this check does not do. Prints a Markdown
report (nothing for a clean PR that is not an entry); exit 0 = pass. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

# HE standard, ternary secret, classic 128-bit (OpenFHE stdlatticeparms): ring dimension -> max log QP
HE_STD_128 = {1024: 27, 2048: 54, 4096: 109, 8192: 218, 16384: 438, 32768: 881, 65536: 1747, 131072: 3523}
FOLDER = re.compile(r"^speedrun/records/(\d{4}-\d{2}-\d{2}_[A-Za-z0-9][A-Za-z0-9-]*)/")
CLIENT = re.compile(r"^speedrun/harness/[a-z0-9_]+_client\.py$")
PIN = re.compile(r'^PERSEUS = "([0-9a-f]{40})"', re.M)
MAINTAINERS = ("alexzilligmm",)
PROTECTED = ("speedrun/harness/", "speedrun/runner/", "speedrun/prompts/", "speedrun/records/", "speedrun/README.md",
             "speedrun/CLAUDE.md", "speedrun/RULINGS.md", "speedrun/LICENSE", "speedrun/.gitignore", ".github/",
             "LICENSE")
CLIENT_CODE = ("crates/pc-client/", "python/perseus/client/")   # keygen, encryption, decryption: the client runs them
FRAMING = ("speedrun/record1.py",)                              # record 1's output framing, next to its plan constants


def git(*args) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def ancestor(a: str, b: str) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", a, b], capture_output=True).returncode == 0


def records(rev: str) -> set[str]:
    """The record folders at `rev`."""
    try:
        return set(git("ls-tree", "--name-only", f"{rev}:speedrun/records").split())
    except subprocess.CalledProcessError:
        return set()


def check(base: str, head: str, author: str = "") -> tuple[list[str], list[str], bool]:
    """(errors, notes for the reviewer, whether the PR is an entry)."""
    errors, notes = [], []
    changes = [ln.split("\t") for ln in git("diff", "--no-renames", "--name-status", f"{base}...{head}").splitlines()]
    folders = {m.group(1) for s, path, *_ in changes if (m := FOLDER.match(path)) and s == "A"} - records(base)
    clients = [path for s, path, *_ in changes if CLIENT.match(path) and s == "A"] if folders else []
    for status, path, *_ in changes:
        m = FOLDER.match(path)
        if not path.startswith(PROTECTED) or path in clients or (m and m.group(1) in folders and status == "A"):
            continue
        if author in MAINTAINERS:
            notes.append(f"`{path}` ({status}): protected, changed by a maintainer")
        else:
            errors.append(f"`{path}` ({status}): the speedrun's harness, rules and records and the repository's "
                          "automation are the maintainers'")
    if not folders or (len(folders) > 1 and author in MAINTAINERS):     # a maintainer's records, not an entry
        return errors, notes, False
    notes.insert(0, f"checked: this PR at `{git('rev-parse', head).strip()}`. The maintainers review and time this "
                    "commit; a new push runs the check again")
    if len(folders) != 1:
        return errors + [f"the PR adds {len(folders)} record folders, not one"], notes, True
    folder = f"speedrun/records/{folders.pop()}"
    if len(clients) > 1:
        errors.append(f"{len(clients)} client files: one backend brings one client")
    for c in clients:
        notes.append(f"new client `{c}`: it holds the secret key, review it like security code")
    try:
        readme = git("show", f"{head}:{folder}/README.md")
        if "## Disclosures" not in readme:
            errors.append(f"`{folder}/README.md` has no Disclosures section")
    except subprocess.CalledProcessError:
        errors.append(f"`{folder}/README.md` is missing")
    try:
        sub = json.loads(git("show", f"{head}:{folder}/submission.json"))
    except (subprocess.CalledProcessError, ValueError) as e:
        return errors + [f"`{folder}/submission.json` is missing or not JSON ({type(e).__name__})"], notes, True
    errors += fields(sub)
    e, n = security(sub.get("ckks", {}))
    e2, n2 = client_code(base, head, clients)
    return errors + e + e2, notes + n + n2, True


def client_code(base, head, clients) -> tuple[list[str], list[str]]:
    """The PR's changes to the client's code: they need the entry's own client, pinned to a commit of the PR whose
    client code is the head's."""
    errors, notes = [], []
    code = git("diff", "--name-only", f"{base}...{head}", "--", *CLIENT_CODE, *FRAMING).splitlines()
    for c in clients:
        own = PIN.search(git("show", f"{head}:{c}"))
        if not own:
            errors.append(f"`{c}` pins no commit: `PERSEUS = \"<sha>\"`, a commit of this PR")
        elif not ancestor(own.group(1), head) or ancestor(own.group(1), base):
            errors.append(f"`{c}` pins `{own.group(1)[:12]}`, which is not a commit of this PR")
        elif git("diff", "--name-only", own.group(1), head, "--", *CLIENT_CODE, *FRAMING).strip():
            errors.append(f"`{c}` pins `{own.group(1)[:12]}`, whose client code differs from the head's")
    if any(f.startswith(CLIENT_CODE) for f in code) and not clients:
        errors.append(f"the PR changes the client's code ({', '.join(f'`{f}`' for f in code)}), which the client runs "
                      "at its pinned commit: bring your own client, `speedrun/harness/<name>_client.py`")
    if code:
        notes.append(f"client-side changes, for the client's review: {', '.join(f'`{f}`' for f in code)}")
    return errors, notes


def fields(sub: dict) -> list[str]:
    errors = []
    if not isinstance(sub.get("name"), str) or not sub["name"]:
        errors.append("`name` is missing")
    tracks = sub.get("tracks")
    if not isinstance(tracks, list) or not tracks or not set(tracks) <= {"prefill", "decode"}:
        errors.append("`tracks` must be a non-empty list of prefill and decode")
    authors = sub.get("authors")
    if not isinstance(authors, list) or not authors or not all(isinstance(a, dict) and a.get("github")
                                                                for a in authors):
        errors.append("`authors` must list each author with a `github` handle")
    for k in ("summary", "dev_result", "ckks"):
        if k not in sub:
            errors.append(f"`{k}` is missing")
    return errors


def security(ckks: dict) -> tuple[list[str], list[str]]:
    try:
        n, log_qp, secret = 1 << int(ckks["log_n"]), int(ckks["log_qp"]), ckks["secret"]
    except (KeyError, TypeError, ValueError):
        return ["`ckks` must declare `log_n`, `log_qp` and `secret`"], []
    errors, notes = [], []
    if secret not in ("uniform-ternary", "ternary"):
        errors.append(f"main secret `{secret}`: the automatic check covers ternary secrets only")
    bound = HE_STD_128.get(n)
    if bound is None:
        errors.append(f"ring dimension 2^{ckks['log_n']} is not in the HE standard's table")
    elif log_qp > bound:
        errors.append(f"log QP {log_qp} > {bound}, the 128-bit bound at N = 2^{ckks['log_n']}")
    else:
        notes.append(f"main key: N = 2^{ckks['log_n']}, log QP {log_qp} <= {bound} (HE standard, 128-bit classic)")
    for k in ckks.get("sparse_keys", []):
        notes.append(f"sparse key for the Lattice Estimator: {json.dumps(k)}")
    return errors, notes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--head", required=True)
    p.add_argument("--author", default="", help="the PR author's GitHub login")
    a = p.parse_args()
    errors, notes, entry = check(a.base, a.head, a.author)
    if not entry and not errors:
        return 0
    print("<!-- speedrun-pr-check -->")
    print("**PR check: " + ("passed.** A maintainer is asked to review." if not errors else "failed.**") + "\n")
    for e in errors:
        print(f"- :x: {e}")
    for n in notes:
        print(f"- {n}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
