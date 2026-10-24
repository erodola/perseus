"""The check every PR to Perseus's speedrun branch gets before a maintainer reviews it (speedrun-pr-check.yml).

    python3 speedrun/harness/check_pr.py --base BASE_SHA --head HEAD_REF [--author LOGIN] [--body FILE]

It reads the PR through git only (`git diff`, `git show`), never running its code. Every PR leaves the speedrun's
harness, rules and records and the repository's automation alone (PROTECTED), unless a maintainer wrote it. A PR that
adds a record folder `speedrun/records/<YYYY-MM-DD>_<Name>/` is an entry: its change to Perseus plus that one folder
(README.md with a Disclosures section, submission.json) and, for an entry with its own key or ciphertext formats, at
most one new client `speedrun/harness/<name>_client.py`. The client runs the commit its client file pins, so an entry
that changes the client's code brings its own client, pinned to the commit before the one that adds the record folder
and the client: the maintainers build that commit, as the server and the client. The report names the commit they
time. submission.json must have the template's fields, and its CKKS parameters must reach 128-bit security on the HE
standard's table (ternary secret, classic estimates): log QP within the bound of the ring dimension. The report says
whether its keys are an existing record's or need the maintainers' Lattice Estimator run, and whether building the
entry runs or reads anything of its own beyond the compiler's input (`build_safety`). An entry's description
(`--body`) accepts the CLA (CLA.md). Prints a Markdown report (nothing for a clean PR that is not an entry), in which
text from the PR appears only as inline code (`code`); exit 0 = pass, and a PR the check cannot read fails it.
Standard library only.
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
CLA = "i have read the perseus cla and agree to it"
PROTECTED = ("speedrun/harness/", "speedrun/runner/", "speedrun/prompts/", "speedrun/records/", "speedrun/MAINTAINING.md",
             "speedrun/RULINGS.md", "speedrun/LICENSE", "speedrun/.gitignore", "README.md", "CLAUDE.md", ".github/",
             "LICENSE", "CLA.md")
# What a build (runner/build.sh: uv pip install, maturin) runs or reads of the commit besides the compiler's input: a PR
# that changes them is built only where an entry may run (speedrun/MAINTAINING.md)
BUILD_FILES = re.compile(r"(^|/)(pyproject\.toml|setup\.(py|cfg)|requirements[^/]*\.txt|MANIFEST\.in|Cargo\.(toml|lock)|"
                         r"build\.rs|rust-toolchain(\.toml)?|\.gitmodules|\.gitattributes)$|^\.cargo/")
COMPILE_TIME = re.compile(r"include_(str|bytes)!|\binclude!\s*\(|\b(option_)?env!\s*\(\s*\"(?!CARGO_)|"
                          r"#\s*include\s*[<\"]\s*(/|\.\./)|#\s*embed\b|[\"'](/(data\w*|home|root|mnt|srv)\b|~/)")
SOURCES = (".rs", ".cu", ".cuh", ".h", ".hpp", ".c", ".cc", ".cpp", ".py", ".sh")
CLIENT_CODE = ("crates/pc-client/", "python/perseus/client/")   # keygen, encryption, decryption: the client runs them
FRAMING = ("speedrun/record1.py",)                              # record 1's output framing, next to its plan constants


def code(text) -> str:
    """Text from the PR as one inline code span: no backtick or line break can end it, so nothing in it renders as
    Markdown or mentions anyone."""
    return "`" + " ".join(str(text).replace("`", "'").split())[:200] + "`"


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


def check(base: str, head: str, author: str = "", body: str | None = None) -> tuple[list[str], list[str], bool]:
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
            notes.append(f"{code(path)} ({status}): protected, changed by a maintainer")
        else:
            errors.append(f"{code(path)} ({status}): the speedrun's harness, rules and records and the repository's "
                          "automation are the maintainers'")
    if not folders or (len(folders) > 1 and author in MAINTAINERS):     # a maintainer's records, not an entry
        return errors, notes, False
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
    if body is None:
        notes.append(f"the PR's description must accept the CLA: \"{CLA[0].upper() + CLA[1:]} for my contributions.\"")
    elif CLA not in " ".join(body.lower().split()):
        errors.append("the PR's description does not accept the CLA (CLA.md): \"I have read the Perseus CLA and agree "
                      "to it for my contributions.\"")
    e, n = security(sub.get("ckks", {}))
    e2, n2, timed = client_code(base, head, clients, folder)
    same = [r for r in sorted(records(base)) if declared(base, r) == sub.get("ckks")]
    n.append(f"keys: the CKKS parameters and sparse keys of record `{same[0]}`, no new estimate" if same else
             "keys: new or changed CKKS parameters or sparse keys, for the Lattice Estimator")
    n.append(build_safety(base, head))
    notes.insert(0, f"checked: this PR at `{git('rev-parse', head).strip()}`; the maintainers time commit `{timed}`. "
                    "A new push runs the check again")
    return errors + e + e2, notes + n + n2, True


def build_safety(base: str, head: str) -> str:
    """Whether building the PR runs or reads anything of its own beyond the compiler's input: build and dependency files,
    symlinks, binary files, changed proc-macro crates, compile-time file inclusion, compile-time environment reads
    (other than Cargo's own) and absolute paths into home or data directories in its sources."""
    found = []
    for line in git("diff", "--no-renames", "--raw", f"{base}...{head}").splitlines():
        meta, path = line.split("\t", 1)
        if BUILD_FILES.search(path):
            found.append(code(path))
        if meta.split()[1] == "120000":
            found.append(f"{code(path)} (a symlink)")
    for line in git("diff", "--no-renames", "--numstat", f"{base}...{head}").splitlines():
        added, _deleted, path = line.split("\t", 2)
        if added == "-":
            found.append(f"{code(path)} (binary)")
    changed = git("diff", "--no-renames", "--name-only", f"{base}...{head}").splitlines()
    for crate in sorted({p.split("/")[1] for p in changed if p.startswith("crates/") and p.count("/") > 1}):
        try:
            if re.search(r"proc-macro\s*=\s*true", git("show", f"{head}:crates/{crate}/Cargo.toml")):
                found.append(f"{code('crates/' + crate)} (a proc-macro crate)")
        except subprocess.CalledProcessError:
            pass
    path, number = None, 0
    for line in git("diff", "--no-renames", "-U0", f"{base}...{head}").splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else None
        elif line.startswith("@@"):
            number = int(re.search(r"\+(\d+)", line).group(1))
        elif line.startswith("+"):
            if path and path.endswith(SOURCES) and COMPILE_TIME.search(line):
                found.append(f"{code(f'{path}:{number}')} (compile-time inclusion, environment or absolute path)")
            number += 1
    if not found:
        return ("build safety: building it runs no code of the PR (no build or dependency file, symlink, binary, "
                "proc-macro crate or compile-time inclusion changed)")
    more = f" and {len(found) - 12} more" if len(found) > 12 else ""
    return f"build safety: build it only in the isolated account: {', '.join(found[:12])}{more}"


def declared(rev: str, record: str):
    """A record's declared `ckks` block at `rev` (None if it has none)."""
    try:
        return json.loads(git("show", f"{rev}:speedrun/records/{record}/submission.json")).get("ckks")
    except (subprocess.CalledProcessError, ValueError, AttributeError):
        return None


def client_code(base, head, clients, folder) -> tuple[list[str], list[str], str]:
    """The PR's changes to the client's code, which need the entry's own client, and the commit the maintainers time:
    the head, or the commit the client pins, from which the head differs only by the record folder and the client."""
    errors, notes, timed = [], [], git("rev-parse", head).strip()
    changed = git("diff", "--name-only", f"{base}...{head}", "--", *CLIENT_CODE, *FRAMING).splitlines()
    for c in clients:
        own = PIN.search(git("show", f"{head}:{c}"))
        if not own:
            errors.append(f"`{c}` pins no commit: `PERSEUS = \"<sha>\"`, a commit of this PR")
        elif not ancestor(own.group(1), head) or ancestor(own.group(1), base):
            errors.append(f"`{c}` pins `{own.group(1)[:12]}`, which is not a commit of this PR")
        elif set(git("diff", "--name-only", own.group(1), head).splitlines()) - {c} - set(
                git("ls-tree", "-r", "--name-only", head, f"{folder}/").splitlines()):
            errors.append(f"`{c}` pins `{own.group(1)[:12]}`: the PR's later commits may only add the record folder "
                          "and the client")
        else:
            timed = own.group(1)
    if any(f.startswith(CLIENT_CODE) for f in changed) and not clients:
        errors.append(f"the PR changes the client's code ({', '.join(code(f) for f in changed)}), which the client "
                      "runs at its pinned commit: bring your own client, `speedrun/harness/<name>_client.py`")
    if changed:
        notes.append(f"client-side changes, for the client's review: {', '.join(code(f) for f in changed)}")
    return errors, notes, timed


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
        errors.append(f"main secret {code(secret)}: the automatic check covers ternary secrets only")
    bound = HE_STD_128.get(n)
    if bound is None:
        errors.append(f"ring dimension 2^{n.bit_length() - 1} is not in the HE standard's table")
    elif log_qp > bound:
        errors.append(f"log QP {log_qp} > {bound}, the 128-bit bound at N = 2^{n.bit_length() - 1}")
    else:
        notes.append(f"main key: N = 2^{n.bit_length() - 1}, log QP {log_qp} <= {bound} (HE standard, 128-bit classic)")
    for k in ckks.get("sparse_keys", []):
        notes.append(f"sparse key for the Lattice Estimator: {code(json.dumps(k))}")
    return errors, notes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--head", required=True)
    p.add_argument("--author", default="", help="the PR author's GitHub login")
    p.add_argument("--body", help="a file holding the PR's description")
    a = p.parse_args()
    body = None
    if a.body:
        with open(a.body, encoding="utf-8", errors="replace") as f:
            body = f.read()
    try:
        errors, notes, entry = check(a.base, a.head, a.author, body)
    except Exception as e:  # noqa: BLE001 (a PR the check cannot read fails it, never passes)
        errors, notes, entry = [f"the check could not read this PR: {code(type(e).__name__ + ': ' + str(e))}"], [], True
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
