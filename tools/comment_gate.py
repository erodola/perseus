"""Comment gate: comments state facts about the code, not the process that produced it.

Fails on development-process words in comments and docstrings (task-card ids, the run log, the predecessor
implementation's history; BANNED_* below) and on module headers longer than MAX_HEADER lines. Rust/CUDA: `//` comments;
Python: `#` comments and docstrings. String literals and code are not checked. Usage: comment_gate.py [-v] [root]
"""
import ast
import io
import pathlib
import re
import subprocess
import sys
import tokenize

# case-sensitive: "m8" in a plan name or "plan" in prose is fine
BANNED_CS = re.compile(r"\bM[0-9]+[a-z]?\b|\bPLAN\b|\bRUNLOG\b")
BANNED_CI = re.compile(r"prototype|orchestrator|successor|\bold stack\b", re.IGNORECASE)
MAX_HEADER = 12
# scripts/: research and reproduction utilities, outside the package
EXCLUDE = ("target/", ".venv/", "bootstrap_placements/", "graphs/", "scripts/")
CODE_EXT = (".rs", ".cu", ".cuh", ".h")


def banned(text: str) -> str | None:
    m = BANNED_CS.search(text) or BANNED_CI.search(text)
    return m.group(0) if m else None


def strip_strings(line: str) -> str:
    return re.sub(r'"(?:\\.|[^"\\])*"', '""', line)


def code_comments(path: pathlib.Path):
    lines = path.read_text(errors="replace").splitlines()
    for i, line in enumerate(lines, 1):
        s = strip_strings(line)
        k = s.find("//")
        if k >= 0:
            yield i, s[k:]
    marker = "//!" if path.suffix == ".rs" else "//"
    body = [x.strip() for x in lines if x.strip() != "#pragma once"]
    header = 0
    while header < len(body) and body[header].startswith(marker):
        header += 1
    yield None, header


def py_comments(path: pathlib.Path):
    src = path.read_text(errors="replace")
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                yield tok.start[0], tok.string
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    header = 0
    try:
        tree = ast.parse(src)
    except SyntaxError:
        yield None, 0
        return
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc and node.body:
                first = node.body[0].lineno
                for j, dl in enumerate(doc.splitlines()):
                    yield first + j, dl
                if isinstance(node, ast.Module):
                    header = len(doc.strip().splitlines())
    yield None, header


def main() -> int:
    verbose = "-v" in sys.argv
    args = [a for a in sys.argv[1:] if a != "-v"]
    root = pathlib.Path(args[0] if args else ".").resolve()
    hits, long_headers = [], []
    try:  # tracked files only (scratch output and build trees are not code)
        listed = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True).stdout
        paths = [root / f for f in listed.splitlines()]
    except (OSError, subprocess.CalledProcessError):
        paths = list(root.rglob("*"))
    for path in sorted(paths):
        rel = path.relative_to(root).as_posix()
        if not path.is_file() or any(rel.startswith(e) or f"/{e}" in rel for e in EXCLUDE):
            continue
        if path.suffix in CODE_EXT:
            gen = code_comments(path)
        elif path.suffix == ".py":
            gen = py_comments(path)
        else:
            continue
        for line, text in gen:
            if line is None:
                if text > MAX_HEADER:
                    long_headers.append(f"{rel}: module header {text} lines (max {MAX_HEADER})")
                continue
            w = banned(text)
            if w:
                hits.append(f"{rel}:{line}: [{w}] {text.strip()}")
    for h in hits + long_headers:
        if verbose:
            print(h)
    print(f"comment gate: {len(hits)} process references, {len(long_headers)} long module headers")
    return 1 if hits or long_headers else 0


if __name__ == "__main__":
    sys.exit(main())
