"""Run one session of one track: the client and the server each in runner/sandbox.sh, the wire in /dev/shm.

    python3 -I runner/run.py --track {prefill,decode} --runs DIR [--requests N] [--offset K] [--warmup W]
                             [--gpu {N,none}] [--numa NODE[:MEMS]] [--bound S] [--mem 100G] [--client-mem 100G]
                             [--time 3600]
                             [--ro PATH]... [--allow DIR]... [--weights FILE] [--keep BYTES]
                             [--perseus DIR [--backend NAME] --prompts FILE --ref FILE]
                             -- SERVER_CMD ARGS...

The server command gets `--track T --inbox IN --outbox OUT` appended, and can read only the `--ro`
paths (its code), its inbox and `--weights` (GPT-2's weights in Perseus's format, named by
$WEIGHTS_PATH). The client reads the outbox and writes the inbox, nothing else of the server's.
`--perseus DIR` gives the client the crypto of that Perseus checkout: `--backend NAME` is
harness/NAME_client.py (`perseus`: record 0; `perseus-core`: record 1; an entry's own client),
with the prompts and the float64 reference's weights it checks against (harness/client.py); without it
the client is the dummy. The client holds the secret key and the prompts, so DIR must be at the commit
its client pins (PERSEUS) with its tracked files unchanged, never an entry's checkout. A bound
directory brings the targets of its top-level symlinks and its venv's interpreter; for the server's
`--ro` paths each must lie inside one of the client checkout's or an `--allow` directory. The run is
void if the server exits before the client quits, or is still running `--bound` seconds after the
client quits. `--numa` pins the server to a node and caps its CPU time at that node's CPUs. Each sandbox's
memory peak is sampled while it runs (systemd's own report at exit is stale). The sandboxes write to /dev/shm;
their logs and regular files, at most `--keep` bytes each side, are copied to DIR/<run>/ next to run.json (the
client's result and these checks).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(os.path.dirname(HERE), "harness")
SANDBOX = os.path.join(HERE, "sandbox.sh")
PIN = re.compile(r'^PERSEUS = "([0-9a-f]{40})"', re.M)
KEEP_FILES = 256


def sandbox(unit, gpu, out, mem, binds, cmd, log, a):
    return subprocess.Popen(
        ["bash", SANDBOX, "--gpu", gpu, "--out", out, "--unit", unit, "--mem", mem, "--time", str(a.time),
         *binds, "--", *cmd], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)


def node_cpus(numa):
    """How many CPUs the NUMA node of `numa` (NODE[:MEMS]) has."""
    with open(f"/sys/devices/system/node/node{int(numa.split(':')[0])}/cpulist", encoding="utf-8") as f:
        spans = [r.split("-") for r in f.read().strip().split(",")]
    return sum(int(r[-1]) - int(r[0]) + 1 for r in spans)


def git(path, *args):
    r = subprocess.run(["git", "-C", path, *args], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def unpinned(checkout, client):
    """Why `checkout` cannot be the client's Perseus (None if it can): it must be at the commit `client` pins with its
    tracked files unchanged."""
    with open(client, encoding="utf-8") as f:
        m = PIN.search(f.read())
    if not m:
        return f"{client} pins no Perseus commit (PERSEUS = \"<sha>\")"
    head = git(checkout, "rev-parse", "HEAD")
    if head != m.group(1):
        return (f"{os.path.basename(client)} runs Perseus {m.group(1)[:12]}, {checkout} is at "
                f"{(head or 'no commit')[:12]}: the client gets your build of that commit, never an entry's checkout")
    if git(checkout, "status", "--porcelain", "--untracked-files=no") != "":
        return f"{checkout} has changes to its tracked files"
    return None


def links(path):
    """What a checkout reaches outside itself: its top-level symlinks' targets and its venv's interpreter (a Perseus
    checkout links .venv, .local_deps, deps_n32 and .cache to directories outside it)."""
    out = []
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            link = os.path.join(path, name)
            if os.path.islink(link) and os.path.exists(link):
                out.append(os.path.realpath(link))
        cfg = os.path.join(path, ".venv", "pyvenv.cfg")
        if os.path.exists(cfg):
            with open(cfg, encoding="utf-8") as f:
                out += [os.path.dirname(v.strip()) for k, _, v in (ln.partition("=") for ln in f)
                        if k.strip() == "home"]
    return out


def inside(path, root):
    path, root = os.path.realpath(path), os.path.realpath(root)
    return os.path.commonpath([path, root]) == root


def ro(paths, allowed=None):
    """--ro binds for `paths` and what they link to. With `allowed` (the server's: an entry's code), each link must
    resolve inside one of those directories, so an entry cannot link its server to a file outside the sandbox."""
    out = []
    for path in map(os.path.abspath, paths):
        out.append(path)
        for target in links(path):
            if allowed is not None and not any(inside(target, d) for d in allowed):
                raise ValueError(f"{path} links to {target}, outside the client checkout's links and --allow")
            out.append(target)
    return [x for path in dict.fromkeys(out) for x in ("--ro", path)]


def keep(src, dst, budget):
    """Copy a regular file a sandbox wrote (on /dev/shm) to `dst`, or a directory's file by file, never through a
    symlink and within `budget` ([bytes, files], spent across calls): a sandbox cannot fill the disk of the runs."""
    if os.path.isdir(src) and not os.path.islink(src):
        for root, _dirs, files in os.walk(src):
            for name in sorted(files):
                path = os.path.join(root, name)
                keep(path, os.path.join(dst, os.path.relpath(path, src)), budget)
        return
    try:
        fd = os.open(src, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return
    with os.fdopen(fd, "rb") as f:
        if not stat.S_ISREG(os.fstat(f.fileno()).st_mode) or budget[1] <= 0:
            return
        data = f.read(budget[0])
    budget[0], budget[1] = budget[0] - len(data), budget[1] - 1
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "wb") as g:
        g.write(data)


def gpu_peak(gpu, done, peak):
    """peak[0]: the GPU's highest used memory (MiB, every process on it) until `done` is set."""
    while not done.is_set():
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", gpu],
                           capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            peak[0] = max(peak[0], int(r.stdout.split()[0]))
        done.wait(0.5)


def mem_peak(units, done, peak):
    """peak[name]: each unit's memory.peak (MiB), read from its cgroup until `done` is set."""
    files = {}
    while not done.is_set():
        for name, unit in units.items():
            if name not in files:
                cg = subprocess.run(["systemctl", "--user", "show", unit, "-p", "ControlGroup", "--value"],
                                    capture_output=True, text=True).stdout.strip()
                if cg:
                    files[name] = f"/sys/fs/cgroup{cg}/memory.peak"
            try:
                with open(files[name], encoding="utf-8") as f:
                    peak[name] = max(peak.get(name, 0), int(f.read()) >> 20)
            except (KeyError, OSError, ValueError):
                pass
        done.wait(0.1)


def stop(unit, proc):
    subprocess.run(["systemctl", "--user", "stop", unit], capture_output=True)
    proc.wait()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--track", choices=("prefill", "decode"), required=True)
    p.add_argument("--runs", required=True)
    p.add_argument("--requests", type=int, default=1)
    p.add_argument("--offset", type=int, default=0, help="request i takes prompt offset + i")
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--gpu", default="none")
    p.add_argument("--numa", help="the server's CPU node and memory nodes (sandbox.sh --numa)")
    p.add_argument("--bound", type=float, default=10.0)
    p.add_argument("--mem", default="100G", help="the server's memory cap")
    p.add_argument("--client-mem", help="the client's (default: --mem)")
    p.add_argument("--time", type=int, default=3600)
    p.add_argument("--ro", action="append", default=[])
    p.add_argument("--allow", action="append", default=[], help="a directory the server's --ro paths may link into")
    p.add_argument("--weights", help="GPT-2's weights for the server, read-only at $WEIGHTS_PATH")
    p.add_argument("--keep", type=int, default=64 << 20, help="bytes kept of each sandbox's log and files")
    p.add_argument("--perseus")
    p.add_argument("--backend", default="perseus")
    p.add_argument("--prompts")
    p.add_argument("--ref")
    p.add_argument("server", nargs="+")
    a = p.parse_args()
    client_cmd, client_ro, report = ["python3", "-I", f"{HARNESS}/client.py"], [HARNESS], {}
    if a.perseus:
        client = f"{HARNESS}/{a.backend.replace('-', '_')}_client.py"
        if not os.path.isfile(client):
            p.error(f"no {os.path.relpath(client, os.path.dirname(HARNESS))}")
        if why := unpinned(a.perseus, client):
            p.error(why)
        prompts, ref = os.path.abspath(a.prompts), os.path.realpath(a.ref)
        client_cmd = ["bash", f"{HARNESS}/with_perseus.sh", os.path.abspath(a.perseus), f"{HARNESS}/client.py",
                      "--backend", a.backend, "--prompts", prompts, "--ref", ref]
        client_ro += [a.perseus, prompts, ref]
        report["client_perseus"] = git(a.perseus, "rev-parse", "HEAD")
        with open(client, "rb") as f:
            report["client_file"] = hashlib.sha256(f.read()).hexdigest()
    try:
        server_ro = ro(a.ro, [*(links(os.path.abspath(a.perseus)) if a.perseus else []), *a.allow])
    except ValueError as e:
        p.error(str(e))
    if a.weights:
        weights = os.path.realpath(a.weights)
        if not os.path.isfile(weights):
            p.error(f"no weights at {a.weights}")
        server_ro += ["--ro", weights, "--setenv", "WEIGHTS_PATH", weights]

    run = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    shm = f"/dev/shm/speedrun-{run}"
    inbox, outbox = f"{shm}/in", f"{shm}/out"
    rundir = os.path.join(os.path.abspath(a.runs), run)
    os.makedirs(inbox, mode=0o700)
    os.makedirs(outbox, mode=0o700)
    os.makedirs(rundir)
    units = {"server": f"sbx-{run}-server", "client": f"sbx-{run}-client"}
    dirty = git(HERE, "status", "--porcelain", "--untracked-files=no", "--", HARNESS, HERE)
    report = {"run": run, "track": a.track, "server_cmd": a.server, "void": None, **report,
              "harness": f"{git(HERE, 'rev-parse', 'HEAD')}{'+dirty' if dirty else ''}"}
    procs = {}
    done, peak, mem = threading.Event(), [0], {}
    if a.gpu != "none":
        threading.Thread(target=gpu_peak, args=(a.gpu, done, peak), daemon=True).start()
    threading.Thread(target=mem_peak, args=(units, done, mem), daemon=True).start()
    try:
        with open(f"{shm}/server.log", "wb") as slog, open(f"{shm}/client.log", "wb") as clog:
            procs["server"] = sandbox(
                units["server"], a.gpu, f"{shm}/server", a.mem,
                ["--ro", inbox, "--rw", outbox, *server_ro,
                 *(["--numa", a.numa, "--cpus", str(node_cpus(a.numa))] if a.numa else [])],
                [*a.server, "--track", a.track, "--inbox", inbox, "--outbox", outbox], slog, a)
            procs["client"] = sandbox(
                units["client"], "none", f"{shm}/client", a.client_mem or a.mem,
                ["--rw", inbox, "--ro", outbox, *ro(client_ro)],
                [*client_cmd, "--track", a.track, "--inbox", inbox, "--outbox", outbox,
                 "--result", "/out/result.json", "--requests", str(a.requests), "--offset", str(a.offset),
                 "--warmup", str(a.warmup)], clog, a)
            server, client = procs["server"], procs["client"]
            while client.poll() is None:
                if server.poll() is not None and not os.path.exists(f"{inbox}/quit"):
                    report["void"] = f"server exited ({server.returncode}) before the client quit"
                    stop(units["client"], client)
                    break
                time.sleep(0.05)
            quit_at = time.monotonic()
            try:
                server.wait(timeout=a.bound)
                report["server_exit_s"] = round(time.monotonic() - quit_at, 3)
            except subprocess.TimeoutExpired:
                report["void"] = report["void"] or f"server still running {a.bound:g} s after quit"
                stop(units["server"], server)
            report["client_rc"], report["server_rc"] = client.returncode, server.returncode
            done.set()
            if a.gpu != "none":
                report["peak_gpu_mib"] = peak[0]
            report["peak_mem_mib"] = mem
    finally:
        for name, proc in procs.items():
            if proc.poll() is None:
                stop(units[name], proc)
        for side in ("server", "client"):
            budget = [a.keep, KEEP_FILES]
            keep(f"{shm}/{side}.log", f"{rundir}/{side}.log", budget)
            keep(f"{shm}/{side}", f"{rundir}/{side}", budget)
        shutil.rmtree(shm, ignore_errors=True)
    try:
        with open(f"{rundir}/client/result.json", encoding="utf-8") as f:
            report["result"] = json.loads(f.read(1 << 24))
    except (OSError, ValueError) as e:
        report["result"] = {"error": f"no client result: {e}"}
    with open(f"{rundir}/run.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)

    res = report["result"]
    for i, r in enumerate(res.get("requests", [])):
        print(f"request {i}{' (warm-up)' if r['warmup'] else ''}: score {r['score']:.4f}"
              f"{' s' if a.track == 'prefill' else ' s/token'}, outputs {'ok' if r['outputs_ok'] else 'WRONG'}"
              + (f", mean KL {r['kl_scored_mean']:.2e}" if "kl_scored_mean" in r else ""))
    if report["void"]:
        print(f"void: {report['void']}")
    if "error" in res:
        print(f"error: {res['error']}")
    print(f"run.json: {rundir}/run.json")
    ok = report["void"] is None and "error" not in res and report["client_rc"] == 0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
