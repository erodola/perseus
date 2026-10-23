# Contributing

## Setup

```bash
git clone https://github.com/gladia-research-group/perseus
cd perseus
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[hf,examples,dev]"
.venv/bin/maturin develop --release        # the Rust + CUDA extension perseus._pc (README: Install)
```

`perseus.plan`, `perseus.calibrate`, `perseus.hub` and `perseus.export` run without the extension; the models, the
recorder and the runtime need it.

## Licence of contributions

Perseus is BUSL-1.1 and also licensed commercially, so every contribution is made under the
[Contributor License Agreement](CLA.md): you keep your copyright and grant the Licensors the right to distribute your
work under any terms. Accept it in your pull request's description: "I have read the Perseus CLA and agree to it for
my contributions." A speedrun entry's PR check requires that sentence.

## Tests

```bash
tools/ci.sh                                # the repo gate: comment gate, Rust build + tests, CPU tier, GPU smoke
.venv/bin/python -m pytest                 # the Python CPU tier only (seconds to minutes, no GPU, no network)
PERSEUS_ALL_PLANS=1 .venv/bin/python -m pytest python/tests/test_paper_plans.py   # every shipped plan regenerates
.venv/bin/python -m pytest -m gpu python/tests/gpu -q -s                         # the GPU tier
```

`CI_GPU=0 tools/ci.sh` skips the GPU smoke. GPU jobs go through `tools/gpu.sh <log> <cmd>` (timeout, log, exit code);
by default the runtime takes the first device in `CUDA_VISIBLE_DEVICES`.

Shared machines can opt into a guard: with `PERSEUS_GPU_GUARD=<gpu>` set, `tools/gpu.sh` pins the job to that GPU,
waits for it to be idle under a machine-wide `flock` (`PERSEUS_GPU_LOCK`, default `/tmp/perseus-gpu<gpu>.lock`),
kills the job if it touches another GPU, and exports `PERSEUS_CORE_LOCKED=1`; the runtime (`pc-cuda` `Device::open`)
then refuses to select a device unless exactly that GPU is visible and the lock variable is set, so a GPU call from a
plain shell fails instead of landing on someone else's GPU. On the lab box, `source tools/agent.sh` sets the guard
(GPU 3) together with the thread budgets. The Rust oracle gates (`pc-check`, the `oracle_*` tests of `pc-core`)
compare against dumps of the C++ runtime that are not in the repository; they skip or fail without them.

## Conventions

- Rust 2021, the toolchain pinned by `rust-toolchain.toml`; CUDA kernels are C++ templates in
  `crates/pc-kernels/kernels/*.cuh` behind `extern "C"` launchers. No process-global state; device memory comes from
  one arena (`pc-gpu/src/mem.rs`), frozen after setup.
- Comments say why, describe the code as it is, and keep module headers short (`tools/comment_gate.py`).
- Plans are bound to the runtime and the calibration: a change to an op's math, a config value, or a default that
  alters the recorded graph needs a new recording and a replan (`scripts/make_plans.sh`); `test_paper_plans.py`
  checks that the shipped plans still regenerate.
- Library code logs through `logging.getLogger(__name__)`; console scripts configure logging themselves.
