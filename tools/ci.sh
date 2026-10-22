#!/bin/bash
# Repo gate: everything that runs without the GPU, then one short GPU smoke (tools/gpu.sh; CI_GPU=0 skips it).
# Thread budgets and the GPU guard come from the caller's environment (on the shared box: `source tools/agent.sh`).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-8} RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-8} OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PC_NVCC_JOBS=${PC_NVCC_JOBS:-2} CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.2}
export PATH="$HOME/.cargo/bin:$CUDA_HOME/bin:$PATH" CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$PWD/target}"
# Device memory comes from ONE arena: no cudaMalloc* call or declaration outside the two mem.rs files
bad=$(grep -rnE --include='*.rs' --include='*.cu' --include='*.cuh' --include='*.h' 'cudaMalloc[A-Za-z]*[[:space:]]*\(' crates \
  | grep -vE '^crates/(pc-cuda/src/mem\.rs|pc-gpu/src/mem\.rs)' || true)
if [ -n "$bad" ]; then echo "ci: cudaMalloc outside mem.rs:"; echo "$bad"; exit 1; fi
# Comments describe the code, not the process that made it; module headers stay short.
python3 tools/comment_gate.py -v
cargo build --workspace --release
cargo test --workspace --release --no-fail-fast
# the Python package on the freshly built extension, CPU tier (GPU tests are marked `gpu`)
PY=${PYTHON:-.venv/bin/python}
VIRTUAL_ENV="$(cd "$(dirname "$PY")/.." && pwd)" "$(dirname "$PY")/maturin" develop --release
# four GPT-2 primitive tests expect the composite (thor) GELU; gpt2_base_n32 ships one Chebyshev series per block
known=(test_config_name_map test_gelu_fake_matches_ref test_gelu_fold_matches_gelu test_chebyshev_bsgs_depth_matches_direct)
CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=8 "$PY" -m pytest -q -m 'not gpu' -p no:cacheprovider \
  "${known[@]/#/--deselect=python/tests/test_gpt2_primitives_cpu.py::}"
if [ "${CI_GPU:-1}" = 1 ]; then
  tools/gpu.sh out/ci.selftest.log "$CARGO_TARGET_DIR/release/pc-kernels-selftest" --no-bench
  grep -q "exit=0" out/ci.selftest.log
  tools/gpu.sh out/ci.pc-gpu-smoke.log "$CARGO_TARGET_DIR/release/pc-gpu-smoke"
  grep -q "exit=0" out/ci.pc-gpu-smoke.log
fi
echo "ci: OK"
