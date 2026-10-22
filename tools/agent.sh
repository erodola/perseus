#!/bin/bash
# This machine's opt-in (a shared box): source it in every shell, `source tools/agent.sh`.
# Budget per agent: 8 threads for everything (build + tests + host side of a GPU job).
# Box rule: <= 36 threads total across all agents; at most 2 agents run at once (user rule 2026-10-19: the 5-hour session usage cap is shared by every agent).
export CARGO_BUILD_JOBS=8
export RAYON_NUM_THREADS=8
export OMP_NUM_THREADS=8
export PC_NVCC_JOBS=2
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.2}
export PATH="$HOME/.cargo/bin:$CUDA_HOME/bin:$PATH"
# per-worktree target dir so concurrent agents never contend on one cargo lock
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)/target}"
# GPU guard: only GPU 3, only through tools/gpu.sh, under the lock shared with the other perseus checkouts
export PERSEUS_GPU_GUARD=3
export PERSEUS_GPU_LOCK=/data02/users/mithrillo/projects/perseus/logs/prefill/.gpu.lock
