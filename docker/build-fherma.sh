#!/usr/bin/env bash
#
# Build perseus-core into one prefix, the layout of the FHERMA image:
#   <prefix>/Cargo.toml, crates/   the Rust workspace, for an answer to path-depend on (crates/pc-gpu, ...)
#   $CARGO_HOME                    the registry warmed with every crate of Cargo.lock, so that answer builds offline
#   <prefix>/venv                  Python 3.12 with perseus (and its extension perseus._pc), numpy and networkx
#   <prefix>/bin                   pc-kernels-selftest and pc-gpu-smoke, the two GPU checks of the runtime
# This is what Dockerfile.fherma runs; it also runs on a host, which is how the image is checked without Docker.
#
#   build-fherma.sh <install-prefix>
#
# Env: PERSEUS_CORE_ARCH (default "89;120": FHERMA's L40S and RTX PRO 6000 Blackwell), CUDA_HOME (default
#      /usr/local/cuda), CARGO_HOME (the registry to warm; cargo's default otherwise), PYTHON (default python3),
#      PC_NVCC_JOBS (default 4), WORK_DIR (scratch, default a fresh temporary directory, removed on success),
#      PERSEUS_REVISION (recorded in BUILD-INFO; defaults to this checkout's HEAD).

set -euo pipefail

[ $# -eq 1 ] || { echo "usage: $0 <install-prefix>" >&2; exit 2; }
PREFIX="$(realpath -m "$1")"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export PERSEUS_CORE_ARCH="${PERSEUS_CORE_ARCH:-89;120}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PC_NVCC_JOBS="${PC_NVCC_JOBS:-4}"
PYTHON="${PYTHON:-python3}"
WORK="${WORK_DIR:-$(mktemp -d)}"
REVISION="${PERSEUS_REVISION:-$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)}"
MATURIN=1.15.0
PY_PINS=(numpy==2.5.2 networkx==3.6.1)

echo "=== perseus-core -> $PREFIX, arch $PERSEUS_CORE_ARCH, CUDA $CUDA_HOME ==="
mkdir -p "$PREFIX" "$WORK"

# The cargo on PATH must be the toolchain rust-toolchain.toml names: the image has no network to fetch another.
WANT="$(sed -n 's/^channel *= *"\(.*\)"/\1/p' "$REPO/rust-toolchain.toml")"
HAVE="$(cd "$REPO" && cargo --version | awk '{print $2}')"
[ "$WANT" = "$HAVE" ] || { echo "ERROR: cargo $HAVE, rust-toolchain.toml wants $WANT" >&2; exit 1; }

# ---- the workspace, and the registry an offline build of it needs
cp "$REPO"/{Cargo.toml,Cargo.lock,rust-toolchain.toml,LICENSE} "$PREFIX"/
rm -rf "$PREFIX/crates"
cp -r "$REPO/crates" "$PREFIX/crates"
find "$PREFIX/crates" -name target -prune -exec rm -rf {} +
cargo fetch --locked --manifest-path "$PREFIX/Cargo.toml"

# ---- the GPU checks, built from the installed workspace with no network: the proof that it builds offline
CARGO_TARGET_DIR="$WORK/target" cargo build --release --locked --offline --manifest-path "$PREFIX/Cargo.toml" \
    -p pc-kernels --bin pc-kernels-selftest -p pc-gpu --bin pc-gpu-smoke
install -Dm755 "$WORK/target/release/pc-kernels-selftest" "$PREFIX/bin/pc-kernels-selftest"
install -Dm755 "$WORK/target/release/pc-gpu-smoke" "$PREFIX/bin/pc-gpu-smoke"

# ---- the Python package; maturin only builds the wheel, and stays out of the prefix
"$PYTHON" -m venv "$WORK/build-venv"
"$WORK/build-venv/bin/pip" install -q --no-cache-dir "maturin==$MATURIN"
(cd "$REPO" && CARGO_TARGET_DIR="$WORK/target" "$WORK/build-venv/bin/maturin" build --release --locked --offline \
    --auditwheel skip --interpreter "$PYTHON" -o "$WORK/dist")
rm -rf "$PREFIX/venv"
"$PYTHON" -m venv "$PREFIX/venv"
"$PREFIX/venv/bin/pip" install -q --no-cache-dir "$WORK"/dist/perseus-*.whl "${PY_PINS[@]}"
# Import only: loading the extension needs cudart and cuBLASLt, not a GPU.
"$PREFIX/venv/bin/python" -c "import perseus, perseus._pc; print('perseus', perseus.__file__)"

# ---- licence and provenance
install -Dm644 "$REPO/LICENSE" "$PREFIX/share/licenses/perseus/LICENSE"
cat > "$PREFIX/BUILD-INFO" <<EOF
perseus   $REVISION
rust      $HAVE
cuda      $("$CUDA_HOME/bin/nvcc" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')
arch      $PERSEUS_CORE_ARCH
python    $("$PREFIX/venv/bin/python" -c 'import sys; print(sys.version.split()[0])')
wheel     $(basename "$WORK"/dist/perseus-*.whl)
EOF

[ -n "${WORK_DIR:-}" ] || rm -rf "$WORK"
echo "=== Installed into $PREFIX ==="
