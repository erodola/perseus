#!/bin/bash
# The Decode-8 plan's capture (CPU only). The shipped decode capture (gpt2_decode_python_n32_gcheb) records one row of
# the OpenWebText calibration pool at position 0, and the plan serves positions 0-15, where the values that read the
# cache grow with the number of keys (block 2's q.K^T sum: a coefficient max 0.045 at position 0, 13.4 at position 15),
# and other rows reach other maxima. Here each node's output_max_* is the largest over the shipped capture and the numpy
# fake's tapes of ROWS rows of the pool (the decode oracle at those pool offsets, scripts/utils/gen_gpt2_oracle.py) at
# positions 0-15 (perseus.examples.gpt2.capture_mags under the record plan's runtime; at position 1 the K / V push takes
# another op sequence and keeps the capture's statistics, every other step is filled), merged by name
# (scripts/utils/merge_mags.py).
#   bash speedrun/make_decode_capture.sh [OUT]      -> graphs/gpt2_decode_python_n32_gcheb16
# POOL: the pool .npy, by default the one perseus.calibrate.data builds (<HF_HOME>/perseus/pools/
# openwebtext_gpt2_2000000.npy); its rows at the ROWS offsets are the owt_pool windows of
# graphs/gpt2_decode_python_n32_cm/cutmax_rows.json (the first 16 tokens each), against which a pool can be checked.
# Another configuration's capture: BASE its position-0 capture, PLAN a plan dir with its runtime= and configs= lines
#   BASE=graphs/gpt2_decode_python_n32_kc16 PLAN=bootstrap_placements/gpt2_decode_python_n32_kc16 \
#       bash speedrun/make_decode_capture.sh graphs/gpt2_decode_python_n32_kc16m
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/local_env.sh
export CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=1
BASE=${BASE:-graphs/gpt2_decode_python_n32_gcheb} OUT=${1:-graphs/gpt2_decode_python_n32_gcheb16}
PLAN=${PLAN:-bootstrap_placements/gpt2_decode_python_n32_m16i_fcf}
POOL=${POOL:-${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}/perseus/pools/openwebtext_gpt2_2000000.npy}
ROWS=${ROWS:-4096 8192 12288 16384}
for r in $ROWS; do
    d="$PERSEUS_DATA/oracle/gpt2/owt_$r"
    [ -f "$d/all_blocks_L00_T128.json" ] || "$PYTHON" scripts/utils/gen_gpt2_oracle.py --model openai-community/gpt2 \
        --pool "$POOL" --pool-offset "$r" --T 16 128 --out "$d"
done
TMP="$(mktemp -d graphs/_positions_XXXX)"
trap 'rm -rf "$TMP"' EXIT
export BASE PLAN TMP
for r in $ROWS; do for t in $(seq 0 15); do echo "$r $t"; done; done | xargs -P "${JOBS:-16}" -L 1 bash -c '
    d="$TMP/r$0_p$1"; cp -r "$BASE" "$d"
    ALL_BLOCKS_IO_DIR="$PERSEUS_DATA/oracle/gpt2/owt_$0" "$PYTHON" -m perseus.examples.gpt2.capture_mags "$d" \
        --argmax --plan "$PLAN" --pos "$1" --keep-differing > "$d.log" 2>&1 || { tail -3 "$d.log"; exit 255; }'
grep -h "keeps the capture" "$TMP"/*.log | sed 's|^.*/r[0-9]*_p\([0-9]*\)/|position \1: |' | sort | uniq -c || true
rm -rf "$OUT"
"$PYTHON" scripts/utils/merge_mags.py "$BASE" "$OUT" "$TMP"/r*_p*/
