#!/bin/bash
# Time an entry against the current record as the maintainers do, on one GPU: pairs of runs in ORDER (default AB BA BA
# AB AB BA; A the record, B the entry), pair p on prompts OFFSET + p N .. OFFSET + (p + 1) N - 1 of the hidden test set
# for both arms, then harness/score.py --official. RECORD and ENTRY are builds (runner/build.sh) of the current record
# and of the commit the PR check names; CLIENT, of the commit the record's client (BACKEND, default perseus-core) pins
# (default RECORD), with GPT-2's weights (build.sh EXPORT=1), which both arms read. An entry with its own client passes
# the client file its PR adds, once reviewed, as CLIENT_FILE: it joins this harness for the run, with ENTRY as its
# client. Writes the runs, score.txt, score.json and report.md (the comment for the PR) to OUT.
#   PROMPTS=<hidden test set> [CLIENT=DIR] [CLIENT_FILE=FILE] [BACKEND=NAME] [N=8] [OFFSET=0] [ORDER="AB BA ..."]
#   [NUMA=NODE[:MEMS]] [OUT=DIR] bash speedrun/runner/official.sh {prefill,decode} GPU RECORD ENTRY
set -euo pipefail
T=$1 G=$2 A=$(cd "$3" && pwd) B=$(cd "$4" && pwd)
H=$(cd "$(dirname "$0")" && pwd)
: "${PROMPTS:?PROMPTS must name the hidden test set}"
N=${N:-8} OFFSET=${OFFSET:-0} ORDER=${ORDER:-AB BA BA AB AB BA} BACKEND=${BACKEND:-perseus-core}
C=$(cd "${CLIENT:-$A}" && pwd)
OUT=${OUT:-$H/../runs/official-$T-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"
BACKEND_B=$BACKEND CLIENT_B=$C
if [ -n "${CLIENT_FILE:-}" ]; then
    name=$(basename "$CLIENT_FILE")
    [[ "$name" =~ ^[a-z0-9_]+_client\.py$ ]] || { echo "official.sh: CLIENT_FILE must be <name>_client.py" >&2; exit 2; }
    dst="$H/../harness/$name"
    [ ! -e "$dst" ] || { echo "official.sh: $dst already exists" >&2; exit 2; }
    cp "$CLIENT_FILE" "$dst"
    trap 'rm -f "$dst"' EXIT
    BACKEND_B=${name%_client.py} CLIENT_B=$B
fi
W="$C/.cache/models/openai-community/gpt2/classic/weights.bin.zip"
p=0
for pair in $ORDER; do
    for arm in $(fold -w1 <<< "$pair"); do
        if [ "$arm" = A ]; then dir=$A cl=$C be=$BACKEND; else dir=$B cl=$CLIENT_B be=$BACKEND_B; fi
        rc=0
        RUNS="$OUT/$arm" CLIENT="$cl" WEIGHTS="$W" bash "$H/entry.sh" "$be" "$dir" "$T" "$G" \
            --requests "$N" --offset $((OFFSET + p * N)) || rc=$?
        [ "$rc" -ne 2 ] || { echo "official.sh: run.py refused the $arm arm (above)" >&2; exit 2; }
    done
    p=$((p + 1))
done
runs() { find "$OUT/$1" -name run.json | sort; }
python3 -I "$H/../harness/score.py" --official --track "$T" --entry $(runs B) --record $(runs A) \
    --json "$OUT/score.json" > "$OUT/score.txt" 2>&1 || true
cat "$OUT/score.txt"
{
    echo "<!-- speedrun-official track=$T -->"
    echo "**Official timing, $([ "$T" = decode ] && echo Decode-8 || echo Prefill-128).** The entry at" \
         "\`$(git -C "$B" rev-parse HEAD)\` against the record at \`$(git -C "$A" rev-parse HEAD)\`, on" \
         "$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$G"): $(wc -w <<< "$ORDER") pairs ($ORDER), each on" \
         "$N prompts of the hidden test set; harness \`$(git -C "$H" rev-parse HEAD)\`."
    echo
    echo '```'
    cat "$OUT/score.txt"
    echo '```'
} > "$OUT/report.md"
echo "official.sh: $OUT/report.md"
