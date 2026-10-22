#!/bin/bash
# usage: tools/gpu.sh <log> <cmd...>
# Runs one GPU job with a timeout (GPU_JOB_TIMEOUT, default 900 s); output and `exit=<rc>` go to <log>.
# Default: the job sees the caller's CUDA_VISIBLE_DEVICES and the runtime takes the first visible device.
# Guarded mode for shared machines, opt-in with PERSEUS_GPU_GUARD=<gpu> (tools/agent.sh sets it on ours): the job
# runs on that GPU only, under a machine-wide flock (PERSEUS_GPU_LOCK), once the GPU is idle, with
# PERSEUS_CORE_LOCKED=1 (while the guard is set the runtime refuses a device without it), and a watchdog kills the
# job if a process of this user shows up on another GPU (child code can rewrite CUDA_VISIBLE_DEVICES).
set -u
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-8}
log=$1; shift
mkdir -p "$(dirname "$log")"
gpu=${PERSEUS_GPU_GUARD:-}
if [ -z "$gpu" ]; then
  date -Is > "$log"
  timeout "${GPU_JOB_TIMEOUT:-900}" "$@" >> "$log" 2>&1
  rc=$?
  echo "exit=$rc" >> "$log"
  exit $rc
fi
export CUDA_VISIBLE_DEVICES=$gpu PERSEUS_CORE_LOCKED=1
exec 9>"${PERSEUS_GPU_LOCK:-/tmp/perseus-gpu$gpu.lock}"
flock 9
until [ -z "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 10; done
{ date -Is; nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader; } > "$log"
mine=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
me=$(id -u)
setsid timeout "${GPU_JOB_TIMEOUT:-900}" "$@" >> "$log" 2>&1 &
job=$!
# watchdog: our processes on any other GPU -> kill the whole job's process group and fail
while kill -0 "$job" 2>/dev/null; do
  bad=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null | while IFS=', ' read -r uuid pid; do
          [ "$uuid" != "$mine" ] && [ "$(stat -c %u /proc/"$pid" 2>/dev/null)" = "$me" ] && echo "$pid@$uuid"; done)
  if [ -n "$bad" ]; then
    echo "[gpu.sh] VIOLATION: process(es) of this user on another GPU: $bad -- killing the job" | tee -a "$log" >&2
    kill -TERM -- -"$job" 2>/dev/null; sleep 3; kill -KILL -- -"$job" 2>/dev/null
    for pb in $bad; do kill -KILL "${pb%@*}" 2>/dev/null; done
    echo "exit=97" >> "$log"
    exit 97
  fi
  sleep 2
done
wait "$job"
rc=$?
echo "exit=$rc" >> "$log"
exit $rc
