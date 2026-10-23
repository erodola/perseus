#!/bin/bash
# Run one command on untrusted entry code: an allowlisted filesystem (no /home, /data02 or /run), one GPU device
# node, no network, no new user namespaces, an empty environment, and memory / task / wall-time limits enforced
# from outside by a systemd user service (named, so the runner can query it and sample its memory: the peak systemd
# prints on exit is stale unless the unit was queried while alive), and a seccomp filter (runner/seccomp.py: no
# io_uring, keyrings, userfaultfd, bpf or perf events, x86-64 calls only). Only --ro paths are readable; only --out is
# writable; /work is a RAM tmpfs (charged to the scope's memory limit) that disappears with the sandbox. --gpu none gives no device
# node at all (the client); --rw binds one more writable path. The client/server wire is two one-way directories
# under /dev/shm: the inbox is --rw for the client and --ro for the server, the outbox the reverse. /work and /tmp
# are size-capped tmpfs (--work, default 64g; /tmp 8g), on top of the scope's memory cap. --setenv adds one variable
# to the empty environment.
#   runner/sandbox.sh --gpu {N|none} --out OUTDIR [--ro PATH]... [--rw PATH]... [--setenv NAME VALUE]...
#                     [--numa NODE[:MEMS]] [--cpus N] [--unit NAME] [--mem 100G] [--time 3600] [--work 64g] -- CMD ARGS...
# --numa runs CMD under numactl --cpunodebind=NODE --membind=MEMS (default NODE; advisory until cpuset is
# delegated: code inside can change its own affinity); --cpus N caps its CPU time at N CPUs (CPUQuota), which holds.
set -euo pipefail
GPU= OUT= MEM=100G TIME=3600 TASKS=4096 WORK=64g TMP=8g NUMA= CPUS= UNIT="sbx-$$" RO=() ENV=()
while [ $# -gt 0 ]; do
    case "$1" in
        --gpu) GPU=$2; shift 2 ;;
        --out) OUT=$2; shift 2 ;;
        --ro) RO+=(--ro-bind "$2" "$2"); shift 2 ;;
        --rw) RO+=(--bind "$2" "$2"); shift 2 ;;
        --setenv) ENV+=(--setenv "$2" "$3"); shift 3 ;;
        --mem) MEM=$2; shift 2 ;;
        --time) TIME=$2; shift 2 ;;
        --work) WORK=$2; shift 2 ;;
        --numa) NUMA=$2; shift 2 ;;
        --cpus) CPUS=$2; shift 2 ;;
        --unit) UNIT=$2; shift 2 ;;
        --) shift; break ;;
        *) echo "sandbox.sh: unknown option $1" >&2; exit 2 ;;
    esac
done
[ -n "$GPU" ] && [ -n "$OUT" ] && [ $# -gt 0 ] || { echo "usage: sandbox.sh --gpu N --out DIR [--ro PATH]... -- CMD" >&2; exit 2; }
mkdir -p "$OUT"
PIN=()
[ -n "$NUMA" ] && PIN=(numactl --cpunodebind="${NUMA%%:*}" --membind="${NUMA#*:}")
DEV=()
if [ "$GPU" != none ]; then
    DEV=(--dev-bind "/dev/nvidia$GPU" "/dev/nvidia$GPU" --dev-bind /dev/nvidiactl /dev/nvidiactl
         --dev-bind /dev/nvidia-uvm /dev/nvidia-uvm)
fi
QUOTA=()
[ -n "$CPUS" ] && QUOTA=(-p CPUQuota="$((CPUS * 100))%")
FILTER=$(mktemp /dev/shm/sbx-seccomp.XXXXXX)        # opened by the service, removed when it ends
trap 'rm -f "$FILTER"' EXIT
python3 -I "$(dirname "$0")/seccomp.py" > "$FILTER"
systemd-run --user --wait --pipe --collect --unit "$UNIT" -p MemoryMax="$MEM" -p MemorySwapMax=0 \
    -p TasksMax="$TASKS" -p OOMPolicy=kill -p RuntimeMaxSec="$TIME" "${QUOTA[@]}" -- \
    /bin/sh -c 'exec bwrap --seccomp 3 "$@" 3< "$0"' "$FILTER" --unshare-all --unshare-user --disable-userns --cap-drop ALL --die-with-parent --new-session --hostname sandbox \
        --clearenv --setenv PATH /usr/local/cuda/bin:/usr/bin --setenv HOME /work --setenv LANG C.UTF-8 "${ENV[@]}" \
        --ro-bind /usr /usr --symlink usr/bin /bin --symlink usr/lib /lib --symlink usr/lib64 /lib64 \
        --symlink usr/sbin /sbin --ro-bind /etc /etc --ro-bind /sys /sys --tmpfs /sys/fs/cgroup \
        --proc /proc --dev /dev \
        "${DEV[@]}" \
        --dir /run --size "$(numfmt --from=iec "${TMP^^}")" --tmpfs /tmp \
        --size "$(numfmt --from=iec "${WORK^^}")" --tmpfs /work --chdir /work \
        "${RO[@]}" --bind "$OUT" /out \
        "${PIN[@]}" "$@"
