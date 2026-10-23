"""The sandbox's seccomp filter as a classic BPF program on stdout, for `bwrap --seccomp FD` (runner/sandbox.sh).

    python3 -I runner/seccomp.py > FILTER

It kills any process that makes a system call through another architecture than x86-64 (i386, x32), and fails with
EPERM the calls that open kernel attack surface a sandboxed entry never needs: io_uring, the key retention service,
userfaultfd (the user-mode flag stays open to unprivileged users here), bpf and perf_event_open. Everything else is
allowed: the sandbox's namespaces and capabilities do the rest. Standard library only.
"""
from __future__ import annotations

import struct
import sys

AUDIT_ARCH_X86_64, X32_BIT = 0xC000003E, 0x40000000
LD_ABS, JEQ, JGE, RET = 0x20, 0x15, 0x35, 0x06          # BPF_LD|W|ABS, BPF_JMP|JEQ|K, BPF_JMP|JGE|K, BPF_RET|K
KILL, ERRNO, ALLOW, EPERM = 0x80000000, 0x00050000, 0x7FFF0000, 1
BLOCKED = {"add_key": 248, "request_key": 249, "keyctl": 250, "userfaultfd": 323, "bpf": 321, "perf_event_open": 298,
           "io_uring_setup": 425, "io_uring_enter": 426, "io_uring_register": 427}


def program() -> bytes:
    nrs = sorted(BLOCKED.values())
    n = len(nrs)
    ops = [(LD_ABS, 0, 0, 4),                           # seccomp_data.arch
           (JEQ, 1, 0, AUDIT_ARCH_X86_64),
           (RET, 0, 0, KILL),
           (LD_ABS, 0, 0, 0),                           # seccomp_data.nr
           (JGE, n + 2, 0, X32_BIT)]                    # an x32 call: to the KILL at the end
    ops += [(JEQ, n - j, 0, nr) for j, nr in enumerate(nrs)]   # a blocked call: to the ERRNO return
    ops += [(RET, 0, 0, ALLOW), (RET, 0, 0, ERRNO | EPERM), (RET, 0, 0, KILL)]
    return b"".join(struct.pack("<HBBI", *op) for op in ops)


if __name__ == "__main__":
    sys.stdout.buffer.write(program())
