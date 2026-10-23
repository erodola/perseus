"""The wire between the harness client and an entry's server: framed files in two one-way directories.

The client writes only the inbox and reads only the outbox; the server does the opposite. A file is
an 8-byte little-endian length followed by that many bytes, written under a temporary name and
renamed into place, so a reader never sees half a file.

The client treats the outbox as hostile. `read` resolves every path component with O_NOFOLLOW from a
directory descriptor, accepts only a regular file, and refuses anything larger than its cap. A server
that plants a symlink, a FIFO or an oversized file gets an error, not the client's bytes.
"""
from __future__ import annotations

import os
import stat
import struct
import time

HEADER = struct.Struct("<Q")


class WireError(Exception):
    """The other side broke the protocol: a bad path, a bad frame, or a timeout."""


def _parts(rel: str) -> list[str]:
    parts = rel.split("/")
    if rel.startswith("/") or any(p in ("", ".", "..") for p in parts):
        raise WireError(f"bad wire path {rel!r}")
    return parts


def write(root: str, rel: str, payload: bytes) -> None:
    """Write one framed file at root/rel, creating its directories."""
    parts = _parts(rel)
    directory = os.path.join(root, *parts[:-1])
    os.makedirs(directory, exist_ok=True)
    final = os.path.join(directory, parts[-1])
    tmp = os.path.join(directory, f".{parts[-1]}.tmp")
    with open(tmp, "wb") as f:
        f.write(HEADER.pack(len(payload)))
        f.write(payload)
    os.rename(tmp, final)


def _open_untrusted(root: str, rel: str) -> int:
    """A descriptor for root/rel, refusing a symlink at every component."""
    parts = _parts(rel)
    dir_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open(root, dir_flags)
    try:
        for name in parts[:-1]:
            nxt = os.open(name, dir_flags, dir_fd=fd)
            os.close(fd)
            fd = nxt
        return os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
    finally:
        os.close(fd)


def read(root: str, rel: str, cap: int) -> bytes:
    """Read one framed file written by the other side; at most `cap` payload bytes."""
    try:
        fd = _open_untrusted(root, rel)
    except OSError as e:
        raise WireError(f"{rel}: {e.strerror}") from None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise WireError(f"{rel}: not a regular file")
        if st.st_size > HEADER.size + cap:
            raise WireError(f"{rel}: {st.st_size} bytes, over the cap of {cap}")
        with os.fdopen(fd, "rb", closefd=False) as f:
            data = f.read(HEADER.size + cap + 1)
    finally:
        os.close(fd)
    if len(data) < HEADER.size:
        raise WireError(f"{rel}: no frame header")
    (n,) = HEADER.unpack_from(data)
    if n != len(data) - HEADER.size:
        raise WireError(f"{rel}: frame says {n} bytes, file holds {len(data) - HEADER.size}")
    return data[HEADER.size:]


def exists(root: str, rel: str) -> bool:
    """Whether root/rel is present, without following any symlink."""
    try:
        os.close(_open_untrusted(root, rel))
        return True
    except OSError:
        return False


def wait(root: str, rel: str, deadline: float, poll: float = 0.0005) -> float:
    """Block until root/rel appears; return the monotonic time it was seen. `deadline` is monotonic."""
    while True:
        if exists(root, rel):
            return time.monotonic()
        if time.monotonic() > deadline:
            raise WireError(f"{rel}: not written before the deadline")
        time.sleep(poll)
