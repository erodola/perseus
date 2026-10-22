"""Where the package's data lives.

Configs, graphs and plans are addressed by repo-relative spellings
(``configs/model/approximation/...``, ``graphs/...``, ``bootstrap_placements/...``): the approximation
configs ship inside the package, the graphs and plans at the repository root (``ASSETS``). ``resolve``
keeps those spellings working from any working directory: an existing path wins, then the package
root, then the repository root.
"""
from __future__ import annotations

import os
from pathlib import Path

PKG = Path(__file__).resolve().parent
# python/perseus -> the repository root (graphs/, bootstrap_placements/)
ASSETS = PKG.parents[1]
CONFIGS = PKG / "configs" / "model" / "approximation"


def resolve(path: str | os.PathLike | None) -> str | None:
    """`path` as given when it exists (or is absolute / None); else the first of PKG/path,
    ASSETS/path that exists; else `path` unchanged (the caller's open names it)."""
    if path is None:
        return None
    p = Path(path)
    if p.is_absolute() or p.exists():
        return str(path)
    for root in (PKG, ASSETS):
        if (root / p).exists():
            return str(root / p)
    return str(path)
