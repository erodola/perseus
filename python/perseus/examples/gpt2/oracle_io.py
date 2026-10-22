"""The decode oracle's files, read in Python: the old ``_core.RunConfig.from_env`` /
``read_teacher_forced_inputs`` / ``read_lm_head_steps`` (C++ app pipeline), so the fake runs need no
extension. Same env names, defaults and file fallbacks as the C++ readers.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


def _env_or(k: str, d: str) -> str:
    v = os.environ.get(k)
    return v if v else d


@dataclass
class RunConfig:
    tokens: int = 1
    steps_t: int = 16
    io_dir: str = ""

    @staticmethod
    def from_env() -> RunConfig:
        return RunConfig(tokens=int(_env_or("MULTI_T", "1")), steps_t=int(_env_or("STEPS_T", "16")),
                         io_dir=_env_or("ALL_BLOCKS_IO_DIR", ""))


@dataclass
class GtSteps:
    T: int = 0
    logits: list[list[float]] = field(default_factory=list)


def read_teacher_forced_inputs(cfg: RunConfig) -> list[list[float]]:
    """The first max(1, tokens) rows of `inp` in all_blocks_L00_T<horizon>.json (horizon = max(steps_t, rows)),
    falling back to the T<rows> file."""
    rows = max(1, cfg.tokens)
    if not cfg.io_dir:
        raise RuntimeError("ALL_BLOCKS_IO_DIR is not set (no default)")
    horizon = max(cfg.steps_t, rows)

    def path_for(t: int) -> str:
        return os.path.join(cfg.io_dir, f"all_blocks_L00_T{t}.json")

    path = path_for(horizon)
    if not os.path.exists(path) and horizon != rows:
        path = path_for(rows)
    with open(path) as f:
        inp = json.load(f)["inp"]
    if len(inp) < rows:
        raise RuntimeError(f"block-0 io has < requested rows: {path}")
    return [list(map(float, r)) for r in inp[:rows]]


def read_lm_head_steps(cfg: RunConfig) -> GtSteps:
    """The per-step logits of all_blocks_lm_head_steps_T<steps_t>.json; empty when the file is absent."""
    path = os.path.join(cfg.io_dir, f"all_blocks_lm_head_steps_T{cfg.steps_t}.json")
    try:
        with open(path) as f:
            steps = json.load(f).get("steps", [])
    except OSError:
        return GtSteps()
    logits = [s["logits"] for s in steps]
    return GtSteps(T=len(logits), logits=logits)
