"""Plan/config contract: stamp the capture-time configuration into plan artifacts, validate at load.

A plan loaded under a different configuration than its capture would die deep in the run with a bare
[plan_weight_error]/[plan_level_error]; the contract makes it a load-time message naming the knob. The stamp
(`placement.json["config"]`) is `Config.load_bearing()` of the capture (`capture_config.json`, else the legacy
`capture_env.json`); legacy-stamped plans carry `placement.json["capture_env"]` and both forms validate. Compared:
the stamp's set fields plus `LOAD_BEARING_ENV` (unset counts); scheduling-only knobs are not load-bearing.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

CONTRACT_KEY = "config"
#: the pre-Config stamp (still read: every shipped plan carries it)
LEGACY_KEY = "capture_env"

# Env vars that shape the captured op sequence or its level schedule (the legacy contract).
LOAD_BEARING_ENV = (
    "CHAIN", "LOGN", "AUTO_BTS_LEVEL", "BTS_ITERATIONS",
    "CKKS_COMPLEX", "GPT2_PACKING", "GPT2_CACHE",
    "GPT2_FOLD_LN1", "GPT2_FOLD_LN2", "GPT2_FOLD_LNF",
    "CACHE_READ_LEVEL_K", "CACHE_READ_LEVEL_V", "FHE_LMHEAD_CAP",
    # routing and fold decisions: they change which bootstraps exist and at which degree
    "SPARSE_BTS_SLOTS", "SPARSE_AUTO", "FUSED_SM_DEN", "FUSED_LN_VAR", "CORRECTION_FACTOR",
    "FHE_PT_COEFF_ENCODE",
    # the softmax division form (impl attention.softmax_thor): a different op sequence
    "SM_DEN_RECIP",
    # the GELU form (impl activation.gelu, examples weights up_scale): a different op sequence
    "GELU_FOLD",
    # the LayerNorm inverse sqrt seed (impl norm.norm) and the softmax's first division count
    "LN_CHEB", "SM_GS_FIRST",
    # the score layout (impl attention.sm_periodic): a different op sequence
    "SM_PERIODIC",
    # the softmax constants folded into masks (impl attention.sm_fold_affine): a different op sequence
    "SM_FOLD",
    # the K/V push form (impl attention.cache_kv_push_packed_complex): a different op sequence
    "KV_LANES",
    # step-scoped ciphertext names (impl rt.Rt): different variable names, the plan's keys
    "IMPL_SCOPED_NAMES",
    # the softmax head sums on one complex ladder (impl attention.head_reduce_sum) and the CutMax cascade's paired
    # 1-slot refreshes (examples head.inv_sigma_cascade): different op sequences
    "SM_HRS_PACK", "CUTMAX_PAIR",
    # graph ops nobody reads skipped (impl poly.eval_chebyshev_bsgs / inv_sqrt_newton_d2): a different op sequence
    "IMPL_DEAD_OPS",
    # rotate-and-sum ladders in hoisted radix groups (src/bindings/ops.cu rotate_and_sum): a different op sequence
    "LADDER_RADIX_BITS",
    # Chebyshev leaves on pre-aligned babies (impl poly._AlignedBabies): a different op sequence
    "CHEB_ALIGN",
    # Chebyshev-scaled Goldschmidt softmax rounds (impl smcheb): different counts and constants
    "SM_CHEB_GS",
    # mha_gqa decode: a different op sequence
    "LLAMA_GQA",
)


class PlanContractError(RuntimeError):
    """A plan's capture-time contract does not match the current session."""


def _pc():
    from perseus import _pc as pc
    return pc


def _field_of_env() -> dict[str, str]:
    """Legacy env name -> Config field name."""
    return {f["env"]: f["name"] for f in _pc().Config.fields()}


def capture_contract(inf=None):
    """The current session's legacy contract dict (written to capture_env.json at capture time).

    Unset vars are stamped as null: set<->unset transitions change the op
    sequence exactly like value changes do (an unset fold flag IS a value)."""
    c = {"env": {k: os.environ.get(k) for k in LOAD_BEARING_ENV}}
    if inf is not None:
        c["session"] = {"level_limit": inf.fhe.level_limit(),
                        "slots": inf.slots, "logN": inf.logN}
    return c


def _is_legacy(stamped) -> bool:
    return isinstance(stamped, dict) and isinstance(stamped.get("env"), dict)


def config_of(stamped):
    """The `_pc.Config` a stamp (Config JSON object, or a legacy `{"env": ...}` contract) states."""
    pc = _pc()
    if stamped is None or isinstance(stamped, pc.Config):
        return stamped
    if _is_legacy(stamped):
        return pc.Config.from_legacy_env(stamped["env"])
    return pc.Config.from_json(json.dumps(stamped))


def stamp_of(config_or_legacy) -> dict:
    """The plan stamp (`placement.json["config"]`): `Config.load_bearing()` as a JSON object."""
    return json.loads(config_of(config_or_legacy).load_bearing().to_json())


def capture_stamp(graph_block_dir) -> dict | None:
    """The stamp of a capture: `capture_config.json` (fakecap v2), else the legacy `capture_env.json`, in the block
    dir or its parent (the graph dir); None when the capture recorded neither."""
    d = Path(graph_block_dir)
    for name in ("capture_config.json", "capture_env.json"):
        for base in (d, d.parent):
            f = base / name
            if f.is_file():
                doc = f.read_text(encoding="utf-8")
                return stamp_of(_pc().Config.from_json(doc) if name == "capture_config.json" else json.loads(doc))
    return None


def session_config():
    """The session's Config as the running code still reads it: the legacy env (until the runtime takes a Config)."""
    names = set(_field_of_env())
    return _pc().Config.from_legacy_env({k: v for k, v in os.environ.items() if k in names})


def _set_fields(doc: dict, prefix: str = ""):
    for k, v in doc.items():
        if isinstance(v, dict):
            yield from _set_fields(v, f"{prefix}{k}.")
        elif v is not None:
            yield f"{prefix}{k}"


def contract_mismatches(stamped, inf=None, session=None):
    """List of human-readable mismatches between a stamp (either form) and the session (`session`: a Config,
    default `session_config()`)."""
    want = config_of(stamped)
    if want is None:
        return []
    have = session if session is not None else session_config()
    f_of = _field_of_env()
    env_of = {v: k for k, v in f_of.items()}
    legacy = _is_legacy(stamped)
    if legacy:
        names = [f_of[k] for k in stamped["env"] if k in f_of]
    else:
        names = sorted(set(_set_fields(json.loads(want.to_json())))
                       | {f_of[k] for k in LOAD_BEARING_ENV if k in f_of})
    out = []
    for name in names:
        a, b = want.get(name), have.get(name)
        if a != b:
            out.append(f"{name} ({env_of.get(name, name)}): captured {a!r}, session has {b!r}")
    if inf is not None:
        cur = {"level_limit": inf.fhe.level_limit(), "slots": inf.slots, "logN": inf.logN}
        sess = (stamped.get("session") or {}) if legacy else {}
        for key, v in cur.items():
            if key in sess and sess[key] != v:
                out.append(f"{key}: captured {sess[key]}, session has {v}")
        if not legacy:
            for key, name in (("level_limit", "bts.auto_bts_level"), ("logN", "chain.logn")):
                a = want.get(name)
                if a is not None and a != cur[key]:
                    out.append(f"{key} ({name}): captured {a}, session has {cur[key]}")
    return out


def validate_contract(stamped, inf=None, source="plan", strict=True, session=None):
    """Raise (strict) or return the mismatch list. None/absent contract passes."""
    if not stamped:
        return []
    bad = contract_mismatches(stamped, inf, session)
    if bad and strict:
        raise PlanContractError(
            f"{source}: capture-time contract mismatch — the plan is op-sequence-"
            f"strict and will fail deep in the run. Fix the configuration or re-plan:\n  "
            + "\n  ".join(bad))
    return bad


def read_stamp(path):
    """The contract stamped in a placement/plan JSON file (the Config stamp, else the legacy one), or None."""
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return None
    return doc.get(CONTRACT_KEY) or doc.get(LEGACY_KEY)


def stamp_file(path, contract):
    """Stamp an existing plan/placement JSON in place with the Config stamp of `contract` (a Config, a stamp, or a
    legacy `{"env": ...}` contract); a legacy stamp already in the file is replaced."""
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    doc.pop(LEGACY_KEY, None)
    doc[CONTRACT_KEY] = stamp_of(contract)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
