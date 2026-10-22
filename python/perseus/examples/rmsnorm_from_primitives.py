"""RMSNorm built from the session's leaf primitives: y = x * rsqrt(mean(x^2) + eps) * gamma.

square -> all-slot sum (``sum_slots``) -> Chebyshev rsqrt (``eval_chebyshev``) -> x * gamma (``mult_pt``) -> product:
4 + ceil(log2 degree) levels (`rmsnorm_depth`). Mirrored in numpy with the SAME Chebyshev approximation, so the
approximation error (`rmsnorm_ref` vs `rmsnorm_exact`) is separable from the FHE noise (`rmsnorm` vs `rmsnorm_ref`).
Precondition: a FRESH token-basis ciphertext (off-lane slots zero). Packing facts, the Chebyshev convention, the
level budget and a usage example: docs/ARCHITECTURE.md; tests/gpu/test_primitives.py is the executable form.
"""
from __future__ import annotations

import functools
import logging
import math
from typing import NamedTuple

import numpy as np
from numpy.polynomial import chebyshev as _cheb

log = logging.getLogger(__name__)

DEFAULT_EPS = 1e-5
DEFAULT_INTERVAL = (0.04, 0.16)     # mean(x^2)+eps for x ~ 0.3 N(0,1), d=768: [0.074, 0.109]
DEFAULT_DEGREE = 7                  # the deepest degree that fits custom_n32's 7 levels

_SLOT_LAYOUT_PACKINGS = ("cachemir", "cachemir_complex")   # lane i at slot i*t (packed_ctx.h)


def _check_interval(interval) -> tuple[float, float]:
    """Validate a fitting interval: finite, ``0 < a < b`` (rsqrt needs a positive argument)."""
    try:
        a, b = (float(v) for v in interval)
    except (TypeError, ValueError):
        raise ValueError(f"interval must be an (a, b) pair, got {interval!r}") from None
    if not (math.isfinite(a) and math.isfinite(b)) or not 0.0 < a < b:
        raise ValueError(f"interval must satisfy 0 < a < b with finite bounds, got {interval!r}")
    return a, b


@functools.cache
def _fit(a: float, b: float, degree: int) -> tuple[float, ...]:
    mid, half = (a + b) / 2.0, (b - a) / 2.0
    c = _cheb.chebinterpolate(lambda y: 1.0 / np.sqrt(mid + half * y), degree)
    return tuple(float(v) for v in c)


def rsqrt_cheb_coeffs(interval, degree: int) -> tuple[float, ...]:
    """Chebyshev coefficients ``c_0 .. c_degree`` of ``u ** -0.5`` on ``interval``.

    Fitted with `numpy.polynomial.chebyshev.chebinterpolate` (interpolation at the
    Chebyshev points of the first kind); ``c_0`` is unhalved -- `chebval` and the runtime's
    `eval_chebyshev` agree on that convention. Cached per (interval, degree).
    """
    a, b = _check_interval(interval)
    degree = int(degree)
    if degree < 1:
        raise ValueError(f"degree must be >= 1, got {degree}")
    return _fit(a, b, degree)


DEFAULT_COEFFS = rsqrt_cheb_coeffs(DEFAULT_INTERVAL, DEFAULT_DEGREE)


def fhe_interval(interval, eps: float, d: int) -> tuple[float, float]:
    """The evaluator interval that makes ``eval_chebyshev(S, coeffs, a', b')`` on the sum of
    squares ``S`` equal the series on ``u = S/d + eps`` over ``[a, b]``.

    With ``a' = (a - eps) d`` and ``b' = (b - eps) d``:
    ``(2S - (a' + b')) / (b' - a') = (2 (S/d + eps) - (a + b)) / (b - a)``, the affine map
    the runtime applies (src/primitives/polynomial.cu) -- so the ``1/d`` and ``+eps`` are
    absorbed into a map that is evaluated anyway, saving one level.
    """
    a, b = _check_interval(interval)
    return (a - eps) * d, (b - eps) * d


def rmsnorm_depth(degree: int) -> int:
    """CKKS levels the chain consumes: square 1, affine map 1, ``T_degree`` tree
    ``ceil(log2 degree)``, weighted sum 1, final ciphertext product 1 (the gamma product
    runs in parallel and is absorbed by the final product's level alignment)."""
    degree = int(degree)
    if degree < 1:
        raise ValueError(f"degree must be >= 1, got {degree}")
    return 4 + math.ceil(math.log2(degree))


def _level_unit(sess) -> int:
    """Primes per CKKS level (`composite_degree`; 1 when the session does not say)."""
    ckks = getattr(getattr(sess, "options", None), "ckks", None)
    return int(getattr(ckks, "composite_degree", 1) or 1)


def level_budget(sess, ct) -> int:
    """Whole CKKS levels ``ct`` can consume before the runtime's reactive bootstrap fires
    (which happens once ``ct.level >= fhe.level_limit()``; both prime-granular on n32)."""
    limit = int(sess.inf.fhe.level_limit())
    return (limit - int(ct.level) - 1) // _level_unit(sess)


def gamma_to_slots(gamma, slots: int, t: int) -> np.ndarray:
    """Lay ``gamma`` out as the token packing does: ``gamma[i]`` at slot ``i * t``, zeros
    elsewhere (the `mult_pt` operand)."""
    g = np.asarray(gamma, dtype=np.float64).ravel()
    if g.shape[0] * t > slots:
        raise ValueError(f"{g.shape[0]} lanes at stride {t} do not fit in {slots} slots")
    out = np.zeros(int(slots))
    out[np.arange(g.shape[0]) * int(t)] = g
    return out


def rmsnorm_exact(x, gamma, eps: float = DEFAULT_EPS) -> np.ndarray:
    """Closed-form ``x * rsqrt(mean(x^2) + eps) * gamma``."""
    x = np.asarray(x, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    return x / np.sqrt(np.mean(x * x) + eps) * gamma


def rmsnorm_ref(x, gamma, *, eps: float = DEFAULT_EPS, interval=DEFAULT_INTERVAL,
                degree: int = DEFAULT_DEGREE) -> np.ndarray:
    """numpy mirror of `rmsnorm`: the SAME Chebyshev series for the rsqrt, so
    ``rmsnorm_ref - rmsnorm_exact`` is the approximation error and ``fhe - rmsnorm_ref`` the
    FHE noise. Raises ValueError when ``mean(x^2) + eps`` leaves ``interval``."""
    x = np.asarray(x, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    a, b = _check_interval(interval)
    coeffs = rsqrt_cheb_coeffs((a, b), degree)
    u = float(np.mean(x * x) + eps)
    if not a <= u <= b:
        raise ValueError(f"mean(x^2)+eps = {u:.4g} lies outside the fitted interval [{a}, {b}]; "
                         "the Chebyshev series diverges there and the FHE side cannot check it")
    r = _cheb.chebval((2.0 * u - (a + b)) / (b - a), coeffs)
    return x * r * gamma


class RMSNormResult(NamedTuple):
    """`rmsnorm`'s output ciphertext with its level report."""
    ct: object
    level_before: int
    level_after: int
    degree: int
    coeffs: tuple[float, ...]


def rmsnorm(sess, ct, gamma, *, eps: float = DEFAULT_EPS, d: int, interval=DEFAULT_INTERVAL,
            degree: int = DEFAULT_DEGREE) -> RMSNormResult:
    """``x * rsqrt(mean(x^2) + eps) * gamma`` on a token ciphertext, from the primitives.

    sess: a `perseus.Session` (anything with ``.inf`` and ``.options``); ct: a fresh
    token-basis ciphertext (`Session.encrypt`) holding ``d`` real features; gamma: length
    ``d``; interval: the ``[a, b]`` window of ``mean(x^2) + eps`` the rsqrt series is fitted
    on (a calibration window -- the mirror refuses inputs outside it, the FHE side cannot);
    degree: the series degree (``rmsnorm_depth(degree)`` levels; see `level_budget`).

    Returns an `RMSNormResult` with ``ct.level`` before and after; the level report is
    logged at INFO. A warning is logged when the degree does not fit the level budget (the
    runtime then bootstraps mid-chain and the level report stops being a pure count).
    """
    inf = sess.inf
    fhe = inf.fhe
    d = int(d)
    degree = int(degree)
    dim, hid = int(inf.size.dim), int(inf.size.hidDim)
    slots = int(inf.slots)
    if not 0 < d <= dim:
        raise ValueError(f"d must be in 1..{dim} (the session's real width), got {d}")
    if hid <= 0 or slots % hid != 0:
        raise ValueError(f"{slots} slots are not a multiple of hidDim {hid}: not the cachemir "
                         "token layout this example relies on")
    if eps < 0:
        raise ValueError(f"eps must be >= 0, got {eps}")
    if degree < 1:
        raise ValueError(f"degree must be >= 1, got {degree}")
    interval = _check_interval(interval)
    g = np.asarray(gamma, dtype=np.float64)
    if g.ndim != 1 or g.shape[0] != d:
        raise ValueError(f"gamma must be 1-D of length d={d}, got shape {g.shape}")
    if not np.isfinite(g).all():
        raise ValueError("gamma contains NaN/inf")
    packing = str(ct.packing)
    if packing not in _SLOT_LAYOUT_PACKINGS:
        raise ValueError(f"ct is packed as {packing!r}; this example relies on the cachemir "
                         "token layout (lane i at slot i*t) and would mis-sum any other")
    t = slots // hid
    needed = {1 << k for k in range(int(math.log2(slots)))}
    loaded = {int(s) for s in (getattr(fhe, "loaded_rot_steps", None) or ())}
    if loaded:
        missing = sorted(needed - loaded)
        if missing:
            raise ValueError(f"sum_slots over {slots} slots needs rotation keys "
                             f"{sorted(needed)}; not loaded: {missing}")
    depth = rmsnorm_depth(degree)
    budget = level_budget(sess, ct)
    if depth > budget:
        log.warning("RMSNorm degree %d needs %d levels but only %d remain before the reactive "
                    "bootstrap (level %d, limit %d): the runtime will bootstrap mid-chain and "
                    "the level report is not a pure count",
                    degree, depth, budget, int(ct.level), int(fhe.level_limit()))

    before = int(ct.level)
    sq = fhe.square(ct)                                    # x^2 on the lanes, 0 elsewhere
    total = inf.sum_slots(sq, slots)                       # sum(x^2) broadcast to every slot
    coeffs = rsqrt_cheb_coeffs(interval, degree)
    a2, b2 = fhe_interval(interval, eps, d)
    r = inf.eval_chebyshev(total, [float(c) for c in coeffs], float(a2), float(b2))
    xg = inf.mult_pt(ct, gamma_to_slots(g, slots, t))      # x * gamma, lane layout
    out = fhe.mult(xg, r)
    after = int(out.level)
    unit = _level_unit(sess)
    log.info("rmsnorm: degree %d, levels %d -> %d (consumed %d; formula %d levels x %d primes)",
             degree, before, after, after - before, depth, unit)
    return RMSNormResult(out, before, after, degree, coeffs)
