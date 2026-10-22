"""The error taxonomy in one importable place.

C++-raised (markers translated by the bindings; all subclass _core.FHEError): FHEError, PlanError [plan_*],
MaskError [mask_*], LayoutError [layout_error]. Python-raised: PlanContractError (plan loaded under a different env
than it was captured in), BundleError (key bundle manifest mismatch), SecurityError (trust-boundary invariant).
Imports without the compiled extension (stand-in classes with the same hierarchy), so
`except perseus.errors.FHEError` is always valid; _client errors are re-raised as these classes by EncClient.
"""
try:
    from ._pc import FHEError, LayoutError, MaskError, PlanError
except ImportError:
    try:
        from ._client import FHEError, LayoutError, MaskError, PlanError  # type: ignore[import-not-found]
    except ImportError:   # no extension built: keep the names and the hierarchy importable

        class FHEError(Exception):
            """Stand-in for perseus._core.FHEError when the extension is absent."""

        class PlanError(FHEError):
            """Stand-in for perseus._core.PlanError when the extension is absent."""

        class MaskError(FHEError):
            """Stand-in for perseus._core.MaskError when the extension is absent."""

        class LayoutError(FHEError):
            """Stand-in for perseus._core.LayoutError when the extension is absent."""


from .plan.contract import PlanContractError  # noqa: E402


class BundleError(RuntimeError):
    """A key bundle cannot be used by this session: its manifest (parameters, chain,
    model family) disagrees with what the caller asked for, or is missing."""


class SecurityError(RuntimeError):
    """A trust-boundary invariant would be violated."""


__all__ = [
    "BundleError",
    "FHEError",
    "LayoutError",
    "MaskError",
    "PlanContractError",
    "PlanError",
    "SecurityError",
]
