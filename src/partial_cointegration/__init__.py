"""Public conditional model API; historical modules use compatibility adapters."""

from .model import Parameters, FilterState, FilterResult, FitResult, FitError
from .filtering import filter_residuals
from .estimation import fit_pair, FitOptions

__all__ = [
    "Parameters",
    "FilterState",
    "FilterResult",
    "FitResult",
    "FitError",
    "filter_residuals",
    "fit_pair",
    "FitOptions",
]
