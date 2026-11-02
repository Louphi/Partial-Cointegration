"""Explicit model contracts for conditional partial cointegration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
Mode = Literal["exact", "legacy"]
MODEL_VERSION = "conditional-v1"


class FitError(ValueError):
    """Data/model cannot produce an accepted finite likelihood estimate."""


@dataclass(frozen=True)
class Parameters:
    """Price-level model. Alpha is a fixed location convention, not estimated."""

    beta: float
    rho: float
    sigma_m: float
    sigma_r: float
    alpha: float = 0.0

    def __post_init__(self) -> None:
        values = [self.beta, self.rho, self.sigma_m, self.sigma_r, self.alpha]
        if not np.isfinite(values).all():
            raise ValueError("Parameters must be finite")
        if abs(self.rho) >= 1:
            raise ValueError("rho must satisfy abs(rho) < 1")
        if (
            min(self.sigma_m, self.sigma_r) < 0
            or self.sigma_m**2 + self.sigma_r**2 <= 0
        ):
            raise ValueError(
                "Noise standard deviations must be nonnegative with positive total variance"
            )

    @property
    def variance_fraction(self) -> float:
        """Fraction of differenced spread variance due to mean reversion."""
        q, r = self.sigma_m**2, self.sigma_r**2
        return float(2 * q / (2 * q + (1 + self.rho) * r))


@dataclass(frozen=True)
class FilterState:
    """Posterior state with an explicit identity and covariance."""

    mean: FloatArray
    covariance: FloatArray
    parameters: Parameters
    mode: Mode
    timestamp: str | None = None
    version: str = MODEL_VERSION


@dataclass(frozen=True)
class FilterResult:
    """Forward filtered components and their conditional innovation density."""

    m: FloatArray
    r: FloatArray
    errors: FloatArray
    variances: FloatArray
    log_likelihood: float
    final_state: FilterState


@dataclass(frozen=True)
class OptimizerDiagnostic:
    """One attempted optimizer run; acceptance is explicit."""

    candidate: str
    success: bool
    finite: bool
    message: str
    evaluations: int
    objective: float | None


@dataclass(frozen=True)
class FitResult:
    """Accepted nested candidate and complete optimization diagnostics."""

    parameters: Parameters
    log_likelihood: float
    null_log_likelihood: float
    ar_log_likelihood: float
    selected: str
    mode: Mode
    diagnostics: tuple[OptimizerDiagnostic, ...]
    n: int
    seed: int
    normalization: str = "alpha=0; M0=0,R0=w0,P0=0"
    version: str = MODEL_VERSION

    @property
    def log_ratio(self) -> float:
        """Legacy negative log-ratio; smaller values favor the alternative."""
        return self.null_log_likelihood - self.log_likelihood

    @property
    def statistic(self) -> float:
        """Equivalent positive likelihood-ratio statistic."""
        return -2 * self.log_ratio

    @property
    def rho_identified(self) -> bool:
        """Rho disappears under a pure random walk."""
        return self.parameters.sigma_m > 0

    def legacy_tuple(self, lrt: bool = True) -> tuple[float, ...]:
        """Adapter for inherited tuple consumers; alpha is normalized."""
        p = self.parameters
        base = (p.alpha, p.beta, p.rho, p.sigma_m, p.sigma_r, self.log_likelihood)
        return base + (self.null_log_likelihood, self.log_ratio) if lrt else base
