"""Calibration keyed to the actual conditional estimator and data-generating design."""

from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from typing import Any
import numpy as np
from .model import FitError, FloatArray, MODEL_VERSION, Mode


@dataclass(frozen=True)
class Calibration:
    """Independent null scores, with failure and model/configuration metadata."""

    scores: FloatArray
    n: int
    mode: Mode
    seed: int
    design: str
    estimator_options: dict[str, Any]
    null: str = "rw"
    version: str = MODEL_VERSION

    @property
    def failures(self) -> int:
        return int((~np.isfinite(self.scores)).sum())

    @property
    def identity(self) -> str:
        metadata = {
            "n": self.n,
            "mode": self.mode,
            "design": self.design,
            "null": self.null,
            "version": self.version,
            "options": self.estimator_options,
        }
        return hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()

    def critical_value(self, alpha: float = 0.05) -> float:
        """Lower quantile for the negative log-ratio; refuse incomplete calibration."""
        if not 0 < alpha < 1 or len(self.scores) < 20:
            raise ValueError(
                "Need a valid level and at least 20 calibration replications"
            )
        if self.failures:
            raise FitError(
                "Calibration contains failed fits; no validated critical value is available"
            )
        return float(np.quantile(self.scores, alpha))

    def validate(
        self, *, n: int, mode: Mode, design: str, estimator_options: dict[str, Any]
    ) -> None:
        """Reject cached quantiles from another model, design or estimator."""
        expected = Calibration(
            np.empty(0), n, mode, self.seed, design, estimator_options, self.null
        )
        if self.version != MODEL_VERSION or self.identity != expected.identity:
            raise ValueError("Calibration model/configuration mismatch")


def rejection_summary(scores: FloatArray, cutoff: float) -> dict[str, float | int]:
    """Failure-aware rejection rates and worst-case unconditional bounds."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(cutoff):
        raise ValueError("Need scores and a finite calibrated cutoff")
    valid = np.isfinite(values)
    rejected = int((values[valid] < cutoff).sum())
    failed = int((~valid).sum())
    return {
        "total": len(values),
        "successful": int(valid.sum()),
        "failed": failed,
        "rejected": rejected,
        "conditional_rate": (
            rejected / int(valid.sum()) if valid.any() else float("nan")
        ),
        "unconditional_lower": rejected / len(values),
        "unconditional_upper": (rejected + failed) / len(values),
    }


def wilson_interval(
    successes: int, n: int, z: float = 2.5758293035489004
) -> tuple[float, float]:
    """99% Wilson binomial interval by default, not a normal approximation."""
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("Invalid binomial counts")
    rate = successes / n
    d = 1 + z * z / n
    center = (rate + z * z / (2 * n)) / d
    half = z * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / d
    return float(center - half), float(center + half)
