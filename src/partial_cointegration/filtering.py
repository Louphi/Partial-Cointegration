"""Exact conditional forward filtering; missing observations are predict-only."""

from __future__ import annotations
from typing import Any
import numpy as np
from numpy.typing import ArrayLike
from .model import FilterResult, FilterState, Mode, Parameters, MODEL_VERSION


def _gains(p: Parameters) -> tuple[float, float]:
    if p.sigma_m == 0:
        return 0.0, 1.0
    if p.sigma_r == 0:
        return 1.0, 0.0
    q, sr = p.sigma_m**2, p.sigma_r
    rad = np.sqrt((1 + p.rho) ** 2 * sr**2 + 4 * q)
    km = float(2 * q / (sr * (rad + (1 + p.rho) * sr) + 2 * q))
    return km, 1 - km


def filter_residuals(
    w: ArrayLike,
    parameters: Parameters,
    *,
    state: FilterState | None = None,
    mode: Mode = "exact",
    timestamps: Any = None,
) -> FilterResult:
    """Filter observations, conditioning on the first when no state is supplied.

    Exact mode uses common M0=0,R0=w0,P0=0, including pure AR/RW boundaries.
    Legacy mode preserves the old fixed-gain, q+r scoring convention and its AR
    initialization. It is a comparison score, not the exact Gaussian density.
    """
    if mode not in ("exact", "legacy"):
        raise ValueError("Unknown likelihood mode")
    values = np.asarray(w, dtype=float)
    if values.ndim != 1 or len(values) == 0 or np.isinf(values).any():
        raise ValueError("Residuals must be a nonempty vector without infinities")
    if timestamps is not None:
        if len(timestamps) != len(values):
            raise ValueError("Timestamp length mismatch")
        import pandas as pd

        dates = pd.Index(timestamps)
        if not dates.is_unique or not dates.is_monotonic_increasing or dates.hasnans:
            raise ValueError("Timestamps must be unique and increasing")
        if (
            state
            and state.timestamp
            and pd.Timestamp(timestamps[0]) <= pd.Timestamp(state.timestamp)
        ):
            raise ValueError("Continuation must start after the saved state")
    elif state is not None and state.timestamp is not None:
        raise ValueError("Timestamped state requires continuation timestamps")
    n = len(values)
    means = np.empty((n, 2))
    errors = np.full(n, np.nan)
    variances = np.full(n, np.nan)
    if state is None:
        if not np.isfinite(values[0]):
            raise ValueError("First residual must be observed to initialize")
        mean = np.array([0.0, values[0]])
        if mode == "legacy" and parameters.sigma_r == 0:
            mean = np.array([values[0], 0.0])
        covariance = np.zeros((2, 2))
        means[0] = mean
        start = 1
    else:
        if (
            state.parameters != parameters
            or state.mode != mode
            or state.version != MODEL_VERSION
        ):
            raise ValueError("Filter state model identity mismatch")
        mean, covariance = state.mean.copy(), state.covariance.copy()
        if (
            mean.shape != (2,)
            or covariance.shape != (2, 2)
            or not np.isfinite(mean).all()
            or not np.isfinite(covariance).all()
        ):
            raise ValueError("Invalid saved filter state")
        if not np.allclose(covariance, covariance.T, atol=1e-12) or np.linalg.eigvalsh(
            covariance
        ).min() < -1e-10 * max(1.0, float(np.linalg.norm(covariance))):
            raise ValueError("Saved covariance must be symmetric positive semidefinite")
        start = 0
    transition = np.diag([parameters.rho, 1.0])
    noise = np.diag([parameters.sigma_m**2, parameters.sigma_r**2])
    h = np.ones(2)
    ll = 0.0
    for i in range(start, n):
        mean = transition @ mean
        covariance = transition @ covariance @ transition.T + noise
        if np.isfinite(values[i]):
            error = float(values[i] - mean.sum())
            variance = float(h @ covariance @ h)
            if not np.isfinite(variance) or variance <= 0:
                raise ValueError("Nonpositive/nonfinite innovation variance")
            gain = covariance @ h / variance
            score_variance = variance
            if mode == "legacy":
                if i > 0 and not np.isfinite(values[i - 1]):
                    raise ValueError(
                        "Legacy fixed-gain mode cannot continue across missing observations"
                    )
                gain = np.array(_gains(parameters))
                score_variance = float(np.trace(noise))
            mean += gain * error
            update = np.eye(2) - np.outer(gain, h)
            covariance = update @ covariance @ update.T
            covariance = 0.5 * (covariance + covariance.T)
            errors[i], variances[i] = error, score_variance
            ll -= 0.5 * (np.log(2 * np.pi * score_variance) + error**2 / score_variance)
        means[i] = mean
    timestamp = str(timestamps[-1]) if timestamps is not None else None
    final = FilterState(mean.copy(), covariance.copy(), parameters, mode, timestamp)
    return FilterResult(means[:, 0], means[:, 1], errors, variances, float(ll), final)
