"""Named DGPs and immutable failure-aware summaries with independent streams."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal
import numpy as np
from .model import FloatArray, Parameters


@dataclass(frozen=True)
class SimulatedPair:
    """Observed pair plus states and generation provenance."""

    x: FloatArray
    y: FloatArray
    m: FloatArray
    r: FloatArray
    seed: int
    design: str
    initialization: str


def simulate_pair(
    n: int,
    parameters: Parameters,
    *,
    seed: int = 0,
    sigma_x: float = 0.0236,
    design: Literal["geometric", "additive"] = "geometric",
    initialization: Literal["known", "stationary", "legacy"] = "known",
    initial_price: float = 100.0,
) -> SimulatedPair:
    """Generate explicit state initial conditions; RNG is never shared with fits."""
    if (
        n < 4
        or not np.isfinite([sigma_x, initial_price]).all()
        or sigma_x <= 0
        or initial_price <= 0
    ):
        raise ValueError("Invalid simulation configuration")
    if design not in ("geometric", "additive") or initialization not in (
        "known",
        "stationary",
        "legacy",
    ):
        raise ValueError("Unknown generation design or initialization")
    sx, sm, sr = np.random.SeedSequence(seed).spawn(3)
    dx = np.random.default_rng(sx).normal(0, sigma_x, n)
    em = np.random.default_rng(sm).normal(0, parameters.sigma_m, n)
    er = np.random.default_rng(sr).normal(0, parameters.sigma_r, n)
    x = np.empty(n)
    m = np.empty(n)
    r = np.empty(n)
    x[0] = initial_price
    m[0] = 0.0
    r[0] = 0.0
    if initialization == "stationary":
        initial_rng = np.random.default_rng(np.random.SeedSequence([seed, 991]))
        m[0] = initial_rng.normal(
            0, parameters.sigma_m / np.sqrt(1 - parameters.rho**2)
        )
    elif initialization == "legacy":
        m[0] = em[0]
        r[0] = er[0]
        x[0] = (
            initial_price * np.exp(dx[0])
            if design == "geometric"
            else initial_price + dx[0]
        )
    for i in range(1, n):
        x[i] = x[i - 1] * np.exp(dx[i]) if design == "geometric" else x[i - 1] + dx[i]
        m[i] = parameters.rho * m[i - 1] + em[i]
        r[i] = r[i - 1] + er[i]
    return SimulatedPair(
        x,
        parameters.alpha + parameters.beta * x + m + r,
        m,
        r,
        seed,
        design,
        initialization,
    )


def summary(estimates: FloatArray, truth: FloatArray) -> dict[str, Any]:
    """Summarize accepted replications with explicit failure counts and MC error."""
    values = np.asarray(estimates, dtype=float)
    actual = np.asarray(truth, dtype=float)
    if values.ndim != 2 or actual.shape != values.shape[1:]:
        raise ValueError("Use rows=replications and named parameter columns")
    valid = np.isfinite(values).all(axis=1)
    good = values[valid]
    if not len(good):
        return {
            "total": len(values),
            "successful": 0,
            "failed": len(values),
            "mean": None,
            "bias": None,
            "mse": None,
            "bias_mcse": None,
        }
    errors = good - actual
    return {
        "total": len(values),
        "successful": int(valid.sum()),
        "failed": int((~valid).sum()),
        "mean": good.mean(axis=0).tolist(),
        "bias": errors.mean(axis=0).tolist(),
        "mse": np.mean(errors**2, axis=0).tolist(),
        "rmse": np.sqrt(np.mean(errors**2, axis=0)).tolist(),
        "bias_mcse": (
            (errors.std(axis=0, ddof=1) / np.sqrt(len(good))).tolist()
            if len(good) > 1
            else None
        ),
        "quantiles": np.quantile(good, [0.025, 0.5, 0.975], axis=0).tolist(),
    }
