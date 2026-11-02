"""Concentrated likelihood with analytic GLS beta and variance scale profiling."""

from __future__ import annotations
from typing import Any
import numpy as np
from .model import FloatArray, Mode


def _profile_impl(
    x: FloatArray,
    y: FloatArray,
    rho: float,
    fraction: float,
    legacy: bool,
    estimate_beta: bool,
) -> tuple[float, float, float]:
    n = len(x) - 1
    p = 0.0
    mx = 0.0
    my = 0.0
    rx = x[0]
    ry = y[0]
    if legacy and fraction == 1.0:
        mx = x[0]
        my = y[0]
        rx = 0.0
        ry = 0.0
    xx = 0.0
    xy = 0.0
    logdet = 0.0
    wx = np.empty(n)
    wy = np.empty(n)
    q = fraction
    r = 1 - fraction
    km = 0.0
    if legacy:
        if fraction == 1.0:
            km = 1.0
        elif fraction > 0.0:
            sr = np.sqrt(r)
            rad = np.sqrt((1 + rho) ** 2 * r + 4 * q)
            km = 2 * q / (sr * (rad + (1 + rho) * sr) + 2 * q)
    # Directly accumulate profiled GLS sufficient statistics. Centering x/y
    # before this routine prevents cancellation from large price levels.
    for i in range(1, len(x)):
        mx *= rho
        my *= rho
        vx = x[i] - mx - rx
        vy = y[i] - my - ry
        variance = 1.0 + (1 - rho) ** 2 * p
        gain = km if legacy else (q + (rho * rho - rho) * p) / variance
        mx += gain * vx
        my += gain * vy
        rx += (1 - gain) * vx
        ry += (1 - gain) * vy
        if legacy:
            variance = 1.0
        else:
            p = (p * (rho * rho * r + q) + q * r) / variance
        wx[i - 1] = vx / np.sqrt(variance)
        wy[i - 1] = vy / np.sqrt(variance)
        xx += wx[i - 1] * wx[i - 1]
        xy += wx[i - 1] * wy[i - 1]
        logdet += np.log(variance)
    if estimate_beta and xx <= np.finfo(np.float64).tiny:
        return np.inf, np.nan, np.nan
    beta = xy / xx if estimate_beta else 0.0
    sse = 0.0
    for i in range(n):
        sse += (wy[i] - beta * wx[i]) ** 2
    if sse <= 0.0:
        # A singular/noiseless regression has no ordinary finite Gaussian MLE.
        return np.inf, beta, 0.0
    scale = sse / n
    nll = 0.5 * (n * (np.log(2 * np.pi * scale) + 1.0) + logdet)
    return nll, beta, scale


try:
    from numba import njit
except ImportError:
    _compiled: Any = _profile_impl
else:
    _compiled = njit(cache=True)(_profile_impl)


def profile(
    x: FloatArray,
    y: FloatArray,
    rho: float,
    fraction: float,
    mode: Mode = "exact",
    *,
    estimate_beta: bool = True,
) -> tuple[float, float, float]:
    """Profile beta and total process variance at fixed rho and noise fraction."""
    if (
        not np.isfinite([rho, fraction]).all()
        or abs(rho) >= 1
        or not 0 <= fraction <= 1
    ):
        return float("inf"), float("nan"), float("nan")
    result = _compiled(
        x, y, float(rho), float(fraction), mode == "legacy", estimate_beta
    )
    return float(result[0]), float(result[1]), float(result[2])
