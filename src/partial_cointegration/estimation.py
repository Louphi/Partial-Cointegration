"""Nested conditional MLE using profiled GLS and observed optimizer diagnostics."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from numpy.typing import ArrayLike
from scipy import optimize
from .data import paired_arrays
from .likelihood import profile
from .model import FitError, FitResult, Mode, OptimizerDiagnostic, Parameters


@dataclass(frozen=True)
class FitOptions:
    """Optimization configuration; no stochastic global state is used."""

    seed: int = 0
    random_starts: int = 2
    rho_limit: float = 0.9999
    max_iterations: int = 300
    tolerance: float = 1e-10

    def __post_init__(self) -> None:
        if (
            self.random_starts < 0
            or not 0 < self.rho_limit < 1
            or self.max_iterations < 1
            or self.tolerance <= 0
        ):
            raise ValueError("Invalid optimizer options")


def fit_pair(
    x: ArrayLike,
    y: ArrayLike,
    *,
    mode: Mode = "exact",
    options: FitOptions | None = None,
    _estimate_beta: bool = True,
) -> FitResult:
    """Fit RW/AR/mixed models with common initial state and exact boundaries.

    Beta and total variance are analytically profiled for each covariance shape.
    The accepted alternative is the best converged mixed or nested candidate;
    failures never become ordinary successful parameter tuples.
    """
    if mode not in ("exact", "legacy"):
        raise ValueError("Unknown likelihood mode")
    config = options or FitOptions()
    a, b = paired_arrays(x, y)
    da, db = np.diff(a), np.diff(b)
    if _estimate_beta and np.linalg.norm(da) == 0:
        raise FitError("Beta is unidentified for a constant predictor")
    if (
        _estimate_beta and np.linalg.matrix_rank(np.column_stack([da, db])) < 2
    ) or np.linalg.norm(db) == 0:
        raise FitError(
            "Deterministic/rank-deficient differences have no nonsingular Gaussian fit"
        )
    # Common-initial-state exact likelihood is invariant to level translations.
    # Legacy AR initialization is not, so preserve its levels explicitly.
    sx = float(np.sqrt(np.mean(da**2))) if _estimate_beta else 1.0
    sy = float(np.sqrt(np.mean(db**2)))
    ax = (a - a[0]) / sx if mode == "exact" else a / sx
    by = (b - b[0]) / sy if mode == "exact" else b / sy
    n = len(a) - 1
    diagnostics: list[OptimizerDiagnostic] = []
    candidates: list[tuple[float, float, float, float, float, str]] = []
    null = profile(ax, by, 0.0, 0.0, mode, estimate_beta=_estimate_beta)
    if not np.isfinite(null).all() or null[2] <= 0:
        raise FitError("Random-walk candidate has a singular or nonfinite likelihood")
    candidates.append((*null, 0.0, 0.0, "rw"))
    diagnostics.append(
        OptimizerDiagnostic(
            "rw", True, True, "Analytic conditional GLS", 1, float(null[0])
        )
    )

    def attempt(
        name: str, start: list[float], ar: bool = False, method: str = "L-BFGS-B"
    ) -> None:
        def objective(z: Any) -> float:
            return profile(
                ax,
                by,
                float(z[0]),
                1.0 if ar else float(z[1]),
                mode,
                estimate_beta=_estimate_beta,
            )[0]

        if ar:
            grid = np.linspace(-config.rho_limit, config.rho_limit, 17)
            scores = np.array([objective([rho]) for rho in grid])
            brackets = []
            for i in range(len(grid)):
                if (i == 0 or scores[i] <= scores[i - 1]) and (
                    i == len(grid) - 1 or scores[i] <= scores[i + 1]
                ):
                    brackets.append(
                        (grid[max(0, i - 1)], grid[min(len(grid) - 1, i + 1)])
                    )
            scalar_results = [
                optimize.minimize_scalar(
                    lambda rho: objective([rho]),
                    bounds=bracket,
                    method="bounded",
                    options={
                        "maxiter": config.max_iterations,
                        "xatol": config.tolerance,
                    },
                )
                for bracket in brackets
            ]
            accepted = [r for r in scalar_results if r.success and np.isfinite(r.fun)]
            result = min(accepted or scalar_results, key=lambda r: r.fun)
            rho = float(result.x)
            fraction = 1.0
        else:
            bounds = [(-config.rho_limit, config.rho_limit), (0.0, 1.0)]
            result = optimize.minimize(
                objective,
                start,
                bounds=bounds,
                method=method,
                options=(
                    {
                        "maxiter": config.max_iterations,
                        "ftol": config.tolerance,
                        "gtol": 1e-7,
                    }
                    if method == "L-BFGS-B"
                    else {
                        "maxiter": config.max_iterations,
                        "ftol": config.tolerance,
                        "xtol": config.tolerance,
                    }
                ),
            )
            rho = float(result.x[0])
            fraction = float(result.x[1])
        value = profile(ax, by, rho, fraction, mode, estimate_beta=_estimate_beta)
        finite = bool(np.isfinite(value).all() and value[2] > 0)
        diagnostics.append(
            OptimizerDiagnostic(
                name,
                bool(result.success),
                finite,
                str(result.message),
                int(result.nfev),
                float(value[0]) if finite else None,
            )
        )
        if result.success and finite:
            candidates.append((*value, rho, fraction, name))

    attempt("ar", [0.0], True)
    starts = [[-0.5, 0.5], [0.2, 0.2], [0.6, 0.5], [0.9, 0.8], [0.98, 0.3]]
    rng = np.random.default_rng(config.seed)
    starts.extend(
        [
            [float(rng.uniform(-0.9, 0.95)), float(rng.uniform(0.05, 0.95))]
            for _ in range(config.random_starts)
        ]
    )
    for start in starts:
        attempt("mixed", start)
    ar_candidates = [c for c in candidates if c[-1] == "ar"]
    mixed_candidates = [c for c in candidates if c[-1].startswith("mixed")]
    if not mixed_candidates and ar_candidates:
        best_ar = min(ar_candidates, key=lambda c: c[0])
        attempt("mixed-powell", [best_ar[3], 0.95], method="Powell")
        mixed_candidates = [c for c in candidates if c[-1].startswith("mixed")]
    if not ar_candidates or not mixed_candidates:
        error = FitError("No accepted AR or mixed optimization; inspect diagnostics")
        error.diagnostics = tuple(diagnostics)  # type: ignore[attr-defined]
        raise error
    best = min(candidates, key=lambda c: c[0])
    score, beta, scale, rho, fraction, selected = best
    p = Parameters(
        beta * sy / sx,
        0.0 if fraction == 0 else rho,
        float(np.sqrt(scale * fraction) * sy),
        float(np.sqrt(scale * (1 - fraction)) * sy),
    )
    offset = n * np.log(sy)
    ar_score = min(c[0] for c in ar_candidates)
    return FitResult(
        p,
        float(-score - offset),
        float(-null[0] - offset),
        float(-ar_score - offset),
        selected,
        mode,
        tuple(diagnostics),
        len(a),
        config.seed,
        normalization=(
            "alpha=0; M0=0,R0=w0,P0=0"
            if mode == "exact"
            else "alpha=0; legacy fixed gains/q+r; pure-AR M0=w0"
        ),
    )


def fit_residuals(
    w: ArrayLike, *, mode: Mode = "exact", options: FitOptions | None = None
) -> FitResult:
    """Fit PAR noise without a regression coefficient or deterministic drift."""
    values = np.asarray(w, dtype=float)
    return fit_pair(
        np.zeros_like(values), values, mode=mode, options=options, _estimate_beta=False
    )
