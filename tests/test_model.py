"""Independent covariance-density tests, identification and fitting regressions."""

from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from partial_cointegration import (
    Parameters,
    filter_residuals,
    fit_pair,
    FitOptions,
    FitError,
)
from partial_cointegration.likelihood import profile
from partial_cointegration.simulation import simulate_pair


def dense(w, p):
    n = len(w) - 1
    bm = np.zeros((n, n))
    br = np.tril(np.ones((n, n))) * p.sigma_r
    for i in range(n):
        for j in range(i + 1):
            bm[i, j] = p.sigma_m * p.rho ** (i - j)
    c = bm @ bm.T + br @ br.T
    z = np.linalg.solve(np.linalg.cholesky(c), w[1:] - w[0])
    return -0.5 * (n * np.log(2 * np.pi) + np.linalg.slogdet(c)[1] + z @ z)


@pytest.mark.parametrize("case", range(60))
def test_exact_matches_independent_dense_density(case):
    rng = np.random.default_rng(case)
    rho = [-0.8, 0, 0.6, 0.9, 0.99][case % 5]
    sm, sr = (
        [0.0, 1.0]
        if case % 6 == 0
        else [1.0, 0.0] if case % 6 == 1 else rng.uniform(0.3, 2, 2)
    )
    p = Parameters(1, rho, sm, sr)
    w = rng.normal(size=[5, 10, 30][case % 3]) + 12
    result = filter_residuals(w, p)
    assert result.log_likelihood == pytest.approx(dense(w, p), abs=1e-9, rel=1e-10)
    np.testing.assert_allclose(result.m + result.r, w, atol=1e-10)
    assert np.linalg.eigvalsh(result.final_state.covariance).min() >= -1e-10
    assert result.m[0] == 0 and result.r[0] == w[0]


def test_profile_equals_filter_at_profiled_parameters():
    pair = simulate_pair(80, Parameters(1.3, 0.7, 1, 1), seed=18)
    for rho, fraction in [(0, 0), (0.6, 1), (-0.6, 0.4), (0.9, 0.8)]:
        nll, beta, scale = profile(
            pair.x - pair.x[0], pair.y - pair.y[0], rho, fraction
        )
        p = Parameters(
            beta, rho, np.sqrt(scale * fraction), np.sqrt(scale * (1 - fraction))
        )
        result = filter_residuals(pair.y - beta * pair.x, p)
        assert -nll == pytest.approx(result.log_likelihood, abs=1e-8)


def test_continuation_and_missing_prices_are_causal():
    p = Parameters(1, 0.6, 1, 1)
    w = np.array([4.0, 1.0, 2.0, np.nan, 7.0, 3.0, 2.0, 5.0])
    full = filter_residuals(w, p)
    train = filter_residuals(w[:4], p)
    test = filter_residuals(w[4:], p, state=train.final_state)
    np.testing.assert_allclose(full.m[4:], test.m, atol=1e-9)
    np.testing.assert_allclose(
        full.final_state.covariance, test.final_state.covariance, atol=1e-9
    )
    assert full.log_likelihood == pytest.approx(
        train.log_likelihood + test.log_likelihood
    )
    changed = w.copy()
    changed[-1] = 1000
    np.testing.assert_allclose(full.m[:-1], filter_residuals(changed, p).m[:-1])
    assert np.isnan(full.errors[3])


def test_domain_alignment_and_location():
    for sm, sr, rho in [(-1, 1, 0.6), (0, 0, 0.6), (1, 1, 1)]:
        with pytest.raises(ValueError):
            Parameters(1, rho, sm, sr)
    with pytest.raises(ValueError):
        filter_residuals([np.nan, 1], Parameters(1, 0.6, 1, 1))
    x = pd.Series([1, 2, 3, 4], index=[0, 1, 2, 3])
    y = x.iloc[::-1]
    with pytest.raises(ValueError):
        fit_pair(x, y)
    w = np.array([1.0, 2.0, -0.4, 3.0])
    p = Parameters(1, 0.6, 1, 1)
    assert filter_residuals(w, p).log_likelihood == pytest.approx(
        filter_residuals(w + 100, p).log_likelihood
    )
    with pytest.raises(FitError):
        fit_pair(np.arange(10), np.arange(10) * 2)
    with pytest.raises(FitError):
        fit_pair(np.ones(10), np.arange(10))


def test_failed_optimizer_is_observable():
    pair = simulate_pair(30, Parameters(1, 0.6, 1, 1), seed=4)
    with patch(
        "partial_cointegration.estimation.optimize.minimize",
        return_value=SimpleNamespace(
            success=False, x=np.array([0.5, 0.5]), message="forced failure", nfev=1
        ),
    ):
        with pytest.raises(FitError, match="No accepted"):
            fit_pair(pair.x, pair.y)


@pytest.mark.parametrize("seed", range(100))
def test_nested_candidates_and_seed_reproducibility(seed):
    pair = simulate_pair(100, Parameters(1, 0.6, 1, 1), seed=seed)
    result = fit_pair(pair.x, pair.y, options=FitOptions(seed=seed, random_starts=1))
    assert result.log_likelihood >= max(
        result.null_log_likelihood, result.ar_log_likelihood
    ) - 1e-7 * (1 + abs(result.log_likelihood))
    assert result.parameters.alpha == 0
    assert result.statistic >= -1e-8
    if seed == 0:
        again = fit_pair(pair.x, pair.y, options=FitOptions(seed=seed, random_starts=1))
        assert result == again


@pytest.mark.parametrize("index", [281, 802])
def test_boundary_line_search_failure_uses_explicit_successful_fallback(index):
    seed, fitseed = map(
        int, np.random.SeedSequence([20261025, 1, 4, index]).generate_state(2)
    )
    pair = simulate_pair(100, Parameters(1, -0.8, 10, 1), seed=seed)
    fit = fit_pair(pair.x, pair.y, options=FitOptions(seed=fitseed))
    assert np.isfinite(fit.log_likelihood)
    assert any(d.candidate == "mixed-powell" and d.success for d in fit.diagnostics)
    assert any(not d.success for d in fit.diagnostics)
