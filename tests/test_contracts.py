import dataclasses
import numpy as np
import pandas as pd
import pytest
from partial_cointegration import (
    Parameters,
    FilterState,
    filter_residuals,
    fit_pair,
    FitOptions,
)
from partial_cointegration.data import paired_arrays, validate_prices, members_at
from partial_cointegration.likelihood import _profile_impl, profile
from partial_cointegration.simulation import simulate_pair, summary
from partial_cointegration.signals import score_components, decisions
from partial_cointegration.study import formation_study, StudyConfig


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("fraction", [0.0, 0.4, 1.0])
def test_python_and_compiled_profile_against_independent_score(legacy, fraction):
    rng = np.random.default_rng(123)
    x = np.cumsum(rng.normal(size=30))
    y = 0.7 * x + rng.normal(size=30)
    result = _profile_impl(x, y, 0.6, fraction, legacy, True)
    np.testing.assert_allclose(
        result,
        profile(x, y, 0.6, fraction, "legacy" if legacy else "exact"),
        atol=1e-10,
    )
    nll, beta, scale = result
    p = Parameters(
        beta, 0.6, np.sqrt(scale * fraction), np.sqrt(scale * (1 - fraction))
    )
    filtered = filter_residuals(y - beta * x, p, mode="legacy" if legacy else "exact")
    assert filtered.log_likelihood == pytest.approx(-nll, abs=1e-9)
    fixed = _profile_impl(np.zeros(30), y, 0.6, fraction, legacy, False)
    assert np.isfinite(fixed).all() and fixed[1] == 0
    assert np.isinf(_profile_impl(np.zeros(30), y, 0.6, fraction, legacy, True)[0])
    assert np.isinf(_profile_impl(x, x, 0.6, fraction, legacy, True)[0])
    assert np.isinf(profile(x, y, 1.0, fraction)[0])


def test_input_and_filter_state_contracts():
    p = Parameters(1, 0.6, 1, 1)
    w = [1.0, 2.0, 3.0, 4.0]
    for invalid in [[], [1, np.inf], [[1, 2], [3, 4]]]:
        with pytest.raises(ValueError):
            filter_residuals(invalid, p)
    with pytest.raises(ValueError):
        filter_residuals(w, p, mode="invalid")
    with pytest.raises(ValueError):
        Parameters(np.nan, 0.6, 1, 1)
    with pytest.raises(ValueError):
        FitOptions(random_starts=-1)
    with pytest.raises(ValueError):
        fit_pair(w, w, mode="invalid")
    with pytest.raises(ValueError):
        paired_arrays(pd.Series(w), w)
    with pytest.raises(ValueError):
        paired_arrays([1, 2], [2, 3])
    with pytest.raises(ValueError):
        paired_arrays([1, 2, 3, np.inf], w)
    dates = pd.date_range("2023-01-01", periods=4)
    state = filter_residuals(w, p, timestamps=dates).final_state
    for time in [[dates[0]], list(reversed(dates)), [dates[0]] * 4]:
        with pytest.raises(ValueError):
            filter_residuals(w, p, timestamps=time)
    with pytest.raises(ValueError):
        filter_residuals([2.0], p, state=state, timestamps=dates[-1:])
    with pytest.raises(ValueError):
        filter_residuals([2.0], p, state=state)
    with pytest.raises(ValueError):
        filter_residuals(
            [2.0],
            Parameters(1, 0.5, 1, 1),
            state=state,
            timestamps=pd.date_range("2024-01-01", periods=1),
        )
    for bad in [
        dataclasses.replace(state, mean=np.ones(3)),
        dataclasses.replace(state, covariance=np.array([[1.0, 2.0], [0.0, 1.0]])),
        dataclasses.replace(state, covariance=-np.eye(2)),
    ]:
        with pytest.raises(ValueError):
            filter_residuals(
                [2.0], p, state=bad, timestamps=pd.date_range("2024-01-01", periods=1)
            )
    with pytest.raises(ValueError):
        filter_residuals([1, np.nan, 2], p, mode="legacy")


def test_data_generation_and_normalization_validation():
    dates = pd.date_range("2023-01-01", periods=5)
    for frame in [
        pd.DataFrame({"A": [1, 0]}, index=dates[:2]),
        pd.DataFrame({"A": [1, np.inf]}, index=dates[:2]),
        pd.DataFrame([[1, 2]], columns=["A", "A"]),
        pd.DataFrame({"A": [1, 2]}, index=dates[:2][::-1]),
    ]:
        with pytest.raises(ValueError):
            validate_prices(frame)
    with pytest.raises(ValueError):
        validate_prices(
            pd.DataFrame({"A": [1, np.nan]}, index=dates[:2]), allow_missing=False
        )
    with pytest.raises(ValueError):
        members_at(
            pd.DataFrame(
                {"from": ["2023-01-02"], "thru": ["2023-01-01"], "co_tic": ["A"]}
            ),
            "2024-01-01",
        )
    for kwargs in [
        {"n": 3},
        {"n": 20, "design": "wrong"},
        {"n": 20, "initialization": "wrong"},
    ]:
        with pytest.raises(ValueError):
            simulate_pair(parameters=Parameters(1, 0.6, 1, 1), **kwargs)
    for initialization in ["stationary", "legacy"]:
        pair = simulate_pair(
            30,
            Parameters(1, 0.6, 1, 1),
            initialization=initialization,
            design="additive",
        )
        np.testing.assert_allclose(pair.y - pair.x, pair.m + pair.r)
    with pytest.raises(ValueError):
        summary(np.ones(4), np.ones(4))
    assert summary(np.full((4, 2), np.nan), np.ones(2))["successful"] == 0
    tr = pd.Series([1.0, 2.0, 3.0], index=dates[:3])
    te = pd.Series([4.0, 5.0], index=dates[3:])
    formation = score_components(tr, te, Parameters(1, 0.6, 1, 1), variant="formation")
    assert len(formation["testing_score"]) == 2
    with pytest.raises(ValueError):
        score_components(tr, te, Parameters(1, 0.6, 1, 1), window=1)
    with pytest.raises(ValueError):
        score_components(tr, tr, Parameters(1, 0.6, 1, 1))
    with pytest.raises(ValueError):
        score_components(tr, te, Parameters(1, 0, 0, 1))
    with pytest.raises(ValueError):
        decisions(te, entry=0, exit=1)
    assert not decisions(pd.Series([np.nan, 2.0, -1.0], index=dates[:3]))[0].iloc[0, 0]


@pytest.mark.parametrize("mode", ["exact", "legacy"])
def test_training_selection_future_invariance_and_qualified_data(mode):
    from partial_cointegration.simulation import simulate_pair

    a = simulate_pair(400, Parameters(1, 0.6, 2, 0.3), seed=17)
    b = simulate_pair(400, Parameters(1, 0.6, 2, 0.3), seed=29)
    dates = pd.bdate_range("2020-01-01", periods=400)
    prices = pd.DataFrame({"A": a.x, "B": a.y, "C": b.x, "D": b.y}, index=dates)
    members = pd.DataFrame(
        {
            "security_id": ["A", "B", "C", "D"],
            "from": ["2010-01-01"] * 4,
            "thru": [None] * 4,
            "sector": ["tech", "tech", "energy", "energy"],
        }
    )
    c = StudyConfig(
        mode=mode,
        normalization="formation",
        ranking_fraction=1.0,
        keep=2,
        rho_min=0.1,
        variance_min=0.1,
    )
    result = formation_study(prices.iloc[:300], prices.iloc[300:], members, config=c)
    assert result["candidate_count"] == 2
    assert result["selected"]
    assert result["ledger"] is not None
    changed = prices.iloc[300:].copy()
    changed.iloc[-1] *= 1.1
    perturbed = formation_study(prices.iloc[:300], changed, members, config=c)
    assert result["selected"] == perturbed["selected"]
    assert not result["certified_paper_replication"]
    rolling = formation_study(
        prices.iloc[:300],
        prices.iloc[300:],
        members,
        config=dataclasses.replace(c, normalization="rolling"),
    )
    assert rolling["candidate_count"] == 2
    with pytest.raises(ValueError):
        formation_study(
            prices.iloc[:300],
            prices.iloc[300:],
            members.drop(columns="sector"),
            config=c,
        )
    with pytest.raises(ValueError):
        formation_study(
            prices.iloc[:300], prices.iloc[300:].drop(columns="A"), members, config=c
        )
    with pytest.raises(ValueError):
        formation_study(prices.iloc[:300], prices.iloc[:300], members, config=c)
    with pytest.raises(ValueError):
        StudyConfig(window=1)
    with pytest.raises(ValueError):
        StudyConfig(mode="wrong")


def test_state_serialization_and_stale_report_rejection(tmp_path):
    import json
    from partial_cointegration.io import (
        state_to_dict,
        state_from_dict,
        estimator_fingerprint,
        load_current_report,
    )

    p = Parameters(1, 0.6, 1, 1)
    tr = filter_residuals([1.0, 2.0, 3.0, 4.0], p)
    saved = json.loads(json.dumps(state_to_dict(tr.final_state)))
    state = state_from_dict(saved)
    a = filter_residuals([3.0, 1.0], p, state=state)
    b = filter_residuals([1.0, 2.0, 3.0, 4.0, 3.0, 1.0], p)
    np.testing.assert_allclose(a.m, b.m[-2:])
    with pytest.raises(ValueError):
        state_from_dict(dict(saved, version="old"))
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"source_hash": "old", "full_design": True}))
    with pytest.raises(ValueError):
        load_current_report(path)
    path.write_text(
        json.dumps({"source_hash": estimator_fingerprint(), "full_design": True})
    )
    assert load_current_report(path)["full_design"]


def test_close_priority_and_replay_restrictions():
    from partial_cointegration.execution import (
        backtest_pairs,
        ExecutionConfig,
        replay_vectorbt,
    )

    dates = pd.date_range("2023-01-01", periods=5)
    p = pd.DataFrame({"A": [10.0] * 5, "B": [20.0] * 5}, index=dates)
    e = pd.DataFrame({"A/B": [True, False, False, False, False]}, index=dates)
    x = e.copy()
    ledger = backtest_pairs(p, {"A/B": ("A", "B", 1)}, e, x)
    assert ledger.orders.empty  # closing instruction wins while flat
    x.iloc[:, :] = False
    e.iloc[:, :] = False
    e.iloc[-2, 0] = True
    closed = backtest_pairs(p, {"A/B": ("A", "B", 1)}, e, x)
    with pytest.raises(ValueError, match="multiple same-bar"):
        replay_vectorbt(closed, p)
    with pytest.raises(ValueError):
        ExecutionConfig(fees=-1)
    with pytest.raises(ValueError):
        backtest_pairs(p, {}, e, x)
    with pytest.raises(ValueError):
        backtest_pairs(p, {"bad": ("A", "B", 1)}, e, x)


def test_overlapping_cohorts_share_capital_and_hac_is_dependence_aware():
    from partial_cointegration.portfolio import aggregate_cohorts, hac_mean_interval

    dates = pd.date_range("2023-01-01", periods=4)
    cohorts = {
        "A": pd.Series([1.0, 1.1, 1.2], index=dates[:3]),
        "B": pd.Series([1.0, 1.1, 1.2], index=dates[1:]),
    }
    out = aggregate_cohorts(cohorts, max_active=2)
    assert out["equity"].iloc[0] == 1
    assert out["equity"].iloc[-1] == pytest.approx(1.2)
    assert out["allocations"].capital.sum() == pytest.approx(1.0)
    assert (out["cash"] >= 0).all()
    with pytest.raises(ValueError):
        aggregate_cohorts(cohorts, max_active=1)
    with pytest.raises(ValueError):
        aggregate_cohorts({"A": pd.Series([2.0, 3.0], index=dates[:2])})
    rng = np.random.default_rng(2)
    returns = np.zeros(100)
    for i in range(1, 100):
        returns[i] = 0.8 * returns[i - 1] + rng.normal(0, 0.01)
    iid = hac_mean_interval(returns, lag=0)
    dependent = hac_mean_interval(returns, lag=6)
    assert dependent["standard_error"] > iid["standard_error"]
    with pytest.raises(ValueError):
        hac_mean_interval([0.1] * 10)


def test_committed_calibration_bank_is_source_bound_and_integrity_checked(tmp_path):
    import json
    from pathlib import Path
    from partial_cointegration.io import calibration_from_report

    path = (
        Path(__file__).resolve().parents[1] / "docs/results/scientific-validation.json"
    )
    report = json.loads(path.read_text())
    calibration = calibration_from_report(path, n=100)
    assert calibration.critical_value(0.05) == pytest.approx(
        report["calibration"]["100"]["quantiles"]["0.05"]
    )
    assert not calibration.scores.flags.writeable
    assert '"initialization": "known"' in calibration.design
    with pytest.raises(ValueError, match="sample size"):
        calibration_from_report(path, n=101)
    copied = tmp_path / "report.json"
    copied.write_text(json.dumps(report))
    (tmp_path / report["null_score_bank_file"]).write_bytes(b"corrupt calibration")
    with pytest.raises(ValueError, match="integrity"):
        calibration_from_report(copied, n=100)
