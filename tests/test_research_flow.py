import numpy as np
import pandas as pd
import pytest
from partial_cointegration import Parameters, fit_pair
from partial_cointegration.data import members_at, split_prices
from partial_cointegration.signals import score_components, decisions
from partial_cointegration.simulation import simulate_pair, summary
from partial_cointegration.execution import (
    backtest_pairs,
    ExecutionConfig,
    replay_vectorbt,
)
from partial_cointegration.inference import (
    Calibration,
    rejection_summary,
    wilson_interval,
)


def fixture(prices=None, entry=0, exit=2):
    dates = pd.date_range("2023-01-01", periods=5)
    prices = (
        prices
        if prices is not None
        else pd.DataFrame(
            {"A": [10.0] * 5, "B": [20.0] * 5, "unused": [30.0] * 5}, index=dates
        )
    )
    opens = pd.DataFrame(False, index=dates, columns=["A/B"])
    closes = opens.copy()
    if entry is not None:
        opens.iloc[entry, 0] = True
    if exit is not None:
        closes.iloc[exit, 0] = True
    return prices, opens, closes


def test_next_close_constant_round_trip_and_actual_engine_oracle():
    p, e, x = fixture()
    config = ExecutionConfig()
    out = backtest_pairs(p, {"A/B": ("A", "B", 1)}, e, x, config=config)
    assert set(out.orders.fill_time) == {p.index[1], p.index[3]}
    assert (out.orders.fill_time > out.orders.decision_time).all()
    unit = 1 / (1 + config.fees + config.slippage) / 30
    # Two adverse spread-crossings plus four notional fees, independently computed.
    expected = (
        1
        - unit * (2 * 10 * config.slippage + 2 * 20 * config.slippage)
        - unit * config.fees * (2 * 10 + 2 * 20)
    )
    assert out.equity.iloc[-1] == pytest.approx(expected, abs=1e-10)
    assert list(out.holdings.columns) == list(p.columns)
    pf = replay_vectorbt(out, p)
    np.testing.assert_allclose(pf.value(), out.equity, atol=1e-10)
    np.testing.assert_allclose(pf.cash(), out.cash, atol=1e-10)
    assert out.holdings.iloc[-1].eq(0).all()


def test_declared_minute_borrow_frequency_matches_hand_worked_charge():
    prices, entries, exits = fixture()
    dates = pd.date_range(
        "2023-01-03 09:31", periods=5, freq="min", tz="America/New_York"
    )
    prices.index = entries.index = exits.index = dates
    for frequency in [252, 252 * 390]:
        result = backtest_pairs(
            prices,
            {"A/B": ("A", "B", 1)},
            entries,
            exits,
            config=ExecutionConfig(
                fees=0,
                slippage=0,
                annual_borrow=0.1,
                borrow_observations_per_year=frequency,
            ),
        )
        # One unit /30: short value 20/30, held over two charged intervals.
        assert result.equity.iloc[-1] == pytest.approx(
            1 - 2 * (20 / 30) * 0.1 / frequency, abs=1e-12
        )
    for invalid in [0, -1, np.inf]:
        with pytest.raises(ValueError):
            ExecutionConfig(borrow_observations_per_year=invalid)


def test_known_price_move_shared_capital_and_final_signal():
    p, e, x = fixture()
    p.loc[p.index[2] :, "A"] = 12.0
    p.loc[p.index[2] :, "B"] = 18.0
    out = backtest_pairs(
        p, {"A/B": ("A", "B", 1)}, e, x, config=ExecutionConfig(fees=0, slippage=0)
    )
    assert out.equity.iloc[-1] == pytest.approx(1 + 4 / 30)
    p, e, x = fixture(entry=4, exit=None)
    last = backtest_pairs(p, {"A/B": ("A", "B", 1)}, e, x)
    assert last.orders.empty
    assert "no later executable quote" in last.rejections.reason.tolist()
    p, e, x = fixture(entry=0, exit=None)
    closed = backtest_pairs(p, {"A/B": ("A", "B", 1)}, e, x)
    assert "scheduled_liquidation" in closed.orders.kind.tolist()


def test_missing_prices_margin_and_negative_beta():
    p, e, x = fixture()
    p.iloc[1, p.columns.get_loc("B")] = np.nan
    out = backtest_pairs(p, {"A/B": ("A", "B", 1)}, e, x)
    assert out.orders.fill_time.min() == p.index[2]
    with pytest.raises(ValueError, match="positive beta"):
        backtest_pairs(p, {"A/B": ("A", "B", -1)}, e, x)
    p, e, x = fixture()
    p.loc[p.index[1] :, ["A", "B"]] *= 2
    out = backtest_pairs(p, {"A/B": ("A", "B", 1)}, e, x)
    assert (
        out.orders.empty
        and "insufficient gross margin" in out.rejections.reason.tolist()
    )
    p, e, x = fixture(exit=None)
    p.iloc[-1, p.columns.get_loc("B")] = np.nan
    with pytest.raises(ValueError, match="settlement"):
        backtest_pairs(p, {"A/B": ("A", "B", 1)}, e, x)
    settled = backtest_pairs(p, {"A/B": ("A", "B", 1)}, e, x, settlements={"B": 0.0})
    assert "settlement" in settled.orders.kind.tolist()
    assert settled.holdings.iloc[-1].eq(0).all()


def test_one_budget_across_multiple_pairs_and_borrow_cost():
    p, e, x = fixture()
    p["C"] = 10.0
    p["D"] = 20.0
    e["C/D"] = e["A/B"]
    x["C/D"] = x["A/B"]
    config = ExecutionConfig(fees=0, slippage=0)
    out = backtest_pairs(
        p, {"A/B": ("A", "B", 1), "C/D": ("C", "D", 1)}, e, x, config=config
    )
    assert out.equity.iloc[0] == 1
    assert out.equity.iloc[-1] == pytest.approx(1)
    assert np.max((out.holdings.abs() * p).sum(axis=1)) <= 1 + 1e-12
    borrowed = backtest_pairs(
        p,
        {"A/B": ("A", "B", 1), "C/D": ("C", "D", 1)},
        e,
        x,
        config=ExecutionConfig(fees=0, slippage=0, annual_borrow=0.1),
    )
    assert borrowed.equity.iloc[-1] < 1
    with pytest.raises(ValueError, match="weights"):
        backtest_pairs(
            p,
            {"A/B": ("A", "B", 1), "C/D": ("C", "D", 1)},
            e,
            x,
            weights={"A/B": 1, "C/D": 1},
        )


def test_point_in_time_intervals_alignment_and_immutable_generation():
    intervals = pd.DataFrame(
        {
            "from": ["2023-06-01", "2020-01-01"],
            "thru": [None, None],
            "co_tic": ["FUTURE", "OLD"],
        }
    )
    original = intervals.copy()
    assert members_at(intervals, "2022-12-31") == ["OLD"]
    pd.testing.assert_frame_equal(original, intervals)
    pair = simulate_pair(100, Parameters(1, 0.6, 1, 1), seed=4)
    np.random.seed(123)
    np.random.normal(size=100)
    again = simulate_pair(100, Parameters(1, 0.6, 1, 1), seed=4)
    np.testing.assert_array_equal(pair.x, again.x)
    np.testing.assert_array_equal(pair.m, again.m)
    dates = pd.date_range("2020-01-01", periods=100)
    w = pd.Series(pair.m + pair.r, index=dates)
    out = score_components(
        w.iloc[:60], w.iloc[60:], Parameters(1, 0.6, 1, 1), variant="rolling"
    )
    changed = w.copy()
    changed.iloc[-1] = 999
    perturb = score_components(
        changed.iloc[:60],
        changed.iloc[60:],
        Parameters(1, 0.6, 1, 1),
        variant="rolling",
    )
    np.testing.assert_allclose(
        out["testing_score"].iloc[:-1],
        perturb["testing_score"].iloc[:-1],
        equal_nan=True,
    )
    opens, closes = decisions(out["testing_score"].rename("pair"))
    assert not (opens & closes).any().any()


def test_failure_aware_inference_and_summary():
    stats = summary(np.array([[1.0, 2.0], [np.nan, 2.0]]), np.array([1.0, 2.0]))
    assert stats["failed"] == 1 and stats["mse"] == [0, 0]
    rep = rejection_summary(np.array([-5.0, np.nan]), -3.0)
    assert rep["unconditional_lower"] == 0.5 and rep["unconditional_upper"] == 1
    cal = Calibration(np.linspace(-10, 0, 100), 100, "exact", 1, "geometric", {})
    assert cal.critical_value(0.05) < 0
    with pytest.raises(ValueError):
        cal.validate(n=1000, mode="exact", design="geometric", estimator_options={})
    broken = Calibration(np.r_[np.arange(99), np.nan], 100, "exact", 1, "geometric", {})
    with pytest.raises(ValueError):
        broken.critical_value()
    assert wilson_interval(5, 100)[0] < 0.05 < wilson_interval(5, 100)[1]


def test_simulation_summary_idempotency_and_adapter_contracts():
    import xarray as xr
    from ResultReplication import f_compute_mse, f_compute_powers_lrt
    from PartialCointegration import partial_cointegration
    from PCIPipeline import Computation, Definition

    values = np.array([[[1, 0.9, 0.5, 1, 0]], [[1, 0.9, 1.5, 1, 0]]])
    ds = xr.Dataset(
        {
            "sample": xr.DataArray(
                values,
                dims=["var_param", "simulation", "estimates"],
                coords={
                    "var_param": [0.5, 1.5],
                    "simulation": [0],
                    "estimates": ["beta", "rho", "sigma_m", "sigma_r", "lrt"],
                },
            )
        }
    )
    params = [0, 1, 0.9, None, 1, 0.0236]
    first = f_compute_mse(ds, params)
    second = f_compute_mse(ds, params)
    xr.testing.assert_equal(first, second)
    assert params[3] is None
    np.testing.assert_allclose(first["sample"], 0)
    with pytest.raises(ValueError, match="independently"):
        f_compute_powers_lrt(ds, [100], [0.05])
    pair = simulate_pair(50, Parameters(1, 0.6, 1, 1), seed=8)
    assert len(Computation(pair.x, pair.y).fit_mle(pair.m + pair.r)) == 3
    assert np.asarray(
        partial_cointegration(pair.x, pair.y).fit_mle(pair.m + pair.r)
    ).shape == (2, 3)
    pairs = Definition.pairDatabase(["A", "B", "C"])
    prices = pd.DataFrame(
        {
            "A": pair.x,
            "B": pair.y,
            "C": pair.y + np.random.default_rng(5).normal(size=len(pair.y)),
        }
    )
    a, b = Computation.f_compute_estimate(pairs["pair"].iloc[1:], prices, pairs)
    assert a.shape == (2, 3) and np.isfinite(b).all()
