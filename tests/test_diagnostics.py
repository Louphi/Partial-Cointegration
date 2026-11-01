"""Diagnostic accounting checks and real Streamlit interaction regressions."""

import numpy as np
import pandas as pd
import pytest
from dataclasses import replace
from unittest.mock import patch

from diagnostics import pair_diagnostics, timeline_chart
from partial_cointegration import Parameters
from partial_cointegration.execution import ExecutionConfig, backtest_pairs


def test_costs_and_exposure_reconcile_to_independent_constant_price_ledger():
    dates = pd.date_range("2023-01-01", periods=7)
    prices = pd.DataFrame({"X": 10.0, "Y": 20.0}, index=dates)
    evaluation = prices.iloc[2:]
    entries = pd.DataFrame(
        {"pair": [True, False, False, False, False]}, index=evaluation.index
    )
    exits = pd.DataFrame(
        {"pair": [False, False, True, False, False]}, index=evaluation.index
    )
    config = ExecutionConfig(capital=10000, annual_borrow=0.05)
    ledger = backtest_pairs(
        evaluation, {"pair": ("X", "Y", 1)}, entries, exits, config=config
    )
    frame, fills = pair_diagnostics(
        prices,
        "X",
        "Y",
        Parameters(1, 0.6, 1, 1),
        pd.Series(1.0, index=evaluation.index),
        ledger,
    )
    costs = frame[["Fees", "Slippage cost", "Borrow cost"]].sum().sum()
    assert config.capital - ledger.equity.iloc[-1] == pytest.approx(costs)
    assert frame["Cumulative costs"].iloc[-1] == pytest.approx(costs)
    assert frame["Gross exposure"].iloc[0] == 0
    assert frame["Gross exposure"].iloc[1] > 0
    assert frame["Gross exposure"].iloc[-1] == 0
    assert frame["Drawdown %"].iloc[-1] == pytest.approx(-costs / config.capital * 100)
    np.testing.assert_allclose(
        frame.Spread, frame["Temporary M"] + frame["Permanent R"]
    )
    assert fills.fill_time.gt(fills.decision_time).all()
    chart = timeline_chart(
        frame, fills, entries["pair"], exits["pair"], 1, -0.5
    ).to_dict()
    assert len(chart["vconcat"]) == 6


def test_no_trade_and_scheduled_liquidation_diagnostics():
    dates = pd.date_range("2023-01-01", periods=5)
    prices = pd.DataFrame({"X": 10.0, "Y": 20.0}, index=dates)
    flags = pd.DataFrame(False, index=dates, columns=["pair"])
    for trade in [False, True]:
        entries = flags.copy()
        entries.iloc[0, 0] = trade
        ledger = backtest_pairs(prices, {"pair": ("X", "Y", 1)}, entries, flags)
        frame, fills = pair_diagnostics(
            prices,
            "X",
            "Y",
            Parameters(1, 0.6, 0, 1),
            pd.Series(0.0, index=dates),
            ledger,
        )
        assert np.isfinite(frame["Cumulative costs"]).all()
        assert frame["Gross exposure"].iloc[-1] == 0
        assert (fills.kind == "scheduled_liquidation").any() == trade
        timeline_chart(frame, fills, entries["pair"], flags["pair"], 1, -0.5).to_dict()


def test_app_diagnostics_persist_and_changed_inputs_hide_old_results():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file("PCI Framework/app.py", default_timeout=60).run()
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    assert {m.label for m in app.metric} >= {
        "Entry decisions",
        "Executed entries",
        "Maximum drawdown",
        "Fees",
    }
    assert len(app.get("download_button")) == 3
    app.run()
    assert len(app.metric) == 8  # A rerun/download retains the computed run.
    app.number_input[0].set_value(2.0).run()
    assert not app.metric
    assert any("Inputs changed" in item.value for item in app.info)
    app.button[0].click().run()
    assert not app.exception
    assert len(app.metric) == 8


def test_app_rejects_invalid_thresholds():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file("PCI Framework/app.py", default_timeout=60).run()
    app.number_input[0].set_value(-1.0)
    app.button[0].click().run()
    assert not app.exception
    assert any("Entry must exceed exit" in error.value for error in app.error)


def test_app_displays_pure_random_walk_as_cash_instead_of_scale_error():
    from streamlit.testing.v1 import AppTest
    from partial_cointegration import fit_pair
    from partial_cointegration.simulation import simulate_pair

    pair = simulate_pair(100, Parameters(1, 0.6, 1, 1), seed=42)
    fitted = replace(
        fit_pair(pair.x, pair.y), parameters=Parameters(1, 0, 0, 1), selected="rw"
    )
    app = AppTest.from_file("PCI Framework/app.py", default_timeout=60).run()
    with patch("partial_cointegration.fit_pair", return_value=fitted):
        app.button[0].click().run()
    assert not app.exception
    assert not app.error
    assert any("pure random walk" in item.value for item in app.info)
    assert next(m.value for m in app.metric if m.label == "Executed entries") == "0"
