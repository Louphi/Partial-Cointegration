"""Data parsing and accounting regressions for the long daily screen."""

import json
import runpy

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def screen():
    return runpy.run_path("experiments/us_daily_screen.py", run_name="screen_test")


def test_vendor_prices_and_non_session_dividend_date(screen, tmp_path):
    def timestamp(date):
        return int(pd.Timestamp(date, tz="UTC").timestamp())

    record = {
        "meta": {"symbol": "X", "currency": "USD"},
        "timestamp": [
            timestamp(d)
            for d in ["2020-01-02 14:30", "2020-01-03 14:30", "2020-01-06 14:30"]
        ],
        "indicators": {"quote": [{"close": [10.0, 11.0, 12.0]}]},
        "events": {
            "dividends": {
                "event": {"date": timestamp("2020-01-04 14:30"), "amount": 0.2}
            }
        },
    }
    path = tmp_path / "X.json"
    path.write_text(json.dumps({"chart": {"result": [record]}}))
    prices, dividends, info = screen["parse_history"](path, "X", "2020-01-07")
    assert len(prices) == 3
    assert dividends.loc["2020-01-06"] == 0.2
    assert info["shifted_dividends"] == [
        {"ex_date": "2020-01-04", "accrual_date": "2020-01-06"}
    ]
    record["meta"]["currency"] = "CAD"
    path.write_text(json.dumps({"chart": {"result": [record]}}))
    with pytest.raises(ValueError, match="currency"):
        screen["parse_history"](path, "X", "2020-01-07")


def test_initial_drawdown_includes_deposit(screen):
    equity = pd.Series([9000.0, 9500.0], index=pd.date_range("2020-01-01", periods=2))
    stats = screen["metrics"](equity, 10000)
    assert stats["max_drawdown"] == pytest.approx(0.1)
    assert stats["total_return"] == pytest.approx(-0.05)


def test_failed_fit_and_descriptive_screen_are_explicit(screen):
    failed = screen["fit_job"](("X/Y", "test", "2020-01-01", np.ones(2), np.ones(2), 0))
    assert failed["status"] == "fit_failed"
    config = {"rho_min": 0.5, "variance_min": 0.5}
    assert not screen["eligible"](failed, config)
    candidate = dict(status="ok", beta=1, rho=0.7, variance_fraction=0.6)
    assert screen["eligible"](candidate, config)
    candidate["beta"] = -1
    assert not screen["eligible"](candidate, config)


def test_forward_ledger_dividends_and_no_trade_baseline(screen):
    dates = pd.bdate_range("2020-10-01", "2021-01-15")
    prices = pd.DataFrame(
        {"X": 100.0, "Y": 100.0 + np.sin(np.arange(len(dates)))}, index=dates
    )
    evaluation = dates[dates >= "2021-01-01"]
    prices.loc[evaluation, "Y"] = [
        110.0,
        110.0,
        90.0,
        90.0,
        110.0,
        110.0,
        90.0,
        90.0,
        100.0,
        100.0,
        100.0,
    ]
    dividends = prices * 0
    dividends.loc[evaluation[2], "X"] = 0.5
    row = dict(
        pair="X/Y",
        beta=1,
        parameters=dict(beta=1, rho=0.6, sigma_m=1, sigma_r=0.2, alpha=0.0),
    )
    config = dict(
        formation_years=4,
        fees=0.001,
        slippage=0.001,
        annual_borrow=0.01,
        entry=1.0,
        exit=-0.5,
    )
    equity, fills, stats = screen["run_ledger"](
        prices,
        dividends,
        [row],
        pd.Timestamp("2021-01-01"),
        pd.Timestamp("2021-02-01"),
        10000.0,
        config,
    )
    assert stats["entries"] > 0
    assert stats["dividends"] > 0
    signal = fills.loc[fills.kind.isin(["entry", "exit"])]
    assert (signal.fill_time > signal.decision_time).all()
    cash, empty, summary = screen["run_ledger"](
        prices,
        dividends,
        [],
        pd.Timestamp("2021-01-01"),
        pd.Timestamp("2021-02-01"),
        10000.0,
        config,
    )
    assert (cash == 10000).all()
    assert empty.empty
    assert summary["entries"] == 0
    assert np.isfinite(equity).all()
