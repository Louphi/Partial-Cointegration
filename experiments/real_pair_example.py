"""One fixed-configuration public-price illustration, not a certified universe backtest.

GOOG/GOOGL; formation 2019-2022; evaluation Jan-Jun 2023. No pair or
threshold search. Raw vendor responses remain in ignored local scratch space.
"""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from partial_cointegration import fit_pair, FitOptions
from partial_cointegration.data import validate_prices
from partial_cointegration.execution import backtest_pairs, ExecutionConfig
from partial_cointegration.io import package_fingerprint, estimator_fingerprint
from partial_cointegration.signals import score_components, decisions


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cache = root / ".test-deps/real-pair-goog-googl"
    cache.mkdir(parents=True, exist_ok=True)
    prices_by_symbol = {}
    sources = {}
    for symbol in ["GOOG", "GOOGL"]:
        query = urlencode(
            {
                "period1": 1546300800,
                "period2": 1688169600,
                "interval": "1d",
                "events": "div,splits",
                "includeAdjustedClose": "true",
            }
        )
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?{query}"
        raw = cache / f"{symbol}.json"
        if not raw.exists():
            with urlopen(
                Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=30
            ) as response:
                raw.write_bytes(response.read())
        payload = json.loads(raw.read_bytes())
        if payload["chart"]["error"]:
            raise ValueError(payload["chart"]["error"])
        record = payload["chart"]["result"][0]
        if record["meta"]["symbol"] != symbol or record["meta"]["currency"] != "USD":
            raise ValueError("Unexpected vendor security/currency identity")
        dates = (
            pd.to_datetime(record["timestamp"], unit="s", utc=True)
            .tz_convert("America/New_York")
            .normalize()
            .tz_localize(None)
        )
        series = pd.Series(
            record["indicators"]["quote"][0]["close"], index=dates, name=symbol
        )
        prices_by_symbol[symbol] = series
        events = record.get("events", {})
        for kind, values in events.items():
            for event in values.values():
                date = (
                    pd.Timestamp(event["date"], unit="s", tz="UTC")
                    .tz_convert("America/New_York")
                    .date()
                )
                if (
                    pd.Timestamp("2023-01-01").date()
                    <= date
                    < pd.Timestamp("2023-07-01").date()
                ):
                    raise ValueError(
                        f"Evaluation contains {kind}; explicit cash-flow handling required"
                    )
        sources[symbol] = {
            "url": url,
            "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
            "events": events,
            "observations": len(series),
        }
    if not prices_by_symbol["GOOG"].index.equals(prices_by_symbol["GOOGL"].index):
        raise ValueError("Vendor trading dates differ; do not silently intersect")
    prices = validate_prices(pd.concat(prices_by_symbol.values(), axis=1))
    training = prices.loc["2019-01-01":"2022-12-31"]
    testing = prices.loc["2023-01-01":"2023-06-30"]
    if training.isna().any().any() or testing.isna().any().any():
        raise ValueError(
            "Missing vendor observations; inspect before claiming a complete example"
        )
    options = FitOptions(seed=20261025)
    fit = fit_pair(training.GOOG, training.GOOGL, options=options)
    parameters = fit.parameters
    if parameters.beta <= 0:
        raise ValueError("Trading implementation requires positive beta")
    components = score_components(
        training.GOOGL - parameters.beta * training.GOOG,
        testing.GOOGL - parameters.beta * testing.GOOG,
        parameters,
        variant="formation",
        window=30,
    )
    score = components["testing_score"].rename("GOOG/GOOGL")
    entries, exits = decisions(score, entry=1.0, exit=-0.5)
    config = ExecutionConfig(
        capital=10000, fees=0.001, slippage=0.001, annual_borrow=0.01
    )
    ledger = backtest_pairs(
        testing,
        {"GOOG/GOOGL": ("GOOG", "GOOGL", parameters.beta)},
        entries,
        exits,
        config=config,
    )
    signal_orders = ledger.orders.loc[ledger.orders.kind.isin(["entry", "exit"])]
    if not (signal_orders.fill_time > signal_orders.decision_time).all():
        raise AssertionError("A decision executed without a later quote")
    if not np.isfinite(ledger.equity).all() or not np.allclose(
        ledger.holdings.iloc[-1], 0
    ):
        raise AssertionError("Nonfinite equity or incomplete liquidation")
    # Independent daily cash reconstruction from executed orders and vendor quotes.
    # No engine-generated cash or holdings are used in this accounting oracle.
    changes = pd.DataFrame(0.0, index=testing.index, columns=testing.columns)
    cash_changes = pd.Series(0.0, index=testing.index)
    gross_price_pnl = 0.0
    slippage_cost = 0.0
    for order in ledger.orders.itertuples():
        mid = float(testing.loc[order.fill_time, order.security])
        expected_fill = mid * (1 + config.slippage * np.sign(order.units))
        np.testing.assert_allclose(order.price, expected_fill, atol=1e-10)
        np.testing.assert_allclose(
            order.fees, abs(order.units) * order.price * config.fees, atol=1e-10
        )
        changes.loc[order.fill_time, order.security] += order.units
        cash_changes.loc[order.fill_time] -= order.units * order.price + order.fees
        gross_price_pnl -= order.units * mid
        slippage_cost += order.units * (order.price - mid)
    positions = changes.cumsum()
    borrow = (
        (-positions.shift(1).fillna(0).clip(upper=0) * testing).sum(axis=1)
        * config.annual_borrow
        / 252
    )
    independent_equity = (
        config.capital
        + (cash_changes - borrow).cumsum()
        + (positions * testing).sum(axis=1)
    )
    np.testing.assert_allclose(independent_equity, ledger.equity, atol=1e-8, rtol=0)
    out = cache / "output"
    out.mkdir(exist_ok=True)
    prices.to_csv(cache / "prices.csv", index_label="date")
    ledger.equity.to_csv(out / "equity.csv")
    ledger.orders.to_csv(out / "orders.csv", index=False)
    ledger.rejections.to_csv(out / "rejections.csv", index=False)
    pd.concat([components["testing_m"].rename("temporary"), score], axis=1).to_csv(
        out / "signals.csv"
    )
    drawdown = ledger.equity / ledger.equity.cummax() - 1
    result = {
        "data_kind": "market",
        "scope": "Exploratory fixed-pair illustration, not full-universe certification",
        "pair_chosen_before_results": ["GOOG", "GOOGL"],
        "pair_rationale": "Same issuer, different share classes; no performance-based pair search",
        "training": {
            "start": str(training.index[0].date()),
            "end": str(training.index[-1].date()),
            "n": len(training),
        },
        "evaluation": {
            "start": str(testing.index[0].date()),
            "end": str(testing.index[-1].date()),
            "n": len(testing),
        },
        "normalization": "formation",
        "entry": 1,
        "exit": -0.5,
        "parameters": asdict(parameters),
        "fit_options": asdict(options),
        "selected_model": fit.selected,
        "log_ratio": fit.log_ratio,
        "execution": asdict(config),
        "stats": ledger.stats().to_dict(),
        "max_drawdown": float(drawdown.min()),
        "completed_pair_entries": int((ledger.orders.kind == "entry").sum() // 2),
        "completed_pair_exits": int(
            ledger.orders.kind.isin(
                ["exit", "scheduled_liquidation", "settlement"]
            ).sum()
            // 2
        ),
        "accounting_oracle": "Daily cash/equity reconstructed independently from executed orders and vendor closes; agreement within 1e-8 USD",
        "executed_trade_attribution_usd": {
            "price_pnl_before_costs": gross_price_pnl,
            "fees": float(ledger.orders.fees.sum()),
            "assumed_slippage": slippage_cost,
            "assumed_borrow": float(borrow.sum()),
            "net_pnl": float(ledger.equity.iloc[-1] - config.capital),
        },
        "unexecuted_requests_by_reason": ledger.rejections.reason.value_counts().to_dict(),
        "cash_benchmark_return": 0,
        "source": "Yahoo Finance daily public chart responses, saved locally with digests",
        "price_basis": "Vendor Close field, split-adjusted historical series; not dividend-adjusted Adj Close. No evaluation-period corporate actions in vendor responses.",
        "training_split": "Both classes had a 20:1 split in July 2022; vendor historical closes share the post-split basis",
        "sources": sources,
        "estimator_fingerprint": estimator_fingerprint(),
        "package_fingerprint": package_fingerprint(),
        "script_sha256": hashlib.sha256(
            Path(__file__).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest(),
        "report_created_utc": datetime.now(timezone.utc).isoformat(),
        "vendor_cache_policy": "Reuse local raw JSON when present; source digests identify the exact downloaded snapshot",
        "qualified_historical_performance_verified": False,
        "limitations": [
            "One retrospectively chosen pair/window; not a preregistration before the historical period or an untouched future test",
            "Current vendor history, not a point-in-time vendor archive; no independent exchange-price reconciliation",
            "Borrow availability and actual borrow fees unknown; 1% annual rate is an assumption",
            "Fractional-share, daily-close ledger; no intraday margin calls, financing or market impact beyond assumed slippage",
            "No calibrated significance claim, universe selection or threshold tuning",
        ],
    }
    target = root / "docs/results/real-pair-goog-googl.json"
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
