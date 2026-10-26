"""Fixed-pair daily walk-forward illustration, 2019-2023; no threshold search."""

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
from partial_cointegration.io import package_fingerprint
from partial_cointegration.signals import score_components, decisions


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cache = root / ".test-deps/longer-real-pair"
    cache.mkdir(parents=True, exist_ok=True)
    series = []
    sources = {}
    for symbol in ["GOOG", "GOOGL"]:
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?"
            + urlencode(
                {
                    "interval": "1d",
                    "events": "div,splits",
                    "includePrePost": "false",
                    "period1": 1420070400,
                    "period2": 1704067200,
                }
            )
        )
        path = cache / f"{symbol}-1d.json"
        if not path.exists():
            with urlopen(
                Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=30
            ) as response:
                path.write_bytes(response.read())
        payload = json.loads(path.read_bytes())
        if payload["chart"]["error"]:
            raise ValueError(payload["chart"]["error"])
        record = payload["chart"]["result"][0]
        if record["meta"]["symbol"] != symbol or record["meta"]["currency"] != "USD":
            raise ValueError("Unexpected security or currency")
        index = (
            pd.to_datetime(record["timestamp"], unit="s", utc=True)
            .tz_convert("America/New_York")
            .normalize()
            .tz_localize(None)
        )
        series.append(
            pd.Series(
                record["indicators"]["quote"][0]["close"], index=index, name=symbol
            )
        )
        if record.get("events", {}).get("dividends"):
            raise ValueError(
                "This illustration requires explicit dividend handling before extending into dividend-paying periods"
            )
        sources[symbol] = {
            "url": url,
            "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "events": record.get("events", {}),
        }
    if not series[0].index.equals(series[1].index):
        raise ValueError("Do not silently intersect vendor calendars")
    prices = validate_prices(pd.concat(series, axis=1))
    if prices.isna().any().any():
        raise ValueError("Inspect missing vendor observations before evaluating")
    capital = 10000.0
    curves = []
    cohorts = []
    orders = []
    rejected = []
    options = FitOptions(seed=20261025)
    for start in pd.date_range("2019-01-01", "2023-07-01", freq="6MS"):
        training = prices.loc[
            (prices.index >= start - pd.DateOffset(years=4)) & (prices.index < start)
        ]
        testing = prices.loc[
            (prices.index >= start) & (prices.index < start + pd.DateOffset(months=6))
        ]
        fit = fit_pair(training.GOOG, training.GOOGL, options=options)
        p = fit.parameters
        label = str(start.date())
        reason = None
        entries = pd.DataFrame(False, index=testing.index, columns=["GOOG/GOOGL"])
        exits = entries.copy()
        if p.beta <= 0 or p.sigma_m == 0:
            reason = (
                "Unsupported hedge sign or no transient component; cash-only cohort"
            )
        else:
            components = score_components(
                training.GOOGL - p.beta * training.GOOG,
                testing.GOOGL - p.beta * testing.GOOG,
                p,
                variant="formation",
                window=30,
            )
            entries, exits = decisions(
                components["testing_score"].rename("GOOG/GOOGL"), 1, -0.5
            )
        config = ExecutionConfig(
            capital=capital, fees=0.001, slippage=0.001, annual_borrow=0.01
        )
        ledger = backtest_pairs(
            testing,
            {"GOOG/GOOGL": ("GOOG", "GOOGL", p.beta if p.beta > 0 else 1)},
            entries,
            exits,
            config=config,
        )
        if not np.allclose(ledger.holdings.iloc[-1], 0):
            raise AssertionError("Cohort failed to liquidate")
        signal_orders = ledger.orders.loc[ledger.orders.kind.isin(["entry", "exit"])]
        if not (signal_orders.fill_time > signal_orders.decision_time).all():
            raise AssertionError("Signal executed without a later observation")
        # Independently reconstruct daily equity from executed orders and prices.
        changes = pd.DataFrame(0.0, index=testing.index, columns=testing.columns)
        cash_changes = pd.Series(0.0, index=testing.index)
        for order in ledger.orders.itertuples():
            mid = testing.loc[order.fill_time, order.security]
            np.testing.assert_allclose(
                order.price,
                mid * (1 + config.slippage * np.sign(order.units)),
                atol=1e-10,
            )
            np.testing.assert_allclose(
                order.fees, abs(order.units) * order.price * config.fees, atol=1e-10
            )
            changes.loc[order.fill_time, order.security] += order.units
            cash_changes.loc[order.fill_time] -= order.units * order.price + order.fees
        positions = changes.cumsum()
        borrow = (
            (-positions.shift(1).fillna(0).clip(upper=0) * testing).sum(axis=1)
            * config.annual_borrow
            / 252
        )
        independent = (
            capital
            + (cash_changes - borrow).cumsum()
            + (positions * testing).sum(axis=1)
        )
        np.testing.assert_allclose(independent, ledger.equity, atol=1e-8, rtol=0)
        n_entries = int((ledger.orders.kind == "entry").sum() // 2)
        cohorts.append(
            {
                "start": label,
                "training_start": str(training.index[0].date()),
                "training_end": str(training.index[-1].date()),
                "training_n": len(training),
                "evaluation_n": len(testing),
                "parameters": asdict(p),
                "selected_model": fit.selected,
                "return": float(ledger.equity.iloc[-1] / capital - 1),
                "pair_entries": n_entries,
                "rejected_requests": len(ledger.rejections),
                "cash_only_reason": reason,
            }
        )
        curves.append(ledger.equity)
        orders.append(ledger.orders.assign(cohort=label))
        rejected.append(ledger.rejections.assign(cohort=label))
        capital = float(ledger.equity.iloc[-1])
        if capital <= 0:
            raise RuntimeError(
                "Capital exhausted; do not silently restart with fresh cash"
            )
        print(
            f"{label}: {n_entries} entries, {cohorts[-1]['return']:.3%}, capital ${capital:.2f}",
            flush=True,
        )
    equity = pd.concat(curves)
    nonempty_orders = [frame for frame in orders if not frame.empty]
    nonempty_rejected = [frame for frame in rejected if not frame.empty]
    all_orders = (
        pd.concat(nonempty_orders, ignore_index=True)
        if nonempty_orders
        else orders[0].copy()
    )
    all_rejected = (
        pd.concat(nonempty_rejected, ignore_index=True)
        if nonempty_rejected
        else rejected[0].copy()
    )
    if not equity.index.is_unique or not equity.index.is_monotonic_increasing:
        raise AssertionError("Overlapping or unordered evaluation curves")
    yearly = {}
    previous = 10000.0
    for year, curve in equity.groupby(equity.index.year):
        yearly[str(year)] = float(curve.iloc[-1] / previous - 1)
        previous = float(curve.iloc[-1])
    out = cache / "output"
    out.mkdir(exist_ok=True)
    equity.to_csv(out / "equity.csv")
    all_orders.to_csv(out / "orders.csv", index=False)
    all_rejected.to_csv(out / "rejections.csv", index=False)
    result = {
        "scope": "Retrospective fixed-pair daily illustration, not certified universe-wide performance",
        "pair": ["GOOG", "GOOGL"],
        "evaluation_start": str(equity.index[0].date()),
        "evaluation_end": str(equity.index[-1].date()),
        "evaluation_n": len(equity),
        "formation": "Preceding four years, refitted every six months",
        "normalization": "formation",
        "entry": 1,
        "exit": -0.5,
        "initial_capital": 10000,
        "final_capital": capital,
        "total_return": capital / 10000 - 1,
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
        "completed_round_trips": int((all_orders.kind == "entry").sum() // 2),
        "rejected_requests": len(all_rejected),
        "fees_paid": float(all_orders.fees.sum()),
        "yearly_returns": yearly,
        "cohorts": cohorts,
        "execution": asdict(
            ExecutionConfig(
                capital=10000, fees=0.001, slippage=0.001, annual_borrow=0.01
            )
        ),
        "accounting": "One capital account rolled forward, flat at each six-month boundary; daily cash/equity independently reconciled within $1e-8",
        "price_basis": "Vendor split-adjusted Close, not dividend-adjusted Adj Close. Share quantities are split-adjusted equivalents before the 2022 split; no dividends in this selected period.",
        "sources": sources,
        "package_fingerprint": package_fingerprint(),
        "script_sha256": hashlib.sha256(
            Path(__file__).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "Same issuer's share classes; no broad pair-universe selection",
            "Vendor history is not point-in-time audited",
            "Borrow availability/actual costs unverified; 1% annual charge assumed",
            "Period selected retrospectively before this run; no parameter or threshold search",
            "No untouched future test or calibrated profitability significance claim",
        ],
    }
    (root / "docs/results/longer-pair-goog-googl.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in [
                    "initial_capital",
                    "final_capital",
                    "total_return",
                    "completed_round_trips",
                    "rejected_requests",
                    "yearly_returns",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
