"""Five-session public-data prototype: three formation sessions, two evaluation.

Minute closes are timestamped at bar END. One-minute borrow uses 252*390
observations/year and positions are forcibly flat at every session close.
This is a session-compressed filter illustration, not an overnight state model
or a quote-level executable-price validation. No parameter/threshold search.
"""

from dataclasses import asdict
import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair", choices=["goog-googl", "pep-ko"], default="goog-googl"
    )
    args = parser.parse_args()
    x, y = ("GOOG", "GOOGL") if args.pair == "goog-googl" else ("PEP", "KO")
    pair_id = f"{x}/{y}"
    root = Path(__file__).resolve().parents[1]
    cache = (
        root
        / ".test-deps"
        / ("longer-real-pair" if args.pair == "goog-googl" else args.pair)
    )
    cache.mkdir(parents=True, exist_ok=True)
    items = []
    sources = {}
    for symbol in [x, y]:
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?"
            + urlencode(
                {
                    "interval": "1m",
                    "events": "div,splits",
                    "includePrePost": "false",
                    "range": "5d",
                }
            )
        )
        path = cache / f"{symbol}-1m.json"
        if not path.exists():
            with urlopen(
                Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=30
            ) as response:
                path.write_bytes(response.read())
        payload = json.loads(path.read_bytes())
        if payload["chart"]["error"]:
            raise ValueError(payload["chart"]["error"])
        record = payload["chart"]["result"][0]
        if record.get("events"):
            raise ValueError(
                "Prototype requires explicit handling of corporate-action events"
            )
        if record["meta"]["symbol"] != symbol or record["meta"]["currency"] != "USD":
            raise ValueError("Unexpected identity/currency")
        index = pd.to_datetime(record["timestamp"], unit="s", utc=True).tz_convert(
            "America/New_York"
        )
        values = pd.Series(
            record["indicators"]["quote"][0]["close"], index=index, name=symbol
        )
        regular = values.between_time("09:30", "16:00", inclusive="left").copy()
        dropped = len(values) - len(regular)
        regular.index += pd.Timedelta(minutes=1)
        items.append(regular)
        sources[symbol] = {
            "url": url,
            "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "vendor_bars": len(values),
            "excluded_outside_regular_session": dropped,
        }
    if not items[0].index.equals(items[1].index):
        raise ValueError("Unsynchronized data; do not silently intersect or fill")
    prices = validate_prices(pd.concat(items, axis=1))
    if prices.isna().any().any():
        raise ValueError("Missing observed prices; stop rather than invent fills")
    dates = sorted(set(prices.index.date))
    if len(dates) != 5:
        raise ValueError("Need exactly five complete sessions")
    for date in dates:
        session = prices.loc[prices.index.date == date]
        expected = pd.date_range(
            f"{date} 09:31", periods=390, freq="min", tz="America/New_York"
        )
        if not session.index.equals(expected):
            raise ValueError(
                "Incomplete/early-close session; fixed 390-minute convention not applicable"
            )
    training = prices.loc[prices.index.date < dates[3]]
    testing = prices.loc[prices.index.date >= dates[3]]
    fit = fit_pair(training[x], training[y], options=FitOptions(seed=20261025))
    p = fit.parameters
    score = None
    if p.beta > 0 and p.sigma_m > 0:
        components = score_components(
            training[y] - p.beta * training[x],
            testing[y] - p.beta * testing[x],
            p,
            variant="formation",
            window=30,
        )
        score = components["testing_score"].rename(pair_id)
    capital = 10000.0
    curves = []
    orders = []
    rejections = []
    sessions = []
    for date in dates[3:]:
        day = testing.loc[testing.index.date == date]
        if score is None:
            entries = pd.DataFrame(False, index=day.index, columns=[pair_id])
            exits = entries.copy()
        else:
            entries, exits = decisions(score.loc[day.index], 1, -0.5)
        config = ExecutionConfig(
            capital=capital,
            fees=0.001,
            slippage=0.001,
            annual_borrow=0.01,
            borrow_observations_per_year=252 * 390,
        )
        ledger = backtest_pairs(
            day,
            {pair_id: (x, y, p.beta if p.beta > 0 else 1)},
            entries,
            exits,
            config=config,
        )
        if not np.allclose(ledger.holdings.iloc[-1], 0):
            raise AssertionError("No overnight positions are permitted")
        signal_orders = ledger.orders.loc[ledger.orders.kind.isin(["entry", "exit"])]
        if not (signal_orders.fill_time > signal_orders.decision_time).all():
            raise AssertionError("Same-bar or earlier execution")
        # Independent cash oracle; scaled borrow applies to held intraday intervals only.
        changes = pd.DataFrame(0.0, index=day.index, columns=day.columns)
        cash_changes = pd.Series(0.0, index=day.index)
        for order in ledger.orders.itertuples():
            mid = day.loc[order.fill_time, order.security]
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
            (-positions.shift(1).fillna(0).clip(upper=0) * day).sum(axis=1)
            * 0.01
            / (252 * 390)
        )
        independent = (
            capital + (cash_changes - borrow).cumsum() + (positions * day).sum(axis=1)
        )
        np.testing.assert_allclose(independent, ledger.equity, atol=1e-8, rtol=0)
        count = int((ledger.orders.kind == "entry").sum() // 2)
        sessions.append(
            {
                "date": str(date),
                "completed_round_trips": count,
                "return": float(ledger.equity.iloc[-1] / capital - 1),
                "rejected_requests": len(ledger.rejections),
            }
        )
        curves.append(ledger.equity)
        orders.append(ledger.orders)
        rejections.append(ledger.rejections)
        capital = float(ledger.equity.iloc[-1])
        if capital <= 0:
            raise RuntimeError("Capital exhausted; no reset permitted")
    equity = pd.concat(curves)
    order_frames = [frame for frame in orders if not frame.empty]
    all_orders = (
        pd.concat(order_frames, ignore_index=True) if order_frames else orders[0]
    )
    rejection_frames = [frame for frame in rejections if not frame.empty]
    all_rejections = (
        pd.concat(rejection_frames, ignore_index=True)
        if rejection_frames
        else rejections[0]
    )
    out = cache / "minute-output"
    out.mkdir(exist_ok=True)
    prices.to_csv(out / "prices.csv", index_label="date")
    equity.to_csv(out / "equity.csv")
    all_orders.to_csv(out / "orders.csv", index=False)
    all_rejections.to_csv(out / "rejections.csv", index=False)
    result = {
        "scope": "Short minute-data mechanics prototype; not an intraday profitability certification",
        "pair": [x, y],
        "formation_sessions": [str(d) for d in dates[:3]],
        "evaluation_sessions": [str(d) for d in dates[3:]],
        "formation_bars": len(training),
        "evaluation_bars": len(testing),
        "parameters": asdict(p),
        "selected_model": fit.selected,
        "cash_only_reason": (
            "No fitted transient component or unsupported hedge sign"
            if score is None
            else None
        ),
        "initial_capital": 10000,
        "final_capital": capital,
        "total_return": capital / 10000 - 1,
        "completed_round_trips": sum(day["completed_round_trips"] for day in sessions),
        "rejected_requests": sum(len(frame) for frame in rejections),
        "rejected_requests_by_reason": all_rejections.reason.value_counts().to_dict(),
        "sessions": sessions,
        "fees_paid": float(all_orders.fees.sum()),
        "normalization": "formation",
        "entry": 1,
        "exit": -0.5,
        "fees_per_fill": 0.001,
        "slippage_per_fill": 0.001,
        "annual_borrow": 0.01,
        "borrow_observations_per_year": 252 * 390,
        "overnight_positions": False,
        "bar_timestamps": "Vendor minute-start timestamps shifted to completed-bar end; fills occur at a later completed bar",
        "accounting": "Session-by-session flat close, rolled capital; independently reconciled equity to $1e-8",
        "sources": sources,
        "package_fingerprint": package_fingerprint(),
        "script_sha256": hashlib.sha256(
            Path(__file__).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "Only two evaluation sessions; not evidence of stable performance",
            "Session-compressed observation-clock filtering: training/test overnight state transitions are not separately modeled",
            "Minute closing trades are not synchronized bid/ask executable quotes; spread/latency/partial fills unverified",
            "Cost/borrow assumptions are illustrative, not verified broker terms",
            "Each session must have all 390 regular minutes; early closes and missing minutes rejected",
            "No reuse of daily parameters or daily critical values; no significance claim",
        ],
    }
    (root / f"docs/results/minute-pair-{args.pair}.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
