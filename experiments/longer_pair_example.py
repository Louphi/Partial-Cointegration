"""Fixed-pair daily walk-forward illustrations; no threshold search."""

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
    parser.add_argument(
        "--extended", action="store_true", help="PEP/KO: evaluate 2000–2025"
    )
    args = parser.parse_args()
    if args.extended and args.pair != "pep-ko":
        parser.error("The extended corporate-action checks support PEP/KO only")
    suffix = "-extended" if args.extended else ""
    x, y = ("GOOG", "GOOGL") if args.pair == "goog-googl" else ("PEP", "KO")
    pair_id = f"{x}/{y}"
    root = Path(__file__).resolve().parents[1]
    cache = (
        root
        / ".test-deps"
        / (("longer-real-pair" if args.pair == "goog-googl" else args.pair) + suffix)
    )
    cache.mkdir(parents=True, exist_ok=True)
    series = []
    sources = {}
    for symbol in [x, y]:
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?"
            + urlencode(
                {
                    "interval": "1d",
                    "events": "div,splits",
                    "includePrePost": "false",
                    "period1": 820454400 if args.extended else 1420070400,
                    "period2": 1767225600 if args.extended else 1704067200,
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
        if args.extended:
            # Yahoo dividends and Close are already in split-adjusted share units.
            # Check the known 2012 KO transition against issuer dividend figures;
            # dividing these dividends by the split ratio again would be wrong.
            events = record.get("events", {})
            splits = events.get("splits", {}).values()
            expected_splits = {1996: 2.0} if symbol == "PEP" else {1996: 2.0, 2012: 2.0}
            actual_splits = {
                datetime.fromtimestamp(e["date"], timezone.utc).year: float(
                    e["numerator"]
                )
                / float(e["denominator"])
                for e in splits
            }
            if actual_splits != expected_splits:
                raise ValueError("Unreviewed corporate action in extended snapshot")
            if symbol == "KO":
                amounts = [
                    float(e["amount"])
                    for e in events.get("dividends", {}).values()
                    if datetime.fromtimestamp(e["date"], timezone.utc).year == 2012
                ]
                np.testing.assert_allclose(amounts, [0.255] * 4, atol=1e-12)
                np.testing.assert_allclose(
                    series[-1].loc["2012-08-13"] / series[-1].loc["2012-08-10"],
                    1.0,
                    atol=0.1,
                )
        elif args.pair == "pep-ko" and record.get("events", {}).get("splits"):
            raise ValueError(
                "Dividend amounts require split-basis reconciliation for this pair"
            )
        sources[symbol] = {
            "url": url,
            "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "events": record.get("events", {}),
        }
    if not series[0].index.equals(series[1].index):
        raise ValueError("Do not silently intersect vendor calendars")
    prices = validate_prices(pd.concat(series, axis=1))
    distributions = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for symbol, source in sources.items():
        for event in source["events"].get("dividends", {}).values():
            date = (
                pd.Timestamp(event["date"], unit="s", tz="UTC")
                .tz_convert("America/New_York")
                .normalize()
                .tz_localize(None)
            )
            if date not in distributions.index:
                if (
                    args.extended
                    and symbol == "KO"
                    and date == pd.Timestamp("2001-09-12")
                ):
                    # NYSE was closed September 11–14. No fills occur in this gap;
                    # credit carried holdings at the first observed close, before fills.
                    date = distributions.index[distributions.index.searchsorted(date)]
                    if date != pd.Timestamp("2001-09-17"):
                        raise ValueError("Unexpected post-closure trading calendar")
                else:
                    raise ValueError(
                        "Dividend date has no corresponding price observation"
                    )
            distributions.loc[date, symbol] += float(event["amount"])
    if prices.isna().any().any():
        raise ValueError("Inspect missing vendor observations before evaluating")
    capital = 10000.0
    curves = []
    cohorts = []
    orders = []
    rejected = []
    options = FitOptions(seed=20261025)
    for start in pd.date_range(
        "2000-01-01" if args.extended else "2019-01-01",
        "2025-07-01" if args.extended else "2023-07-01",
        freq="6MS",
    ):
        training = prices.loc[
            (prices.index >= start - pd.DateOffset(years=4)) & (prices.index < start)
        ]
        testing = prices.loc[
            (prices.index >= start) & (prices.index < start + pd.DateOffset(months=6))
        ]
        if len(training) < 990 or len(testing) < 120:
            raise ValueError("Incomplete formation or evaluation period")
        fit = fit_pair(training[x], training[y], options=options)
        p = fit.parameters
        label = str(start.date())
        reason = None
        entries = pd.DataFrame(False, index=testing.index, columns=[pair_id])
        exits = entries.copy()
        if p.beta <= 0 or p.sigma_m == 0:
            reason = (
                "Unsupported hedge sign or no transient component; cash-only cohort"
            )
        else:
            components = score_components(
                training[y] - p.beta * training[x],
                testing[y] - p.beta * testing[x],
                p,
                variant="formation",
                window=30,
            )
            entries, exits = decisions(
                components["testing_score"].rename(pair_id), 1, -0.5
            )
        config = ExecutionConfig(
            capital=capital, fees=0.001, slippage=0.001, annual_borrow=0.01
        )
        ledger = backtest_pairs(
            testing,
            {pair_id: (x, y, p.beta if p.beta > 0 else 1)},
            entries,
            exits,
            config=config,
            cash_distributions=distributions.loc[testing.index],
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
            + (positions.shift(1).fillna(0) * distributions.loc[testing.index])
            .sum(axis=1)
            .cumsum()
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
                "net_dividend_accrual": float(ledger.cash_distributions.sum()),
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
        "pair": [x, y],
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
        "price_basis": "Vendor split-adjusted Close, not dividend-adjusted Adj Close. Dividend entitlements accrue to carried holdings on ex-dates: longs receive and shorts owe. No separate payment-date receivables/taxes modeled. Model/signal inputs also use Close, not adjusted-close units.",
        "net_dividend_accrual": sum(
            cohort["net_dividend_accrual"] for cohort in cohorts
        ),
        "sources": sources,
        "corporate_action_review": (
            "Extended snapshot: PEP 1996 and KO 1996/2012 splits checked; "
            "Yahoo Close and dividend events use split-adjusted units. KO 2012 "
            "four quarterly dividends are each 0.255 (pre-split actual 0.51); "
            "no second split adjustment applied. PEP 1997 spin-off is in formation "
            "history only; no positions held then. KO September 12, 2001 dividend "
            "accrues on September 17 before fills because the exchange was closed."
            if args.extended
            else "Original snapshot corporate-action guard"
        ),
        "package_fingerprint": package_fingerprint(),
        "script_sha256": hashlib.sha256(
            Path(__file__).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "Fixed pair; no broad pair-universe selection or reverse-orientation search",
            "Ex-dividend cash-accrual approximation, not actual payment-date cash settlement; raw ex-dividend price gaps remain in signal inputs",
            "Vendor history is not point-in-time audited",
            "Borrow availability/actual costs unverified; 1% annual charge assumed",
            "Period selected retrospectively before this run; no parameter or threshold search",
            "No untouched future test or calibrated profitability significance claim",
        ],
    }
    (root / f"docs/results/longer-pair-{args.pair}{suffix}.json").write_text(
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
