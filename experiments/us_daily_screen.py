"""Long-history US-stock PCI research: frozen peers, annual walk-forward fits.

Run from an editable installation. Raw vendor responses and fit checkpoints are
reused only when their source/design digests match. No performance-based tuning.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from partial_cointegration import FitError, FitOptions, Parameters, fit_pair
from partial_cointegration.execution import ExecutionConfig, backtest_pairs
from partial_cointegration.io import estimator_fingerprint, package_fingerprint
from partial_cointegration.signals import decisions, score_components


def digest(path: Path) -> str:
    """Identify a frozen input or cached response."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(task: tuple[str, str, str]) -> dict[str, Any]:
    """Cache one maximum-history daily response, retaining source identity."""
    symbol, end, directory = task
    path = Path(directory) / f"{symbol}.json"
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?" + urlencode(
        dict(
            interval="1d",
            events="div,splits",
            period1=-2208988800,
            period2=int(pd.Timestamp(end, tz="UTC").timestamp()),
        )
    )
    try:
        if not path.exists():
            with urlopen(
                Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60
            ) as response:
                data = response.read()
            # Validate the payload before caching it.
            payload = json.loads(data)
            if payload["chart"]["error"]:
                raise ValueError(str(payload["chart"]["error"]))
            path.write_bytes(data)
        return {"symbol": symbol, "url": url, "sha256": digest(path), "status": "ok"}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {
            "symbol": symbol,
            "url": url,
            "status": "unavailable",
            "error": str(exc),
        }


def parse_history(path: Path, symbol: str, end: str) -> tuple[Any, Any, dict[str, Any]]:
    """Parse split-adjusted Close plus same-vendor cash dividend events.

    Non-session ex-dates accrue at the first following observed session, before
    fills. Corporate-action units remain provisional without issuer tie-outs.
    """
    record = json.loads(path.read_bytes())["chart"]["result"][0]
    meta = record["meta"]
    if meta["symbol"] != symbol or meta["currency"] != "USD":
        raise ValueError("Unexpected vendor symbol/currency")
    index = (
        pd.to_datetime(record["timestamp"], unit="s", utc=True)
        .tz_convert("America/New_York")
        .normalize()
        .tz_localize(None)
    )
    prices = pd.Series(
        record["indicators"]["quote"][0]["close"], index=index, name=symbol, dtype=float
    )
    prices = prices.loc[prices.index < pd.Timestamp(end)].sort_index()
    if (
        not prices.index.is_unique
        or (prices.dropna() <= 0).any()
        or np.isinf(prices).any()
    ):
        raise ValueError("Duplicate dates or invalid observed prices")
    dividends = pd.Series(0.0, index=prices.index, name=symbol)
    shifted = []
    for event in record.get("events", {}).get("dividends", {}).values():
        date = (
            pd.Timestamp(event["date"], unit="s", tz="UTC")
            .tz_convert("America/New_York")
            .normalize()
            .tz_localize(None)
        )
        if date < prices.index[0] or date >= pd.Timestamp(end):
            continue
        pos = prices.index.searchsorted(date)
        if pos >= len(prices):
            raise ValueError("Dividend after final observed price")
        actual = prices.index[pos]
        if actual != date:
            shifted.append(
                {"ex_date": str(date.date()), "accrual_date": str(actual.date())}
            )
        dividends.loc[actual] += float(event["amount"])
    info = dict(
        symbol=symbol,
        name=meta.get("longName", meta.get("shortName", symbol)),
        first=str(prices.index[0].date()),
        last=str(prices.index[-1].date()),
        observations=int(prices.notna().sum()),
        missing=int(prices.isna().sum()),
        splits=len(record.get("events", {}).get("splits", {})),
        dividends=len(record.get("events", {}).get("dividends", {})),
        shifted_dividends=shifted,
    )
    return prices, dividends, info


def fit_job(task: tuple[str, str, str, Any, Any, int]) -> dict[str, Any]:
    """Fit one formation-only window with explicit failure reporting."""
    pair, group, start, x, y, seed = task
    try:
        fitted = fit_pair(x, y, options=FitOptions(seed=seed))
        p = fitted.parameters
        return dict(
            pair=pair,
            group=group,
            start=start,
            status="ok",
            parameters=asdict(p),
            log_ratio=fitted.log_ratio,
            rho=p.rho,
            variance_fraction=p.variance_fraction,
            beta=p.beta,
            selected_model=fitted.selected,
            training_n=len(x),
        )
    except (FitError, ValueError) as exc:
        return dict(
            pair=pair, group=group, start=start, status="fit_failed", error=str(exc)
        )


def eligible(row: dict[str, Any], config: dict[str, Any]) -> bool:
    """Fixed descriptive screen, not a significance or multiple-testing claim."""
    return bool(
        row["status"] == "ok"
        and row["beta"] > 0
        and row["rho"] >= config["rho_min"]
        and row["variance_fraction"] >= config["variance_min"]
    )


def metrics(equity: Any, capital: float) -> dict[str, float]:
    """Compute returns and drawdown with the initial deposit in the high watermark."""
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25)
    multiple = float(equity.iloc[-1] / capital)
    return dict(
        total_return=multiple - 1,
        cagr=multiple ** (1 / years) - 1 if multiple > 0 else -1.0,
        max_drawdown=float(-(equity / equity.cummax().clip(lower=capital) - 1).min()),
    )


def run_ledger(
    prices: Any,
    dividends: Any,
    fits: list[dict[str, Any]],
    start: Any,
    end: Any,
    capital: float,
    config: dict[str, Any],
) -> tuple[Any, Any, dict[str, Any]]:
    """Backtest fixed formation decisions; independently reconcile cash/equity."""
    test_index = prices.loc[(prices.index >= start) & (prices.index < end)].index
    if len(test_index) < 2:
        raise ValueError("Evaluation needs at least two observed sessions")
    if not fits:
        curve = pd.Series(capital, index=test_index, name="equity")
        return (
            curve,
            pd.DataFrame(),
            dict(entries=0, rejections=0, fees=0.0, dividends=0.0, borrow=0.0),
        )
    pairs = {}
    entries, exits = {}, {}
    names = set()
    for row in fits:
        pair = row["pair"]
        x, y = pair.split("/")
        names.update([x, y])
        pairs[pair] = (x, y, row["beta"])
        training = prices.loc[
            (prices.index >= start - pd.DateOffset(years=config["formation_years"]))
            & (prices.index < start),
            [x, y],
        ]
        testing = prices.loc[test_index, [x, y]]
        p = Parameters(**row["parameters"])
        components = score_components(
            training[y] - p.beta * training[x], testing[y] - p.beta * testing[x], p
        )
        opening, closing = decisions(
            components["testing_score"].rename(pair), config["entry"], config["exit"]
        )
        entries[pair], exits[pair] = opening[pair], closing[pair]
    testing = prices.loc[test_index, sorted(names)]
    cash_dividends = dividends.reindex(
        index=test_index, columns=testing.columns
    ).fillna(0)
    execution = ExecutionConfig(
        capital=capital,
        fees=config["fees"],
        slippage=config["slippage"],
        annual_borrow=config["annual_borrow"],
    )
    ledger = backtest_pairs(
        testing,
        pairs,
        pd.DataFrame(entries),
        pd.DataFrame(exits),
        config=execution,
        cash_distributions=cash_dividends,
    )
    # Reconstruct positions from fill records, rather than trusting ledger holdings.
    changes = pd.DataFrame(0.0, index=test_index, columns=testing.columns)
    flows = pd.Series(0.0, index=test_index)
    for fill in ledger.orders.itertuples():
        changes.loc[fill.fill_time, fill.security] += fill.units
        flows.loc[fill.fill_time] -= fill.units * fill.price + fill.fees
    positions = changes.cumsum()
    marks = testing.ffill()
    borrow = (
        (positions.shift(1).fillna(0).clip(upper=0).abs() * marks).sum(axis=1)
        * execution.annual_borrow
        / 252
    )
    divflow = (positions.shift(1).fillna(0) * cash_dividends).sum(axis=1)
    independent = (
        capital + (flows - borrow + divflow).cumsum() + (positions * marks).sum(axis=1)
    )
    np.testing.assert_allclose(independent, ledger.equity, atol=1e-7, rtol=1e-10)
    np.testing.assert_allclose(positions, ledger.holdings, atol=1e-10)
    if not np.allclose(positions.iloc[-1], 0):
        raise AssertionError("Annual cohort did not liquidate")
    signal_fills = ledger.orders.loc[ledger.orders.kind.isin(["entry", "exit"])]
    if not (signal_fills.fill_time > signal_fills.decision_time).all():
        raise AssertionError("Signal filled without a later close")
    stats = dict(
        entries=int((ledger.orders.kind == "entry").sum() // 2),
        rejections=len(ledger.rejections),
        fees=float(ledger.orders.fees.sum()),
        dividends=float(divflow.sum()),
        borrow=float(borrow.sum()),
    )
    return ledger.equity, ledger.orders, stats


def backtest_pair(task: tuple[Any, ...]) -> dict[str, Any]:
    """Evaluate one pair independently; worker files have disjoint pair names."""
    pair, group, price_frame, dividend_frame, rows, c, output = task
    out = Path(output)
    lookup = {(r["pair"], r["start"]): r for r in rows}
    final = pd.Timestamp(c["end_exclusive"])
    watchlist_date = price_frame.index[-1] + pd.Timedelta(days=1)
    evaluation_starts = [
        pd.Timestamp(r["start"])
        for r in rows
        if pd.Timestamp(r["start"]) < watchlist_date
    ]
    results: list[dict[str, Any]] = []
    cohort_rows: list[dict[str, Any]] = []
    capital = c["capital"]
    curves, pair_orders = [], []
    active_years, trade_count, reject_count = 0, 0, 0
    x, y = pair.split("/")
    invalid = []
    for start in evaluation_starts:
        row = lookup[(pair, str(start.date()))]
        if row["status"] != "ok":
            # Unavailable formation is not assigned a zero return.
            continue
        test_end = min(start + pd.DateOffset(years=1), final)
        if (
            len(
                price_frame.loc[
                    (price_frame.index >= start) & (price_frame.index < test_end)
                ]
            )
            < 2
        ):
            continue
        passed = eligible(row, c)
        try:
            equity, orders, stats = run_ledger(
                price_frame[[x, y]],
                dividend_frame[[x, y]],
                [row] if passed else [],
                start,
                test_end,
                capital,
                c,
            )
        except (ValueError, AssertionError) as exc:
            invalid.append(dict(start=str(start.date()), error=str(exc)))
            break  # Do not silently skip a failed evaluation and resume later.
        ret = float(equity.iloc[-1] / capital - 1)
        cohort_rows.append(
            dict(
                pair=pair,
                group=group,
                year=start.year,
                passed=passed,
                return_=ret,
                **stats,
            )
        )
        capital = float(equity.iloc[-1])
        curves.append(equity)
        pair_orders.append(orders)
        active_years += passed
        trade_count += stats["entries"]
        reject_count += stats["rejections"]
        if capital <= 0:
            break
    if not curves:
        return {"result": None, "cohorts": cohort_rows}
    curve = pd.concat(curves)
    # A gap from a later formation failure makes annualized statistics invalid.
    expected_years = set(range(curve.index[0].year, curve.index[-1].year + 1))
    missing_years = sorted(expected_years - set(curve.index.year))
    curve.to_csv(
        out / "pair-equity" / (pair.replace("/", "-") + ".csv"), index_label="date"
    )
    pd.concat(
        [orders for orders in pair_orders if len(orders)] or [pd.DataFrame()]
    ).to_csv(out / "pair-equity" / (pair.replace("/", "-") + "-fills.csv"), index=False)
    price_frame[[x, y]].dropna(how="all").to_csv(
        out / "prices" / (pair.replace("/", "-") + ".csv"), index_label="date"
    )
    latest = lookup[(pair, str(watchlist_date.date()))]
    prior = lookup[(pair, f"{final.year}-01-01")]
    recent = curve.loc[curve.index >= "2016-01-01"]
    previous = curve.loc[curve.index < "2016-01-01"]
    recent_base = float(previous.iloc[-1]) if len(previous) else c["capital"]
    results.append(
        dict(
            pair=pair,
            group=group,
            start=str(curve.index[0].date()),
            end=str(curve.index[-1].date()),
            sessions=len(curve),
            active_years=active_years,
            entries=trade_count,
            rejections=reject_count,
            latest_eligible=eligible(latest, c),
            latest_rho=latest.get("rho"),
            latest_fraction=latest.get("variance_fraction"),
            latest_log_ratio=latest.get("log_ratio"),
            jan2026_eligible=eligible(prior, c),
            **metrics(curve, c["capital"]),
            recent_return=(
                float(recent.iloc[-1] / recent_base - 1) if len(recent) else None
            ),
            missing_years=missing_years,
            invalid=invalid,
        )
    )
    print(
        f"Backtested {pair}: {trade_count} entries, {curve.index[0].year}–{curve.index[-1].year}",
        flush=True,
    )

    return {"result": results[0], "cohorts": cohort_rows}


def main() -> None:
    """Download, checkpoint all formation fits, and evaluate frozen rules."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/configs/us-daily-screen.json")
    parser.add_argument("--cache", default=".test-deps/us-daily-screen")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    config_path = Path(args.config)
    c = json.loads(config_path.read_text())
    cache, out = Path(args.cache), Path(args.output)
    raw = cache / "raw"
    for directory in [raw, cache / "fits", out, out / "prices", out / "pair-equity"]:
        directory.mkdir(parents=True, exist_ok=True)
    symbols = sorted({s for values in c["groups"].values() for s in values})
    with ThreadPoolExecutor(max_workers=3) as executor:
        sources = list(
            executor.map(download, [(s, c["end_exclusive"], str(raw)) for s in symbols])
        )
    prices, dividends, coverage = {}, {}, []
    for source in sources:
        if source["status"] != "ok":
            print(f"Unavailable: {source['symbol']}: {source['error']}", flush=True)
            continue
        symbol = source["symbol"]
        try:
            price, dividend, info = parse_history(
                raw / f"{symbol}.json", symbol, c["end_exclusive"]
            )
            prices[symbol], dividends[symbol] = price, dividend
            coverage.append(info)
        except (ValueError, KeyError, TypeError) as exc:
            source.update(status="invalid", error=str(exc))
    # Union calendar retains missing prices rather than silently intersecting pairs.
    price_frame = pd.DataFrame(prices).sort_index()
    dividend_frame = pd.DataFrame(dividends).reindex(price_frame.index).fillna(0)
    price_frame.to_csv(out / "all-prices.csv", index_label="date")
    dividend_frame.to_csv(out / "all-dividends.csv", index_label="date")
    pd.DataFrame(coverage).to_csv(out / "data-coverage.csv", index=False)
    (out / "sources.json").write_text(json.dumps(sources, indent=2))
    universe = []
    for group, members in c["groups"].items():
        for x, y in combinations(sorted(members), 2):
            if group == "share_classes" and (x, y) not in [
                ("GOOG", "GOOGL"),
                ("FOX", "FOXA"),
                ("NWS", "NWSA"),
            ]:
                continue
            if x in prices and y in prices:
                universe.append((f"{x}/{y}", group))
    identity = hashlib.sha256(
        json.dumps(
            dict(config=c, sources=sources, estimator=estimator_fingerprint()),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    final = pd.Timestamp(c["end_exclusive"])
    starts = [
        pd.Timestamp(year=y, month=1, day=1)
        for y in range(c["first_evaluation_year"], final.year + 1)
    ]
    # Last screen is a watchlist only: it has no subsequent evaluation data.
    watchlist_date = price_frame.index[-1] + pd.Timedelta(days=1)
    if watchlist_date not in starts:
        starts.append(watchlist_date)
    fits = []
    print(
        f"Data: {len(prices)} / {len(symbols)} symbols; {len(universe)} peer pairs; {price_frame.index[0].date()} to {price_frame.index[-1].date()}",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for start in starts:
            checkpoint = cache / "fits" / f"{start.date()}.json"
            if checkpoint.exists():
                saved = json.loads(checkpoint.read_text())
                if saved["identity"] == identity:
                    fits.extend(saved["rows"])
                    print(f"Reused formation {start.date()}", flush=True)
                    continue
            tr = price_frame.loc[
                (price_frame.index >= start - pd.DateOffset(years=c["formation_years"]))
                & (price_frame.index < start)
            ]
            tasks, rows = [], []
            for pair, group in universe:
                x, y = pair.split("/")
                # No evaluation observations participate in eligibility checks.
                if (
                    len(tr) < c["minimum_formation_observations"]
                    or tr[[x, y]].isna().any().any()
                ):
                    rows.append(
                        dict(
                            pair=pair,
                            group=group,
                            start=str(start.date()),
                            status="insufficient_formation",
                        )
                    )
                    continue
                tasks.append(
                    (
                        pair,
                        group,
                        str(start.date()),
                        tr[x].to_numpy(),
                        tr[y].to_numpy(),
                        c["seed"],
                    )
                )
            rows.extend(executor.map(fit_job, tasks, chunksize=4))
            checkpoint.write_text(json.dumps(dict(identity=identity, rows=rows)))
            fits.extend(rows)
            print(f"Formation {start.date()}: {len(tasks)} fits", flush=True)
    (out / "all-fits.json").write_text(json.dumps(fits))
    fit_table = pd.DataFrame(
        [{k: v for k, v in row.items() if k != "parameters"} for row in fits]
    )
    fit_table.to_csv(out / "all-screens.csv", index=False)
    results, cohort_rows = [], []
    evaluation_starts = [s for s in starts if s < watchlist_date]
    pair_tasks = []
    for pair, group in universe:
        x, y = pair.split("/")
        pair_tasks.append(
            (
                pair,
                group,
                price_frame[[x, y]],
                dividend_frame[[x, y]],
                [r for r in fits if r["pair"] == pair],
                c,
                str(out),
            )
        )
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(backtest_pair, pair_tasks, chunksize=1):
            if result["result"] is not None:
                results.append(result["result"])
            cohort_rows.extend(result["cohorts"])
    table = pd.DataFrame(results).sort_values(
        ["latest_eligible", "latest_log_ratio", "pair"], ascending=[False, True, True]
    )
    table.to_csv(out / "pair-summary.csv", index=False)
    pd.DataFrame(cohort_rows).to_csv(out / "annual-results.csv", index=False)
    # Independent annual selection: rank by past likelihood, disjoint securities,
    # equal allocations, one shared-capital ledger. No return-based pair selection.
    capital = c["capital"]
    portfolio_curves: list[Any] = []
    selection: list[dict[str, Any]] = []
    for start in evaluation_starts:
        ranked = sorted(
            [r for r in fits if r["start"] == str(start.date()) and eligible(r, c)],
            key=lambda r: (r["log_ratio"], r["pair"]),
        )
        selected: list[dict[str, Any]] = []
        used: set[str] = set()
        for row in ranked:
            names = set(row["pair"].split("/"))
            if not names & used and len(selected) < c["portfolio_pairs"]:
                selected.append(row)
                used.update(names)
        if not selected and not portfolio_curves:
            continue
        try:
            equity, orders, stats = run_ledger(
                price_frame,
                dividend_frame,
                selected,
                start,
                min(start + pd.DateOffset(years=1), final),
                capital,
                c,
            )
        except (ValueError, AssertionError) as exc:
            selection.append(dict(year=start.year, invalid=str(exc)))
            break
        selection.append(
            dict(year=start.year, pairs=[r["pair"] for r in selected], **stats)
        )
        portfolio_curves.append(equity)
        capital = float(equity.iloc[-1])
        if capital <= 0:
            break
    portfolio = pd.concat(portfolio_curves)
    portfolio.to_csv(out / "portfolio-equity.csv", index_label="date")
    (out / "portfolio-selection.json").write_text(json.dumps(selection, indent=2))
    summary = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        config=c,
        config_sha256=digest(config_path),
        identity=identity,
        estimator=estimator_fingerprint(),
        package=package_fingerprint(),
        runner_sha256=digest(Path(__file__)),
        source="Yahoo Finance chart API; cached raw JSON SHA256 in sources.json",
        symbols_requested=len(symbols),
        symbols_downloaded=len(prices),
        candidate_pairs=len(universe),
        pairs_backtested=len(table),
        fit_status_counts=fit_table.status.value_counts().to_dict(),
        watchlist_asof=str(price_frame.index[-1].date()),
        portfolio=dict(
            start=str(portfolio.index[0].date()),
            end=str(portfolio.index[-1].date()),
            **metrics(portfolio, c["capital"]),
        ),
        limitations=[
            "Present-day hand-selected peer groups: survivor/selection bias, not point-in-time index membership.",
            "Multiple comparisons; likelihood rank and parameter screens are not calibrated significance.",
            "Fixed annual 4-year formation / subsequent 1-year evaluation, with partial final 2026 cohort.",
            "Yahoo Close and dividend share units provisional across mergers/spinoffs; issuer action tie-outs not performed.",
            "Annual liquidation and refitting; fixed initial allocation sizing within each annual cohort.",
            "Long predictor / short response only; alphabetical orientation frozen, not optimized.",
            "10bp fee and 10bp adverse slippage per fill plus 1% annual short borrow; availability, financing and taxes unverified.",
            "No money paid for historical data; unavailable/delisted histories reported, no fabricated observations.",
            "Latest formation watchlist has no subsequent evaluation; backtests are retrospective exploratory research.",
        ],
    )
    (out / "study.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["portfolio"], indent=2), flush=True)


if __name__ == "__main__":
    main()
