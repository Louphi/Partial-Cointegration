"""Summarize an executed daily screen without changing rankings or fitting rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    """Write a readable research report, audited candidate table and figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    args = parser.parse_args()
    root = Path(args.study)
    study = json.loads((root / "study.json").read_text())
    table = pd.read_csv(root / "pair-summary.csv")
    annual = pd.read_csv(root / "annual-results.csv")
    screens = pd.read_csv(root / "all-screens.csv")
    coverage = pd.read_csv(root / "data-coverage.csv")
    current = screens.loc[
        screens.start
        == (pd.Timestamp(study["watchlist_asof"]) + pd.Timedelta(days=1))
        .date()
        .isoformat()
    ]
    table = table.merge(
        current[["pair", "selected_model"]].rename(
            columns={"selected_model": "latest_model"}
        ),
        on="pair",
        how="left",
    )
    table["continuous_history"] = (table.missing_years == "[]") & (
        table.invalid == "[]"
    )
    table.loc[~table.continuous_history, "cagr"] = float("nan")
    last_year = annual.loc[
        annual.year == pd.Timestamp(study["watchlist_asof"]).year, ["pair", "return_"]
    ].rename(columns={"return_": "return_2026_ytd"})
    table = table.merge(last_year, on="pair", how="left")
    table.to_csv(root / "research-summary.csv", index=False)
    current_pass = table.loc[table.latest_eligible].copy()
    leaders = current_pass.loc[current_pass.continuous_history].head(10)
    current_pass.to_csv(root / "current-watchlist.csv", index=False)
    portfolio = pd.read_csv(
        root / "portfolio-equity.csv", index_col="date", parse_dates=True
    ).iloc[:, 0]
    selections = json.loads((root / "portfolio-selection.json").read_text())
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, height_ratios=[2, 1])
    axes[0].plot(
        portfolio.index, portfolio, label="Annual past-data selection", color="#155e75"
    )
    axes[0].axhline(
        study["config"]["capital"],
        color="#666666",
        linestyle="--",
        label="No-trade cash (no interest)",
    )
    axes[0].set_ylabel("Account value ($)")
    axes[0].legend()
    axes[0].set_title(
        "Shared-capital portfolio · 10bp fees + 10bp slippage per fill + 1% annual borrow"
    )
    drawdown = (
        portfolio / portfolio.cummax().clip(lower=study["config"]["capital"]) - 1
    ) * 100
    axes[1].fill_between(drawdown.index, drawdown, 0, color="#b45309", alpha=0.4)
    axes[1].set_ylabel("Drawdown (%)")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(root / "portfolio-performance.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    for ax, row in zip(axes.flat, leaders.head(6).itertuples()):
        equity = pd.read_csv(
            root / "pair-equity" / (row.pair.replace("/", "-") + ".csv"),
            index_col="date",
            parse_dates=True,
        ).iloc[:, 0]
        ax.plot(equity.index, equity / study["config"]["capital"], color="#155e75")
        ax.axhline(1, color="#666666", linestyle="--", linewidth=0.7)
        ax.set_title(f"{row.pair} · {row.start[:4]}–{row.end[:4]}")
        ax.set_ylabel("Initial capital multiple")
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Latest-screen candidates: retrospective annual walk-forward illustrations\nChosen by latest formation likelihood rank, not by historical profits"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(root / "candidate-histories.png", dpi=150)
    plt.close(fig)

    def percent(value: Any) -> str:
        return "n/a" if pd.isna(value) else f"{value:+.1%}"

    pm = study["portfolio"]
    lines = [
        "# US daily partial-cointegration research",
        f"Snapshot through **{study['watchlist_asof']}**. This is an executed retrospective screen, not a trade recommendation or a calibrated discovery claim.",
        f"**Finding:** {len(current_pass)} pairs pass the latest descriptive screen, but only {int((table.total_return > 0).sum())} of {len(table)} pair backtests were profitable after modeled costs. The past-data-selected shared-capital portfolio returned {percent(pm['total_return'])}. Mean-reverting fit quality alone did not produce a profitable strategy under these rules.",
        "## Coverage and candidate funnel",
        f"- {study['symbols_downloaded']} of {study['symbols_requested']} requested US-stock histories; {int(coverage.observations.sum()):,} observed stock-day prices.",
        f"- Earliest price: {coverage['first'].min()}; latest: {coverage['last'].max()}. Individual listing histories vary.",
        f"- {study['candidate_pairs']} fixed peer pairs; {study['pairs_backtested']} produced backtests.",
        f"- {study['fit_status_counts'].get('ok',0):,} accepted formation fits; {study['fit_status_counts'].get('fit_failed',0):,} failed fits; {study['fit_status_counts'].get('insufficient_formation',0):,} unavailable four-year formation windows.",
        f"- {len(current_pass)} pairs meet the latest descriptive screen; {int((current_pass.latest_model == 'mixed').sum())} select the mixed M+R model. Pure-AR boundary fits are identified separately in the CSV.",
        "- Requested ticker K returned HTTP 404; it was not replaced with an invented history or another security.",
        "## What the backtest actually did",
        "Peer groups and parameters were declared before return computation. Within each pair, four calendar years form the model; the following calendar year is evaluated. Windows need at least 900 observed sessions and complete aligned formation prices (historical exchange closures mean four years do not always contain 1,000 sessions).",
        "The exact conditional model estimates beta, rho and component variances. The descriptive trading screen requires beta > 0, rho >= 0.5 and transient variance fraction >= 0.5. Failing the screen means cash for that year. Pair orientation is fixed alphabetically and only long-predictor/short-response entries are supported. Entries use score > 1, exits score < -0.5, with formation-only scaling and continued filter state. All remaining holdings are liquidated at year end. The final 2026 evaluation stops at the snapshot date.",
        "Costs: 10bp fees and 10bp adverse slippage on each leg at each fill, plus 1% annual short borrow (252 observations/year). Ex-date dividend cash accrues to previous-row holdings, long receives and short owes. Yahoo Close is used with its dividend events in vendor share units. This is provisional for corporate actions until issuer tie-outs are complete. No dividend-adjusted price plus dividend double counting is used.",
        "Every successful ledger is independently reconciled from executed units, prices, fees, borrow and dividend flows. Signal fills must follow their decision date; annual holdings must end at zero. Margin rejections remain recorded in annual results. Pair results use separate $10,000 accounts. The portfolio uses a single $10,000 account, chooses up to ten disjoint pairs annually by prior formation likelihood rank, and splits funding equally. Short-sale proceeds do not add capacity.",
        "## Shared-capital result",
        f"From **{pm['start']} to {pm['end']}**: total return **{percent(pm['total_return'])}**, CAGR **{percent(pm['cagr'])}**, maximum drawdown **{pm['max_drawdown']:.1%}**. Cash earns 0% in this comparison. Costs are already deducted.",
        "![Portfolio performance](portfolio-performance.png)",
        "## Latest watchlist and historical illustrations",
        "Sorted by the latest formation likelihood rank, never by backtest return. The latest screen uses data through October 30, 2026 and has **no subsequent test period**. Historical performance of this now-selected watchlist is conditional on that retrospective selection; it is not out-of-sample validation of today's picks. Whole-history windows differ and must not be treated as directly comparable investments.",
        "| Pair | Model | rho | M variance fraction | Annual backtest span | Net total | 2016 onward | 2026 YTD | Max drawdown | Entries |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in leaders.itertuples():
        lines.append(
            f"| {row.pair} | {row.latest_model} | {row.latest_rho:.3f} | {row.latest_fraction:.1%} | {row.start}–{row.end} | {percent(row.total_return)} | {percent(row.recent_return)} | {percent(row.return_2026_ytd)} | {row.max_drawdown:.1%} | {row.entries} |"
        )
    lines.extend(
        [
            "![Latest candidate histories](candidate-histories.png)",
            "A mixed fit models both transient and permanent variation. A pure-AR fit sits on the fully mean-reverting boundary and should not be described as confirmed partial cointegration. A likelihood ranking is not a p-value; rho and variance-fraction screens alone do not resolve multiple testing.",
            "## Research judgment and rejection rules",
            "Actionability: watchlist / needs proof. The model flags temporary spread behavior, not verified mispricing. No issuer-specific catalyst, valuation wedge, short availability, or executable bid/ask evidence has been established. Why now: these pairs pass the latest frozen statistical screen. What would make them investable: calibrated pair-specific inference, corporate-action and identifier validation, realistic borrow/financing, and a new period collected after this screen. What kills the idea: unstable parameters, disappearance of the transient component, failed corporate-action reconciliation, excessive drawdown or costs, or unavailable borrow. Do not select historical winners and call that prospective validation.",
            "## Limitations and quality flags",
            *[f"- {item}" for item in study["limitations"]],
            "- Data are a present-day vendor snapshot, including revisions and back-adjustments. Point-in-time security identities, dead companies and historical group membership are absent. More years do not remove survivor bias.",
            "- Some pair histories have missing formation years. `research-summary.csv` flags these as non-continuous and suppresses their CAGR; do not connect their gaps as a continuous qualified backtest. Evaluation errors stop that pair rather than being silently discarded.",
            f"- {int((~table.continuous_history).sum())} pair summaries have an evaluation error or a gap. The table above shows continuous, error-free histories only.",
            "## Use these data in the app",
            "Choose Historical CSV/parquet and paste the absolute path to a file under this study's `prices/` folder. For example, `prices/AMAT-KLAC.csv` contains the full available history for those securities. Choose a period in which both stocks have complete formation prices, then choose the first ticker as predictor and the second as response. The app uses one fit/split and currently omits dividends; its result will differ from this annual refitting, dividend-accruing study.",
            "## Files and sources",
            "- `research-summary.csv`: all pairs, current screen, whole-history and recent returns, YTD and quality flags.",
            "- `current-watchlist.csv`: latest descriptive-screen candidates, including quality flags.",
            "- `all-prices.csv`, `all-dividends.csv`, `data-coverage.csv`: full downloaded panels and coverage.",
            "- `prices/`: app-compatible two-column files; `pair-equity/`: annual walk-forward curves and actual fills.",
            "- `annual-results.csv`: every evaluated pair/year, including no-trade years and rejection counts.",
            "- `all-screens.csv`, `all-fits.json`: all formation outcomes and estimates; `portfolio-selection.json`: prior-data annual selections.",
            "- `sources.json`: exact Yahoo request URLs and SHA256 of cached raw responses; `study.json`: frozen design, source/model hashes and limitations.",
            "Data: [Yahoo Finance](https://finance.yahoo.com/). Adjustment conventions: [Yahoo historical-data help](https://uk.help.yahoo.com/kb/finance/historical-prices-adjusted-close-sln2311.html) and [adjusted-close help](https://in.help.yahoo.com/kb/adjusted-close-sln28256.html). The broker connector was checked but caps requests at 1,000 bars, so the long history uses Yahoo. Raw responses are retained in the repository's ignored `.test-deps/us-daily-screen/raw/` directory.",
        ]
    )
    invalid_portfolio = [r for r in selections if "invalid" in r]
    if invalid_portfolio:
        lines.insert(
            lines.index("## Shared-capital result") + 1,
            f"**Portfolio evaluation stopped with an error:** {invalid_portfolio}. The displayed span is incomplete.",
        )
    report = "\n\n".join(lines)
    report = report.replace("\n\n|", "\n|").replace("\n| Pair", "\n\n| Pair")
    (root / "REPORT.md").write_text(report, encoding="utf-8")
    print(
        leaders[
            ["pair", "latest_model", "total_return", "recent_return", "return_2026_ytd"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
