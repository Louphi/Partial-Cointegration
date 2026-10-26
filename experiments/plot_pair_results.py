"""Plot saved backtest ledgers; does not refit or select strategy parameters."""

import json
import argparse
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair", choices=["goog-googl", "pep-ko"], default="goog-googl"
    )
    parser.add_argument(
        "--extended", action="store_true", help="Plot 2000–2025 PEP/KO daily test only"
    )
    args = parser.parse_args()
    if args.extended and args.pair != "pep-ko":
        parser.error("Extended results support PEP/KO only")
    extension = "-extended" if args.extended else ""
    root = Path(__file__).resolve().parents[1]
    output = root / "docs/results"
    daily_report = json.loads(
        (output / f"longer-pair-{args.pair}{extension}.json").read_text()
    )
    cache = (
        root
        / ".test-deps"
        / (("longer-real-pair" if args.pair == "goog-googl" else args.pair) + extension)
    )
    pair_label = " / ".join(daily_report["pair"])
    suffix = ("" if args.pair == "goog-googl" else f"-{args.pair}") + extension
    daily = pd.read_csv(
        cache / "output/equity.csv",
        index_col=0,
        parse_dates=True,
    ).iloc[:, 0]
    if not daily.index.is_unique or not daily.index.is_monotonic_increasing:
        raise ValueError("Need unique ordered observations")
    np.testing.assert_allclose(daily.iloc[-1], daily_report["final_capital"], atol=1e-8)
    plt.rcParams.update(
        {"font.size": 12, "axes.spines.top": False, "axes.spines.right": False}
    )
    dollars = FuncFormatter(lambda value, _: f"${value:,.0f}")
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8, 9),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.3, 1]},
    )
    axes[0].plot(
        daily.index,
        daily,
        color="#176b95",
        linewidth=2.2,
        label="Strategy, after assumed costs",
    )
    axes[0].axhline(
        10000, color="#8b8b8b", linestyle="--", label="Cash reference, no interest"
    )
    axes[0].yaxis.set_major_formatter(dollars)
    daily_span = max(float(daily.max() - daily.min()), 100)
    axes[0].set_ylim(
        min(daily.min(), 10000) - daily_span * 0.12,
        max(daily.max(), 10000) + daily_span * 0.15,
    )
    axes[0].set_title(
        rf"Account value: \$10,000 → \${daily.iloc[-1]:,.2f} ({daily_report['total_return']:+.2%})",
        loc="left",
        fontsize=13,
    )
    axes[0].set_ylabel("Account value (USD)")
    axes[0].legend(loc="lower left", fontsize=10)
    axes[0].grid(alpha=0.2)
    axes[0].xaxis.set_major_locator(mdates.YearLocator(5 if args.extended else 1))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0].set_xlim(
        pd.Timestamp(daily_report["evaluation_start"]),
        pd.Timestamp(daily_report["evaluation_end"]),
    )
    years = list(daily_report["yearly_returns"])
    returns = [100 * daily_report["yearly_returns"][year] for year in years]
    axes[1].bar(
        years,
        returns,
        color=["#27825b" if v > 0 else "#ad4b46" for v in returns],
        width=0.6,
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    padding = 5.0 if args.extended else 0.8
    axes[1].set_ylim(min([0] + returns) - padding, max([0] + returns) + padding)
    axes[1].set_title("Return in each calendar year", loc="left", fontsize=13)
    axes[1].set_ylabel("Annual return (%)")
    axes[1].grid(axis="y", alpha=0.2)
    for year, value in zip(years, returns):
        axes[1].text(
            year,
            value + (0.12 if value >= 0 else -0.12),
            f"{value:+.2f}%",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=7 if args.extended else 12,
            rotation=90 if args.extended else 0,
        )
    if args.extended:
        axes[1].tick_params(axis="x", labelrotation=90, labelsize=8)
    fig.suptitle(
        f"{pair_label} · daily walk-forward test\n{daily_report['evaluation_start'][:4]}–{daily_report['evaluation_end'][:4]} · {daily_report['completed_round_trips']} round trips · {daily_report['rejected_requests']} rejected requests",
        fontsize=15,
    )
    fig.savefig(output / f"daily-pair-performance{suffix}.png", dpi=170)
    plt.close(fig)
    if args.extended:
        print(f"Saved daily-pair-performance{suffix}.png")
        return

    minute_report = json.loads((output / f"minute-pair-{args.pair}.json").read_text())
    minute = pd.read_csv(
        cache / "minute-output/equity.csv", index_col=0, parse_dates=True
    ).iloc[:, 0]
    if not minute.index.is_unique or not minute.index.is_monotonic_increasing:
        raise ValueError("Need unique ordered minute observations")
    np.testing.assert_allclose(
        minute.iloc[-1], minute_report["final_capital"], atol=1e-8
    )

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True, sharey=True)
    for ax, session in zip(axes, minute_report["sessions"]):
        date = pd.Timestamp(session["date"]).date()
        curve = minute.loc[minute.index.date == date]
        ax.plot(curve.index, curve, color="#176b95", linewidth=2)
        ax.axhline(10000, color="#8b8b8b", linestyle="--")
        minute_span = max(float(minute.max() - minute.min()), 100)
        ax.set_ylim(
            min(minute.min(), 10000) - minute_span * 0.12,
            max(minute.max(), 10000) + minute_span * 0.15,
        )
        ax.yaxis.set_major_formatter(dollars)
        ax.xaxis.set_major_locator(mdates.HourLocator(tz=ZoneInfo("America/New_York")))
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%H:%M", tz=ZoneInfo("America/New_York"))
        )
        ax.set_title(
            f"{date} · {session['completed_round_trips']} round trip · {session['rejected_requests']} rejected requests",
            loc="left",
            fontsize=12,
        )
        ax.set_ylabel("Account value (USD)")
        ax.set_xlabel("New York time · positions closed at session end")
        ax.grid(alpha=0.2)
    fig.suptitle(
        f"{pair_label} · one-minute prototype\n"
        + rf"\$10,000 → \${minute.iloc[-1]:,.2f} ({minute_report['total_return']:+.2%}) · only two test sessions",
        fontsize=15,
    )
    fig.savefig(output / f"minute-pair-performance{suffix}.png", dpi=170)
    plt.close(fig)
    print(
        f"Saved daily-pair-performance{suffix}.png and minute-pair-performance{suffix}.png"
    )


if __name__ == "__main__":
    main()
