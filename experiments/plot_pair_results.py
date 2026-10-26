"""Plot saved backtest ledgers; does not refit or select strategy parameters."""

import json
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
    root = Path(__file__).resolve().parents[1]
    output = root / "docs/results"
    daily_report = json.loads((output / "longer-pair-goog-googl.json").read_text())
    minute_report = json.loads((output / "minute-pair-goog-googl.json").read_text())
    daily = pd.read_csv(
        root / ".test-deps/longer-real-pair/output/equity.csv",
        index_col=0,
        parse_dates=True,
    ).iloc[:, 0]
    minute = pd.read_csv(
        root / ".test-deps/longer-real-pair/minute-output/equity.csv",
        index_col=0,
        parse_dates=True,
    ).iloc[:, 0]
    for curve, report in [(daily, daily_report), (minute, minute_report)]:
        if not curve.index.is_unique or not curve.index.is_monotonic_increasing:
            raise ValueError("Need unique ordered observations")
        np.testing.assert_allclose(curve.iloc[-1], report["final_capital"], atol=1e-8)
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
    axes[0].set_ylim(9100, 10120)
    axes[0].set_title(
        r"Account value: \$10,000 → \$9,234.47 (−7.66%)", loc="left", fontsize=13
    )
    axes[0].set_ylabel("Account value (USD)")
    axes[0].legend(loc="lower left", fontsize=10)
    axes[0].grid(alpha=0.2)
    axes[0].xaxis.set_major_locator(mdates.YearLocator())
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0].set_xlim(pd.Timestamp("2019-01-01"), pd.Timestamp("2023-12-31"))
    years = list(daily_report["yearly_returns"])
    returns = [100 * daily_report["yearly_returns"][year] for year in years]
    axes[1].bar(years, returns, color="#ad4b46", width=0.6)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylim(min(returns) - 0.8, 0.3)
    axes[1].set_title("Return in each calendar year", loc="left", fontsize=13)
    axes[1].set_ylabel("Annual return (%)")
    axes[1].grid(axis="y", alpha=0.2)
    for year, value in zip(years, returns):
        axes[1].text(
            year, value - 0.12, f"{value:.2f}%", ha="center", va="top", fontsize=12
        )
    fig.suptitle(
        "GOOG / GOOGL · daily walk-forward test\n2019–2023 · 19 round trips · 56 rejected requests",
        fontsize=15,
    )
    fig.savefig(output / "daily-pair-performance.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True, sharey=True)
    for ax, session in zip(axes, minute_report["sessions"]):
        date = pd.Timestamp(session["date"]).date()
        curve = minute.loc[minute.index.date == date]
        ax.plot(curve.index, curve, color="#176b95", linewidth=2)
        ax.axhline(10000, color="#8b8b8b", linestyle="--")
        ax.set_ylim(9900, 10015)
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
        "GOOG / GOOGL · one-minute prototype\n"
        + r"\$10,000 → \$9,921.84 (−0.78%) · only two test sessions",
        fontsize=15,
    )
    fig.savefig(output / "minute-pair-performance.png", dpi=170)
    plt.close(fig)
    print("Saved daily-pair-performance.png and minute-pair-performance.png")


if __name__ == "__main__":
    main()
