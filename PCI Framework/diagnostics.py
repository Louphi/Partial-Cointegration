"""Read-only pair diagnostics derived from the execution ledger."""

from typing import Any

import altair as alt
import numpy as np
import pandas as pd

from partial_cointegration import Parameters, filter_residuals
from partial_cointegration.execution import LedgerResult


def pair_diagnostics(
    prices: Any,
    x: str,
    y: str,
    parameters: Parameters,
    score: Any,
    ledger: LedgerResult,
) -> tuple[Any, Any]:
    """Return evaluation observations and fills without modifying model outputs.

    Prices include formation history so the component state continues naturally.
    Costs are positive cash drags; slippage is already embedded in fill prices.
    """
    spread = prices[y] - parameters.beta * prices[x]
    filtered = filter_residuals(spread.to_numpy(), parameters, timestamps=prices.index)
    frame = (
        pd.DataFrame(
            {
                f"Price: {x}": prices[x],
                f"Price: {y}": prices[y],
                "Spread": spread,
                "Temporary M": filtered.m,
                "Permanent R": filtered.r,
            }
        )
        .loc[ledger.equity.index]
        .copy()
    )
    frame["Score"] = score
    frame["Equity"] = ledger.equity
    frame["Cash"] = ledger.cash
    marks = prices[[x, y]].ffill().loc[frame.index]
    gross = (ledger.holdings * marks).abs().sum(axis=1)
    frame["Gross exposure"] = gross
    frame["Gross limit"] = ledger.equity.clip(lower=0) * ledger.config.gross_leverage
    frame["Capital usage %"] = gross.div(frame["Gross limit"].replace(0, np.nan)) * 100
    high = ledger.equity.cummax().clip(lower=ledger.config.capital)
    frame["Drawdown %"] = (ledger.equity / high - 1) * 100
    frame["Stale valuation"] = ledger.stale_valuation
    fills = ledger.orders.copy()
    if len(fills):
        fills["Quoted close"] = [
            prices.loc[t, s] for t, s in zip(fills.fill_time, fills.security)
        ]
        fills["Slippage cost"] = (
            fills.units * (fills.price - fills["Quoted close"])
        ).where(fills.kind != "settlement", 0.0)
    else:
        fills["Quoted close"] = pd.Series(dtype=float)
        fills["Slippage cost"] = pd.Series(dtype=float)
    frame["Fees"] = (
        fills.groupby("fill_time")
        .fees.sum()
        .reindex(frame.index, fill_value=0.0)
        .astype(float)
    )
    frame["Slippage cost"] = (
        fills.groupby("fill_time")["Slippage cost"]
        .sum()
        .reindex(frame.index, fill_value=0.0)
        .astype(float)
    )
    carried = ledger.holdings.shift(1, fill_value=0).clip(upper=0).abs()
    frame["Borrow cost"] = (carried * marks).sum(axis=1) * (
        ledger.config.annual_borrow / ledger.config.borrow_observations_per_year
    )
    frame["Cumulative costs"] = (
        frame[["Fees", "Slippage cost", "Borrow cost"]].sum(axis=1).cumsum()
    )
    return frame, fills


def timeline_chart(
    frame: Any, fills: Any, entries: Any, exits: Any, entry: float, exit: float
) -> Any:
    """Build aligned interactive panels with explicit decision/fill markers."""
    data = frame.rename_axis("Date").reset_index()

    def lines(columns: list[str], title: str) -> Any:
        long = data.melt(
            "Date", value_vars=columns, var_name="Series", value_name="Value"
        )
        return (
            alt.Chart(long)
            .mark_line()
            .encode(
                x=alt.X("Date:T", title=None),
                y=alt.Y("Value:Q", scale=alt.Scale(zero=False), title=None),
                color=alt.Color("Series:N", legend=alt.Legend(orient="top")),
                strokeDash="Series:N",
                tooltip=["Date:T", "Series:N", alt.Tooltip("Value:Q", format=".4f")],
            )
            .properties(height=150, title=title)
        )

    price_columns = list(frame.columns[:2])
    prices = lines(price_columns, "Prices · observed closes")
    components = lines(
        ["Spread", "Temporary M", "Permanent R"],
        "Spread decomposition · y − βx = M + R",
    )
    signal = lines(["Score"], "Signal · decisions and actual executions")
    thresholds = (
        alt.Chart(
            pd.DataFrame(
                {"Level": [entry, exit], "Rule": ["Entry threshold", "Exit threshold"]}
            )
        )
        .mark_rule(strokeDash=[5, 5])
        .encode(y="Level:Q", tooltip=["Rule:N", "Level:Q"])
    )
    events = []
    for flags, label in [(entries, "Entry decision"), (exits, "Exit decision")]:
        for date in flags.index[flags]:
            events.append(
                {"Date": date, "Score": frame.loc[date, "Score"], "Event": label}
            )
    for row in fills.drop_duplicates(["fill_time", "kind"]).itertuples():
        events.append(
            {
                "Date": row.fill_time,
                "Score": frame.loc[row.fill_time, "Score"],
                "Event": row.kind.replace("_", " ") + " fill",
            }
        )
    markers = (
        alt.Chart(pd.DataFrame(events, columns=["Date", "Score", "Event"]))
        .mark_point(size=85, filled=True)
        .encode(
            x="Date:T",
            y="Score:Q",
            shape=alt.Shape("Event:N", legend=alt.Legend(orient="bottom")),
            tooltip=["Date:T", "Score:Q", "Event:N"],
            color=alt.value("#b45309"),
        )
    )
    equity = lines(["Equity", "Cash"], "Account · equity and cash")
    drawdown = lines(
        ["Drawdown %"], "Drawdown · from initial capital or subsequent peak (%)"
    )
    capital = lines(
        ["Gross exposure", "Gross limit"],
        "Capacity · actual gross exposure and margin limit",
    )
    zoom = alt.selection_interval(bind="scales", encodings=["x"])
    return (
        alt.vconcat(
            prices, components, signal + thresholds + markers, equity, drawdown, capital
        )
        .resolve_scale(x="shared", color="independent", strokeDash="independent")
        .add_params(zoom)
    )
