"""Configured exploration UI; historical performance requires qualified data."""

from pathlib import Path
from typing import Any
import hashlib
import altair as alt
import streamlit as st
import pandas as pd
from partial_cointegration import fit_pair, Parameters, FitError
from partial_cointegration.simulation import simulate_pair
from partial_cointegration.signals import score_components, decisions
from partial_cointegration.execution import backtest_pairs, ExecutionConfig
from partial_cointegration.data import validate_prices
from diagnostics import pair_diagnostics, timeline_chart


def render_diagnostics(run: dict[str, Any]) -> None:
    """Present a saved run so inspecting or downloading never discards it."""
    fit, ledger = run["fit"], run["ledger"]
    frame, fills = run["frame"], run["fills"]
    entries, exits = run["entries"], run["exits"]
    st.subheader(f"{run['x']} / {run['y']} · pair diagnostics")
    st.caption(
        f"Formation: {run['formation_start']} to {run['formation_end']} · "
        f"Evaluation: {frame.index[0].date()} to {frame.index[-1].date()} · "
        f"{run['variant']} normalization. Estimates use formation data only."
    )
    metrics = st.columns(4)
    metrics[0].metric(
        "Net return", f"{(ledger.equity.iloc[-1] / ledger.config.capital - 1):.2%}"
    )
    metrics[1].metric("Maximum drawdown", f"{-frame['Drawdown %'].min():.2f}%")
    metrics[2].metric("Entry decisions", int(entries.sum()))
    metrics[3].metric(
        "Executed entries", fills.loc[fills.kind == "entry", "decision_time"].nunique()
    )
    if fit.parameters.sigma_m == 0:
        st.info(
            "The fitted relationship is a pure random walk: no temporary component was found. Staying in cash is a valid result."
        )
    elif not entries.any():
        st.info(
            "No entry threshold was crossed during evaluation; the account stayed in cash."
        )
    elif not (fills.kind == "entry").any():
        st.info(
            "Entry decisions occurred, but none executed. Review the rejection reasons below."
        )
    timeline, execution, model = st.tabs(
        ["Timeline", "Execution & costs", "Model & data"]
    )
    with timeline:
        st.caption(
            "All panels share evaluation dates. Drag to pan, scroll to zoom, double-click to reset. Symbols distinguish decisions from fills; hover for values."
        )
        with alt.data_transformers.disable_max_rows():
            st.altair_chart(
                timeline_chart(frame, fills, entries, exits, run["entry"], run["exit"]),
                use_container_width=True,
            )
        st.caption(
            "Entry = long predictor / short response. Decisions request trades; fills occur at a later available close. Scheduled liquidation closes remaining holdings at the evaluation end. Missing quotes can hide a signal marker; the fill table retains the execution."
        )
    with execution:
        cols = st.columns(4)
        cols[0].metric("Fees", f"{frame.Fees.sum():,.2f}")
        cols[1].metric("Slippage cost", f"{frame['Slippage cost'].sum():,.2f}")
        cols[2].metric("Borrow cost", f"{frame['Borrow cost'].sum():,.2f}")
        cols[3].metric("Rejected / unfilled requests", len(ledger.rejections))
        st.caption(
            "Amounts are in account currency. Fees, adverse slippage and borrow are already reflected in equity; do not subtract them again. Borrow assumes 252 daily observations per year. Dividends and corporate actions are not modeled by this explorer."
        )
        st.line_chart(frame[["Capital usage %"]])
        st.caption(
            "Usage = marked gross holdings / (positive equity × gross leverage). Entry sizing uses initial capital; after losses a new request can exceed the available margin. Short proceeds do not add capacity. End-of-row exposure excludes positions already liquidated."
        )
        st.line_chart(frame[["Cumulative costs"]])
        st.subheader("Rejection reasons")
        if ledger.rejections.empty:
            st.info("No rejected or unfilled requests.")
        else:
            st.dataframe(
                ledger.rejections.groupby("reason").size().rename("Requests"),
                use_container_width=True,
            )
            st.dataframe(ledger.rejections, use_container_width=True)
        st.caption(
            "Decision counts and executed entries need not match: pending requests, existing holdings, missing quotes and margin limits affect execution. Each fill table row is one security leg, not one pair trade."
        )
        st.subheader("Actual fills")
        st.dataframe(fills, use_container_width=True)
        st.subheader("Holdings · security units")
        st.dataframe(ledger.holdings, use_container_width=True)
    with model:
        p = fit.parameters
        st.write(
            {
                "beta": p.beta,
                "rho": p.rho,
                "sigma_M": p.sigma_m,
                "sigma_R": p.sigma_r,
                "variance_fraction": p.variance_fraction,
                "selected_model": fit.selected,
                "mode": fit.mode,
                "formation_observations": fit.n,
            }
        )
        st.caption(
            "Alpha is fixed at zero. The variance fraction is not an R-squared or profit probability. This view does not apply a calibrated significance test."
        )
        st.dataframe(frame, use_container_width=True)
        if frame["Stale valuation"].any():
            st.warning(
                "Some holdings use the last observed quote for valuation. See Stale valuation in the table; missing quotes are not executable prices."
            )
    for label, data, filename in [
        ("Download diagnostics", frame, "pair-diagnostics.csv"),
        ("Download fills", fills, "pair-fills.csv"),
        ("Download rejections", ledger.rejections, "pair-rejections.csv"),
    ]:
        st.download_button(
            label, data.to_csv(index=True).encode("utf-8"), filename, "text/csv"
        )


def run_streamlit_app() -> None:
    """Fit formation prices and inspect one pair's evaluation ledger."""
    st.set_page_config(page_title="Pair diagnostics", layout="wide")
    st.title("Partial cointegration — research explorer")
    source = st.selectbox(
        "Data source", ["Synthetic demonstration", "Historical CSV/parquet"]
    )
    if source == "Synthetic demonstration":
        st.info(
            "Synthetic data only. Results do not describe historical investment performance."
        )
        pair = simulate_pair(600, Parameters(1, 0.6, 1, 1), seed=20261025)
        prices = pd.DataFrame(
            {"X": pair.x, "Y": pair.y}, index=pd.bdate_range("2020-01-01", periods=600)
        )
    else:
        path = st.text_input("Path to date-indexed price CSV/parquet")
        if not path:
            st.info(
                "Choose data with a documented adjustment basis and point-in-time security mapping."
            )
            return
        try:
            file = Path(path)
            if file.suffix.lower() == ".csv":
                prices = validate_prices(pd.read_csv(file, index_col="date"))
            elif file.suffix.lower() in (".parquet", ".pq"):
                prices = validate_prices(pd.read_parquet(file))
            else:
                raise ValueError("Choose a CSV or parquet price file")
        except (ValueError, FileNotFoundError, KeyError) as exc:
            st.error(str(exc))
            return
    cols = list(prices.columns)
    if len(prices) < 30:
        st.error("Need at least 30 observations for formation and evaluation.")
        return
    if len(cols) < 2:
        st.error("Need at least two security columns.")
        return
    period = st.date_input(
        "Research period",
        value=(prices.index[0].date(), prices.index[-1].date()),
        min_value=prices.index[0].date(),
        max_value=prices.index[-1].date(),
    )
    if not isinstance(period, tuple) or len(period) != 2:
        st.info("Choose both the start and end of the research period.")
        return
    prices = prices.loc[
        (prices.index.date >= period[0]) & (prices.index.date <= period[1])
    ]
    if len(prices) < 30:
        st.error("Need at least 30 observations in the selected research period.")
        return
    x = st.selectbox("Predictor security", cols, index=0)
    y = st.selectbox("Response security", cols, index=1)
    split = st.slider(
        "Formation observations",
        min_value=20,
        max_value=max(20, len(prices) - 10),
        value=min(400, len(prices) - 10),
    )
    variant = st.selectbox("Normalization", ["formation", "rolling"])
    window = st.slider("Rolling window", min_value=2, max_value=100, value=30)
    entry = st.number_input("Entry score", value=1.0)
    exit = st.number_input("Exit score", value=-0.5)
    with st.expander("Capital & daily execution assumptions"):
        capital = st.number_input("Initial capital", min_value=1.0, value=10000.0)
        fees = st.number_input("Fee per fill (%)", min_value=0.0, value=0.1)
        slippage = st.number_input(
            "Slippage per fill (%)", min_value=0.0, max_value=99.0, value=0.1
        )
        borrow = st.number_input("Annual short borrow (%)", min_value=0.0, value=0.0)
    st.caption(
        "Daily-close explorer. Split the selected period into formation and evaluation with the slider above. Use the intraday experiment runner for minute data."
    )
    signature = (
        hashlib.sha256(
            pd.util.hash_pandas_object(prices, index=True).values.tobytes()
        ).hexdigest(),
        tuple(cols),
        source,
        x,
        y,
        split,
        variant,
        window,
        entry,
        exit,
        capital,
        fees,
        slippage,
        borrow,
    )
    if st.button("Run research example"):
        try:
            if x == y:
                raise ValueError("Choose two distinct securities")
            tr = prices.iloc[:split]
            te = prices.iloc[split:]
            fit = fit_pair(tr[x], tr[y])
            p = fit.parameters
            if p.beta <= 0:
                raise ValueError("Trading currently requires positive beta")
            if p.sigma_m == 0:
                score = pd.DataFrame(float("nan"), index=te.index, columns=["pair"])
            else:
                result = score_components(
                    tr[y] - p.beta * tr[x],
                    te[y] - p.beta * te[x],
                    p,
                    variant=variant,
                    window=window,
                )
                score = result["testing_score"].rename("pair").to_frame()
            opens, closes = decisions(score, entry, exit)
            ledger = backtest_pairs(
                te[[x, y]],
                {"pair": (x, y, p.beta)},
                opens,
                closes,
                config=ExecutionConfig(
                    capital=capital,
                    fees=fees / 100,
                    slippage=slippage / 100,
                    annual_borrow=borrow / 100,
                ),
            )
            frame, fills = pair_diagnostics(prices, x, y, p, score["pair"], ledger)
            st.session_state["pair_run"] = dict(
                signature=signature,
                fit=fit,
                ledger=ledger,
                frame=frame,
                fills=fills,
                entries=opens["pair"],
                exits=closes["pair"],
                x=x,
                y=y,
                entry=entry,
                exit=exit,
                variant=variant,
                formation_start=tr.index[0].date(),
                formation_end=tr.index[-1].date(),
            )
        except (ValueError, FitError) as exc:
            st.error(str(exc))
            return
    saved = st.session_state.get("pair_run")
    if saved and saved["signature"] == signature:
        render_diagnostics(saved)
    elif saved:
        st.info("Inputs changed. Run research example to refresh diagnostics.")


if __name__ == "__main__":
    run_streamlit_app()
