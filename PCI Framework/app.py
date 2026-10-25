"""Configured exploration UI; historical performance requires qualified data."""

from pathlib import Path
import streamlit as st
import pandas as pd
from partial_cointegration import fit_pair, Parameters, FitError
from partial_cointegration.simulation import simulate_pair
from partial_cointegration.signals import score_components, decisions
from partial_cointegration.execution import backtest_pairs, ExecutionConfig
from PCIPipeline import Definition


def run_streamlit_app():
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
            prices = Definition.f_load_adjPrice(Path(path))
        except (ValueError, FileNotFoundError, KeyError) as exc:
            st.error(str(exc))
            return
    cols = list(prices.columns)
    if len(cols) < 2:
        st.error("Need at least two security columns.")
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
                config=ExecutionConfig(),
            )
        except (ValueError, FitError) as exc:
            st.error(str(exc))
            return
        st.write(
            {
                "beta": p.beta,
                "rho": p.rho,
                "sigma_M": p.sigma_m,
                "sigma_R": p.sigma_r,
                "mode": fit.mode,
                "alpha": "fixed at zero; not estimated",
            }
        )
        st.caption(
            "Positive-beta spread; decisions execute at a later observed close. One shared committed-capital ledger. No calibrated significance or profitability claim."
        )
        st.write(ledger.stats())
        st.line_chart(ledger.equity)
        st.dataframe(ledger.orders)
        st.dataframe(ledger.rejections)


if __name__ == "__main__":
    run_streamlit_app()
