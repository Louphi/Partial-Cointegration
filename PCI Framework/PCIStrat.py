"""Compatibility strategy facade using continuous signals and the shared ledger."""

from types import SimpleNamespace
import numpy as np
import pandas as pd
from PCIPipeline import Computation
from partial_cointegration import Parameters, filter_residuals
from partial_cointegration.execution import backtest_pairs, ExecutionConfig
from partial_cointegration.signals import decisions


class vbt_strategyPCI:
    @staticmethod
    def f_pre_vectorBT(pairListSelected, estimatorData, pairData, inSample, outSample):
        train = pd.DataFrame(
            index=inSample.index, columns=pairListSelected, dtype=float
        )
        test = pd.DataFrame(
            index=outSample.index, columns=pairListSelected, dtype=float
        )
        ht = train.copy()
        he = test.copy()
        valid_train = train.copy().astype(bool)
        valid_test = test.copy().astype(bool)
        if inSample.empty or outSample.empty:
            raise ValueError("Need nonempty formation and evaluation samples")
        for pair in pairListSelected:
            x, y = Computation.get_stock1_stock2(pair, pairData)
            alpha, beta, rho, sm, sr = estimatorData.loc[pair].to_list()
            p = Parameters(beta, rho, sm, sr, alpha)
            tr = filter_residuals(
                (inSample[y] - alpha - beta * inSample[x]).to_numpy(),
                p,
                timestamps=inSample.index,
            )
            te = filter_residuals(
                (outSample[y] - alpha - beta * outSample[x]).to_numpy(),
                p,
                state=tr.final_state,
                timestamps=outSample.index,
            )
            train[pair] = tr.m
            test[pair] = te.m
            # No decisions on a missing observed spread, despite predicted states.
            valid_train[pair] = inSample[[x, y]].notna().all(axis=1)
            valid_test[pair] = outSample[[x, y]].notna().all(axis=1)
            ht[pair] = beta / inSample[y]
            he[pair] = beta / outSample[y]
        stocks = Computation.get_stock_list(pairData, pairListSelected)
        return {
            "Mt": {"inSample": train, "outSample": test},
            "ValidQuotes": {"inSample": valid_train, "outSample": valid_test},
            "hedgeS1": {"inSample": ht, "outSample": he},
            "Price": {"inSample": inSample[stocks], "outSample": outSample[stocks]},
        }

    @staticmethod
    def custom_indicator(Mt, window=20):
        return pd.DataFrame(Mt).apply(
            lambda s: Computation.rolling_z_score(s, window), axis=0
        )

    @staticmethod
    def f_get_Mt_normalize(Mt, window):
        return SimpleNamespace(z_score=vbt_strategyPCI.custom_indicator(Mt, window))

    @staticmethod
    def f_get_signals(strategy, entry_threshold, exit_threshold):
        return decisions(strategy.z_score, entry_threshold, exit_threshold)

    @staticmethod
    def f_pre_backtest(
        dict_output, window, entry_threshold, exit_threshold, *, variant="rolling"
    ):
        if window < 2:
            raise ValueError("window must be at least 2")
        tr = dict_output["Mt"]["inSample"]
        te = dict_output["Mt"]["outSample"]
        if variant == "rolling":
            combined = pd.concat([tr, te])
            scores = vbt_strategyPCI.custom_indicator(combined, window)
            ztr = scores.iloc[: len(tr)]
            zte = scores.iloc[len(tr) :]
        elif variant == "formation":
            scale = tr.std(ddof=1).replace(0, np.nan)
            ztr = tr / scale
            zte = te / scale
        else:
            raise ValueError("Choose rolling or formation normalization")
        out = {}
        for name, z in [("inSample", ztr), ("outSample", zte)]:
            if "ValidQuotes" in dict_output:
                z = z.where(dict_output["ValidQuotes"][name])
            strategy = SimpleNamespace(z_score=z)
            opens, closes = decisions(z, entry_threshold, exit_threshold)
            out[name] = {
                "clean_upper_crossed": opens,
                "clean_lower_crossed": closes,
                "Mt": strategy,
            }
        return out

    @staticmethod
    def f_backtest(
        dict_pre_backtest, type_back, hedgeS1, price, pairBacktested, pairData
    ):
        x, y = Computation.get_stock1_stock2(pairBacktested, pairData)
        beta_values = (hedgeS1[pairBacktested] * price[y]).dropna()
        if beta_values.empty:
            raise ValueError("No identified hedge ratio")
        beta = float(beta_values.iloc[0])
        if not np.allclose(beta_values, beta):
            raise ValueError(
                "Time-varying hedge ratios require an explicit rebalance policy"
            )
        signal = dict_pre_backtest[type_back]
        ledger = backtest_pairs(
            price[[x, y]],
            {pairBacktested: (x, y, beta)},
            signal["clean_upper_crossed"][[pairBacktested]],
            signal["clean_lower_crossed"][[pairBacktested]],
            config=ExecutionConfig(),
        )
        return {
            "pf": ledger,
            "Price": {"X1": price[x], "X2": price[y]},
            "positions": {
                "entries": signal["clean_upper_crossed"][pairBacktested],
                "exits": signal["clean_lower_crossed"][pairBacktested],
            },
            "sizing": ledger.holdings,
            "ledger": ledger,
        }
