"""Compatibility facade. All active mathematics lives in partial_cointegration."""

from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from partial_cointegration import (
    Parameters,
    fit_pair,
    FitOptions,
    FitError,
    filter_residuals,
)
from partial_cointegration.estimation import fit_residuals
from partial_cointegration.data import paired_arrays, split_prices, validate_prices


class DataTable:
    def __init__(self, pairName, constituents):
        if len(pairName) != len(constituents):
            raise ValueError("Pair labels and values must match")
        self.dict_data = dict(zip(pairName, constituents))

    def getData(self):
        return self.dict_data

    def setData(self):
        return self.dict_data


class Definition:
    @staticmethod
    def f_load_adjPrice(filepath, *, trusted_pickle=False):
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Historical price data unavailable: {path}")
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path, index_col="date")
        elif path.suffix.lower() == ".parquet":
            frame = pd.read_parquet(path)
        elif path.suffix.lower() in (".pkl", ".pickle") and trusted_pickle:
            frame = pd.read_pickle(path)
            if "date" in frame:
                frame = frame.set_index("date")
        else:
            raise ValueError(
                "Use date-indexed CSV/parquet or explicitly trusted_pickle=True"
            )
        return validate_prices(frame)

    @staticmethod
    def f_selectTicker(tickerList, adjPrice):
        return adjPrice.loc[:, adjPrice.columns.isin(tickerList)].copy()

    f_insample_outsample = staticmethod(split_prices)

    @staticmethod
    def pairDatabase(consituentList):
        # One deterministic orientation per unordered pair avoids duplicated risk.
        rows = list(combinations(sorted(set(consituentList)), 2))
        frame = pd.DataFrame(rows, columns=["Stock 1", "Stock 2"])
        frame["pair"] = frame["Stock 1"] + "/" + frame["Stock 2"]
        return frame


class Computation:
    last_batch_errors = []

    def __init__(self, X1, X2, *, mode="exact", seed=0):
        self.X1, self.X2 = paired_arrays(X1, X2)
        self.mode, self.seed = mode, seed
        self.last_fit = None

    @staticmethod
    def Rsquare(sigmaM, sigmaR, rho):
        return Parameters(0, rho, sigmaM, sigmaR).variance_fraction

    def fit_pci(self, tol=1e-10, LRT_mode=True):
        self.last_fit = fit_pair(
            self.X1,
            self.X2,
            mode=self.mode,
            options=FitOptions(seed=self.seed, tolerance=tol),
        )
        return self.last_fit.legacy_tuple(LRT_mode)

    def fit_mle(self, W, tol=1e-10):
        fitted = fit_residuals(
            W, mode=self.mode, options=FitOptions(seed=self.seed, tolerance=tol)
        )
        p = fitted.parameters
        return np.array([p.rho, p.sigma_m, p.sigma_r])

    def lagvar_estimate_par(self, W):
        w = np.asarray(W, dtype=float)
        if w.ndim != 1 or len(w) < 4 or not np.isfinite(w).all():
            raise ValueError("Need four finite residual observations")
        v1, v2, v3 = [np.var(w[k:] - w[:-k]) for k in (1, 2, 3)]
        if v1 <= np.finfo(float).eps * max(1.0, np.mean(np.diff(w) ** 2)):
            raise FitError("Degenerate lag variances; no moment initialization")
        fallback = (0.0, np.sqrt(v1 / 2), np.sqrt(v1 / 2))
        denominator = 2 * v1 - v2
        if abs(denominator) <= np.finfo(float).eps * max(v1, v2):
            return fallback
        rho = -(v1 - 2 * v2 + v3) / denominator
        if not -1 < rho < 1:
            return fallback
        qm = 0.5 * (rho + 1) / (rho - 1) * (v2 - 2 * v1)
        qr = 0.5 * (v2 - 2 * qm)
        if min(qm, qr) < 0 or qm + qr <= 0:
            return fallback
        return rho, np.sqrt(qm), np.sqrt(qr)

    def fit_ols_on_diff(self):
        dx, dy = np.diff(self.X1), np.diff(self.X2)
        if dx @ dx == 0:
            raise FitError("Constant predictor")
        return float(dx @ dy / (dx @ dx))

    def kalman_estimate(self, W, rho, sigma_M, sigma_R):
        result = filter_residuals(
            W, Parameters(0, rho, sigma_M, sigma_R), mode=self.mode
        )
        errors = result.errors.copy()
        errors[0] = 0.0
        return result.m, result.r, errors

    def calc_log_like(self, W, rho, sigma_M, sigma_R):
        return filter_residuals(
            W, Parameters(0, rho, sigma_M, sigma_R), mode=self.mode
        ).log_likelihood

    def f_to_min_par(self, x_i, W):
        return -self.calc_log_like(W, *x_i)

    def f_to_min_pci(self, x_i, X1, X2):
        if len(x_i) == 5:
            alpha, beta, rho, sm, sr = x_i
        elif len(x_i) == 3:
            alpha, beta, sr = x_i
            rho = sm = 0.0
        else:
            raise ValueError("Expected 3 or 5 parameters")
        return -self.calc_log_like(
            np.asarray(X2) - beta * np.asarray(X1) - alpha, rho, sm, sr
        )

    @staticmethod
    def f_get_elligibility(Stock1, Stock2, inSample):
        model = Computation(inSample[Stock1], inSample[Stock2])
        out = model.fit_pci()
        alpha, beta, rho, sm, sr, _, _, lrt = out
        return {
            "elligibility": [rho, Computation.Rsquare(sm, sr, rho), lrt],
            "estimators": [alpha, beta, rho, sm, sr],
        }

    @staticmethod
    def f_compute_estimate(pairName, inSample, pairData):
        labels = list(pairName)
        eligible = np.full((len(labels), 3), np.nan)
        est = np.full((len(labels), 5), np.nan)
        mapping = pairData.set_index("pair")
        errors = []
        for i, label in enumerate(labels):
            try:
                row = mapping.loc[label]
                result = Computation.f_get_elligibility(
                    row["Stock 1"], row["Stock 2"], inSample
                )
                eligible[i] = result["elligibility"]
                est[i] = result["estimators"]
            except (ValueError, KeyError, RuntimeError) as exc:
                errors.append(
                    {"pair": str(label), "error": str(exc), "type": type(exc).__name__}
                )
        Computation.last_batch_errors = errors
        return eligible, est

    @staticmethod
    def f_create_dataframe(elligibilityData, estimationData, pairName):
        frame = pd.DataFrame(
            elligibilityData,
            index=list(pairName),
            columns=["rho", "Rsquare", "ll_ratio"],
        )
        frame = frame.loc[
            np.isfinite(frame).all(axis=1) & (frame.rho > 0.5) & (frame.Rsquare > 0.5)
        ]
        estimators = pd.DataFrame(
            estimationData,
            index=list(pairName),
            columns=["alpha_hat", "beta_hat", "rho_hat", "sigma_M_hat", "sigma_R_hat"],
        )
        return frame.sort_values("ll_ratio"), estimators

    @staticmethod
    def f_pairSelected(frame, n_keep):
        return frame.index[: max(0, int(n_keep))].tolist()

    @staticmethod
    def get_stock_list(pairData, col):
        return sorted(
            set(
                pairData.loc[pairData["pair"].isin(col), ["Stock 1", "Stock 2"]]
                .to_numpy()
                .ravel()
            )
        )

    @staticmethod
    def get_stock1_stock2(pair, pairData):
        return tuple(pairData.set_index("pair").loc[pair, ["Stock 1", "Stock 2"]])

    @staticmethod
    def get_stock_price(df_price, stockList):
        return df_price[stockList[0]], df_price[stockList[1]]

    @staticmethod
    def f_create_Mt(pairListSelected, estimatorData, pairData, stock_price):
        mt = pd.DataFrame(
            index=stock_price.index, columns=pairListSelected, dtype=float
        )
        hedge = mt.copy()
        for pair in pairListSelected:
            x, y = Computation.get_stock1_stock2(pair, pairData)
            alpha, beta, rho, sm, sr = estimatorData.loc[pair].to_list()
            residual = stock_price[y] - alpha - beta * stock_price[x]
            p = Parameters(beta, rho, sm, sr, alpha)
            mt[pair] = filter_residuals(
                residual.to_numpy(), p, timestamps=stock_price.index
            ).m
            hedge[pair] = beta / stock_price[y]
        return mt, hedge

    @staticmethod
    def rolling_z_score(series, window):
        if window < 2:
            raise ValueError("window must be at least 2")
        values = pd.Series(series, copy=False)
        roll = values.rolling(window)
        return ((values - roll.mean()) / roll.std(ddof=1).replace(0, np.nan)).to_numpy()

    @staticmethod
    def percentile_zscore(value, data):
        values = np.asarray(data)
        valid = values[np.isfinite(values)]
        if len(valid) < 2:
            raise ValueError("Need >=2 reference observations")
        return float(2 * np.sum(valid < value) / (len(valid) - 1) - 1)

    @staticmethod
    def calculate_zscores_historic(historical_data, initial_in_sample_size=2):
        if initial_in_sample_size < 2:
            raise ValueError("Need >=2 baseline observations")
        out = historical_data.copy() * np.nan
        sample = []
        for label, value in historical_data.items():
            if pd.isna(value):
                continue
            sample.append(value)
            if len(sample) >= initial_in_sample_size:
                out.loc[label] = Computation.percentile_zscore(value, sample)
        return out
