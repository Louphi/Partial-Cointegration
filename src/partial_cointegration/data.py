"""Price alignment, interval membership, and point-in-time splitting."""

from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from .model import FloatArray


def paired_arrays(
    x: ArrayLike, y: ArrayLike, minimum: int = 4
) -> tuple[FloatArray, FloatArray]:
    """Validate labeled alignment before converting positional numerical arrays."""
    if isinstance(x, pd.Series) != isinstance(y, pd.Series):
        raise ValueError("Pass either two indexed Series or two positional arrays")
    if isinstance(x, pd.Series) and isinstance(y, pd.Series):
        if (
            not x.index.equals(y.index)
            or not x.index.is_unique
            or not x.index.is_monotonic_increasing
        ):
            raise ValueError("Series indices must match and be unique and increasing")
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape or len(a) < minimum:
        raise ValueError(
            f"Expected matching one-dimensional series with at least {minimum} rows"
        )
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Training series must contain only finite observations")
    return a.copy(), b.copy()


def validate_prices(prices: Any, allow_missing: bool = True) -> Any:
    """Require unique ordered dates and strictly positive observed prices."""
    out = prices.copy()
    out.index = pd.to_datetime(out.index)
    if (
        not out.index.is_unique
        or not out.index.is_monotonic_increasing
        or out.index.hasnans
    ):
        raise ValueError("Price dates must be valid, unique and increasing")
    if not out.columns.is_unique:
        raise ValueError("Price/security columns must be unique")
    values = out.to_numpy(dtype=float)
    if np.isinf(values).any() or (values[np.isfinite(values)] <= 0).any():
        raise ValueError("Observed prices must be positive and finite")
    if not allow_missing and np.isnan(values).any():
        raise ValueError("Missing observations are not supported here")
    return out


def split_prices(data: Any, year: int) -> tuple[Any, Any]:
    """Four-year training, six-month evaluation; future completeness is unused."""
    prices = validate_prices(data)
    start = pd.Timestamp(year=int(year), month=1, day=1)
    training = prices.loc[
        (prices.index >= start - pd.DateOffset(years=4)) & (prices.index < start)
    ].dropna(how="all")
    training = training.dropna(axis=1, how="any")
    testing = prices.loc[
        (prices.index >= start) & (prices.index < start + pd.DateOffset(months=6)),
        training.columns,
    ]
    return training, testing


def members_at(intervals: Any, timestamp: Any, column: str = "co_tic") -> list[str]:
    """Membership intervals are inclusive; never replace open ends with today."""
    start = pd.to_datetime(intervals["from"], errors="raise")
    end = pd.to_datetime(intervals["thru"], errors="raise")
    if start.isna().any() or ((end < start) & end.notna()).any():
        raise ValueError("Invalid membership interval")
    day = pd.Timestamp(timestamp)
    selected = intervals.loc[(start <= day) & (end.isna() | (end >= day)), column]
    return sorted(str(v) for v in selected.dropna().unique())
