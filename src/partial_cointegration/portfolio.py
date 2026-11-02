"""Cash-funded overlapping cohorts and dependence-aware descriptive inference."""

from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import norm


def aggregate_cohorts(
    cohorts: dict[str, Any], *, max_active: int = 6
) -> dict[str, Any]:
    """Allocate from available cash before observing each date's cohort return.

    A new cohort receives min(previous equity/max_active, available cash).
    Matured cohorts return their closing value to cash after that date's starts.
    Curves are unit-capital NAVs from already causal, fully liquidated ledgers.
    Forward-filled NAVs are valuation only, never execution prices.
    """
    if not cohorts or max_active < 1:
        raise ValueError("Need cohorts and a positive simultaneous-capital limit")
    for key, curve in cohorts.items():
        if (
            len(curve) < 2
            or not curve.index.is_unique
            or not curve.index.is_monotonic_increasing
            or not np.isfinite(curve.to_numpy()).all()
            or not np.isclose(curve.iloc[0], 1, atol=1e-10, rtol=0)
        ):
            raise ValueError(f"{key}: need ordered finite NAV starting at one")
    calendar = sorted(set().union(*(set(curve.index) for curve in cohorts.values())))
    for day in calendar:
        if (
            sum(curve.index[0] <= day <= curve.index[-1] for curve in cohorts.values())
            > max_active
        ):
            raise ValueError("Cohort schedule exceeds the prespecified active limit")
    cash = 1.0
    previous_equity = 1.0
    active: dict[str, float] = {}
    last: dict[str, float] = {}
    records = []
    allocations = []
    for day in calendar:
        for key in sorted(cohorts):
            if cohorts[key].index[0] == day:
                amount = min(max(previous_equity, 0) / max_active, max(cash, 0))
                cash -= amount
                active[key] = amount
                last[key] = 1.0
                allocations.append(
                    {
                        "cohort": key,
                        "start": day,
                        "capital": amount,
                        "funding": "prior-close equity and available cash",
                    }
                )
        stale = False
        for key in list(active):
            curve = cohorts[key]
            if day in curve.index:
                last[key] = float(curve.loc[day])
            else:
                stale = True
            if day == curve.index[-1]:
                cash += active.pop(key) * last[key]
        equity = cash + sum(amount * last[key] for key, amount in active.items())
        records.append(
            {
                "date": day,
                "equity": equity,
                "cash": cash,
                "active_cohorts": len(active),
                "stale_nav_valuation": stale,
            }
        )
        previous_equity = equity
    table = pd.DataFrame(records).set_index("date")
    return {
        "equity": table["equity"],
        "cash": table["cash"],
        "ledger": table,
        "allocations": pd.DataFrame(allocations),
    }


def hac_mean_interval(
    returns: Any, *, lag: int = 6, level: float = 0.95
) -> dict[str, Any]:
    """Asymptotic Bartlett/Newey-West interval; not a finite-sample guarantee."""
    values = np.asarray(returns, dtype=float)
    if (
        values.ndim != 1
        or not np.isfinite(values).all()
        or len(values) <= max(20, lag + 2)
        or lag < 0
        or not 0 < level < 1
    ):
        raise ValueError(
            "Need >max(20,lag+2) finite returns and a valid interval level"
        )
    n = len(values)
    mean = float(values.mean())
    errors = values - mean
    variance = float(errors @ errors / n)
    for k in range(1, lag + 1):
        variance += 2 * (1 - k / (lag + 1)) * float(errors[k:] @ errors[:-k] / n)
    if variance < 0:
        raise ValueError("Numerically negative HAC variance; inspect conditioning")
    se = float(np.sqrt(variance / n))
    critical = float(norm.ppf((1 + level) / 2))
    return {
        "mean": mean,
        "standard_error": se,
        "interval": [mean - critical * se, mean + critical * se],
        "lag": lag,
        "n": n,
        "method": "asymptotic Bartlett HAC mean interval; selection and model risk remain",
    }
