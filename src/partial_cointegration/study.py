"""Prespecified experiment variants and training-only selection orchestration."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal, TypedDict
import numpy as np
import pandas as pd
from .data import members_at, validate_prices
from .estimation import fit_pair, FitOptions
from .model import FitError, FitResult
from .signals import score_components, decisions
from .execution import backtest_pairs, ExecutionConfig


class Candidate(TypedDict):
    pair: str
    x: str
    y: str
    fit: FitResult
    ratio: float
    beta: float
    rho: float
    variance_fraction: float
    training_rank: float


@dataclass(frozen=True)
class StudyConfig:
    """Paper-inspired reference and Nasdaq extension are explicitly separate."""

    name: str = "nasdaq_extension"
    mode: Literal["exact", "legacy"] = "exact"
    normalization: Literal["formation", "rolling"] = "rolling"
    window: int = 30
    entry: float = 1.0
    exit: float = -0.5
    keep: int = 10
    ranking_fraction: float = 0.05
    rho_min: float = 0.5
    variance_min: float = 0.5
    same_sector: bool = True
    seed: int = 20261025
    fees: float = 0.001
    slippage: float = 0.001

    def __post_init__(self) -> None:
        if (
            not np.isfinite(
                [
                    self.entry,
                    self.exit,
                    self.rho_min,
                    self.variance_min,
                    self.fees,
                    self.slippage,
                ]
            ).all()
            or self.keep < 1
            or not 0 < self.ranking_fraction <= 1
            or self.window < 2
            or self.entry <= self.exit
        ):
            raise ValueError("Invalid prespecified study configuration")
        if self.mode not in ("exact", "legacy") or self.normalization not in (
            "formation",
            "rolling",
        ):
            raise ValueError("Unknown model or normalization mode")


def formation_study(
    training: Any, testing: Any, membership: Any, *, config: StudyConfig | None = None
) -> dict[str, Any]:
    """Fit/rank on formation data, then evaluate a locked selection.

    Ranking fraction is a cross-sectional screen, not a calibrated p-value.
    Duplicate-security exposures are prevented by deterministic disjoint pairing.
    This implementation is a corrected study variant, not certified paper parity.
    """
    c = config or StudyConfig()
    tr = validate_prices(training)
    te = validate_prices(testing)
    if tr.empty or te.empty or tr.index[-1] >= te.index[0]:
        raise ValueError("Formation must precede a nonempty evaluation period")
    names = members_at(membership, tr.index[-1], column="security_id")
    names = sorted(set(names) & set(tr.columns))
    names = [s for s in names if tr[s].notna().all()]
    if not set(names) <= set(te.columns):
        raise ValueError("Evaluation schema omits formation securities")
    active = membership.loc[
        (pd.to_datetime(membership["from"]) <= tr.index[-1])
        & (
            membership["thru"].isna()
            | (pd.to_datetime(membership["thru"]) >= tr.index[-1])
        )
    ]
    if (
        "sector" in active
        and active.groupby("security_id")["sector"].nunique().gt(1).any()
    ):
        raise ValueError("Conflicting overlapping point-in-time sectors")
    sectors = (
        dict(zip(active["security_id"].astype(str), active["sector"]))
        if "sector" in active
        else {}
    )
    if c.same_sector and any(s not in sectors or pd.isna(sectors[s]) for s in names):
        raise ValueError("Same-sector screen requires point-in-time sector data")
    candidates: list[Candidate] = []
    failures: list[dict[str, str]] = []
    for i, x in enumerate(names):
        for y in names[i + 1 :]:
            if c.same_sector and sectors[x] != sectors[y]:
                continue
            key = x + "/" + y
            try:
                fitted = fit_pair(
                    tr[x], tr[y], mode=c.mode, options=FitOptions(seed=c.seed)
                )
                p = fitted.parameters
                candidates.append(
                    {
                        "pair": key,
                        "x": x,
                        "y": y,
                        "fit": fitted,
                        "ratio": fitted.log_ratio,
                        "beta": p.beta,
                        "rho": p.rho,
                        "variance_fraction": p.variance_fraction,
                        "training_rank": -np.inf,
                    }
                )
            except (FitError, ValueError) as exc:
                failures.append({"pair": key, "error": str(exc)})
    ranked = sorted(candidates, key=lambda d: (d["ratio"], d["pair"]))
    cutoff = (
        float(np.quantile([d["ratio"] for d in ranked], c.ranking_fraction))
        if ranked
        else -np.inf
    )
    eligible = [
        d
        for d in ranked
        if d["ratio"] <= cutoff
        and d["beta"] > 0
        and d["rho"] > c.rho_min
        and d["variance_fraction"] > c.variance_min
    ]
    # In-sample ranking is explicitly exploratory/training-only. It must not be
    # presented as an independent test of historical performance.
    for item in eligible:
        p = item["fit"].parameters
        x = item["x"]
        y = item["y"]
        w = tr[y] - p.beta * tr[x]
        from .filtering import filter_residuals

        m = pd.Series(filter_residuals(w.to_numpy(), p, mode=c.mode).m, index=tr.index)
        if c.normalization == "formation":
            score = m / m.std(ddof=1)
        else:
            roll = m.rolling(c.window)
            score = (m - roll.mean()) / roll.std(ddof=1).replace(0, np.nan)
        opens, closes = decisions(score.rename(item["pair"]), c.entry, c.exit)
        training_ledger = backtest_pairs(
            tr[[x, y]],
            {item["pair"]: (x, y, p.beta)},
            opens,
            closes,
            config=ExecutionConfig(fees=c.fees, slippage=c.slippage),
        )
        returns = training_ledger.equity.pct_change().dropna()
        sd = float(returns.std(ddof=1))
        item["training_rank"] = float(returns.mean() / sd) if sd > 0 else -np.inf
    used: set[str] = set()
    selected: list[Candidate] = []
    for item in sorted(eligible, key=lambda d: (-d["training_rank"], d["pair"])):
        if not np.isfinite(item["training_rank"]):
            continue
        if item["x"] in used or item["y"] in used:
            continue
        selected.append(item)
        used.update([item["x"], item["y"]])
        if len(selected) >= c.keep:
            break
    entries = pd.DataFrame(index=te.index)
    exits = entries.copy()
    pairs: dict[str, tuple[str, str, float]] = {}
    for item in selected:
        p = item["fit"].parameters
        x, y = item["x"], item["y"]
        name = item["pair"]
        if c.mode == "legacy":
            from .filtering import filter_residuals

            wtr = tr[y] - p.beta * tr[x]
            wte = te[y] - p.beta * te[x]
            a = filter_residuals(wtr.to_numpy(), p, mode="legacy", timestamps=tr.index)
            b = filter_residuals(
                wte.to_numpy(),
                p,
                mode="legacy",
                state=a.final_state,
                timestamps=te.index,
            )
            combined = pd.Series(np.r_[a.m, b.m], index=tr.index.append(te.index))
            if c.normalization == "formation":
                score = combined.iloc[len(tr) :] / np.std(a.m, ddof=1)
            else:
                rolling = combined.rolling(c.window)
                score = ((combined - rolling.mean()) / rolling.std(ddof=1)).iloc[
                    len(tr) :
                ]
        else:
            components = score_components(
                tr[y] - p.beta * tr[x],
                te[y] - p.beta * te[x],
                p,
                variant=c.normalization,
                window=c.window,
            )
            score = components["testing_score"]
        op, cl = decisions(score.rename(name), c.entry, c.exit)
        entries[name] = op[name]
        exits[name] = cl[name]
        pairs[name] = (x, y, p.beta)
    ledger = (
        None
        if not pairs
        else backtest_pairs(
            te[sorted(used)],
            pairs,
            entries,
            exits,
            config=ExecutionConfig(fees=c.fees, slippage=c.slippage),
        )
    )
    # Prespecified conventional comparator: OLS level-spread, same selected
    # securities, formation scale, entry=2/exit=0; not a cointegration test.
    baseline_pairs: dict[str, tuple[str, str, float]] = {}
    baseline_entries = pd.DataFrame(index=te.index)
    baseline_exits = pd.DataFrame(index=te.index)
    for item in selected:
        x, y, name = item["x"], item["y"], item["pair"]
        intercept, beta = np.linalg.lstsq(
            np.column_stack([np.ones(len(tr)), tr[x].to_numpy()]),
            tr[y].to_numpy(),
            rcond=None,
        )[0]
        spread = tr[y] - intercept - beta * tr[x]
        scale = float(spread.std(ddof=1))
        if beta <= 0 or scale <= 0 or not np.isfinite([beta, scale]).all():
            continue
        z = ((te[y] - intercept - beta * te[x]) / scale).rename(name)
        op, cl = decisions(z, 2.0, 0.0)
        baseline_entries[name] = op[name]
        baseline_exits[name] = cl[name]
        baseline_pairs[name] = (x, y, float(beta))
    baseline_ledger = (
        None
        if not baseline_pairs
        else backtest_pairs(
            te[sorted(used)],
            baseline_pairs,
            baseline_entries,
            baseline_exits,
            config=ExecutionConfig(fees=c.fees, slippage=c.slippage),
        )
    )
    selected_info = [{k: v for k, v in item.items() if k != "fit"} for item in selected]
    return {
        "selected": selected_info,
        "failures": failures,
        "candidate_count": len(candidates),
        "ledger": ledger,
        "ranking_cutoff": cutoff,
        "no_trade_equity": pd.Series(1.0, index=te.index),
        "ols_spread_baseline": baseline_ledger,
        "certified_paper_replication": False,
    }
