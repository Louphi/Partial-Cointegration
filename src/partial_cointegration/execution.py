"""Independent shared-capital ledger with causal next-quote execution."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from .data import validate_prices


@dataclass(frozen=True)
class ExecutionConfig:
    """Costs are per fill; borrow accrues per row at a declared frequency.

    The default is daily (252 rows/year). A minute adapter must explicitly
    supply its frequency and handle session boundaries/overnight exposure.
    """

    capital: float = 1.0
    fees: float = 0.001
    slippage: float = 0.001
    annual_borrow: float = 0.0
    gross_leverage: float = 1.0
    liquidate: bool = True
    borrow_observations_per_year: float = 252.0

    def __post_init__(self) -> None:
        if (
            not np.isfinite(
                [
                    self.capital,
                    self.fees,
                    self.slippage,
                    self.annual_borrow,
                    self.gross_leverage,
                    self.borrow_observations_per_year,
                ]
            ).all()
            or self.capital <= 0
            or self.gross_leverage <= 0
            or self.borrow_observations_per_year <= 0
            or min(self.fees, self.slippage, self.annual_borrow) < 0
            or self.slippage >= 1
        ):
            raise ValueError("Invalid execution configuration")


@dataclass(frozen=True)
class LedgerResult:
    """One portfolio equity/cash ledger and explicitly rejected/unfilled orders."""

    equity: Any
    cash: Any
    holdings: Any
    orders: Any
    rejections: Any
    stale_valuation: Any
    config: ExecutionConfig
    cash_distributions: Any = None

    def value(self) -> Any:
        """Portfolio value on one committed-capital denominator."""
        return self.equity

    def stats(self) -> Any:
        """Descriptive returns; no annualized significance or profitability claims."""
        return pd.Series(
            {
                "Initial capital": self.config.capital,
                "Final value": self.equity.iloc[-1],
                "Total return": self.equity.iloc[-1] / self.config.capital - 1,
                "Fees paid": self.orders["fees"].sum() if len(self.orders) else 0.0,
                "Rejected/unfilled requests": len(self.rejections),
            }
        )


def backtest_pairs(
    prices: Any,
    pairs: dict[str, tuple[str, str, float]],
    entries: Any,
    exits: Any,
    *,
    config: ExecutionConfig | None = None,
    weights: dict[str, float] | None = None,
    settlements: dict[str, float] | None = None,
    cash_distributions: Any = None,
) -> LedgerResult:
    """Fill decisions at a strictly later observed quote, using decision-time units.

    Prices at the next quote determine fills, not decisions or initial units.
    Portfolio gross margin is checked before entry; short sale proceeds are not
    treated as additional capacity. Missing marks are valuation-only and flagged.
    Terminal settlement values are explicit cash payments, never inferred prices.
    Optional per-share cash distributions accrue before executions to positions
    carried from the preceding row: longs receive, shorts owe. Ex-date inputs
    imply ex-date cash accrual, not a modeled payment-date receivable schedule.
    """
    c = config or ExecutionConfig()
    frame = validate_prices(prices)
    if (
        not pairs
        or len(frame) < 2
        or not frame.index.equals(entries.index)
        or not frame.index.equals(exits.index)
    ):
        raise ValueError("Need pairs, >=2 price rows, and matching decision dates")
    if set(entries.columns) != set(pairs) or set(exits.columns) != set(pairs):
        raise ValueError("Decision columns must equal pair IDs")
    for name, (x, y, beta) in pairs.items():
        if (
            x == y
            or x not in frame
            or y not in frame
            or not np.isfinite(beta)
            or beta <= 0
        ):
            raise ValueError(
                f"{name}: only distinct securities with positive beta are tradable"
            )
    allocation = weights or {k: 1 / len(pairs) for k in pairs}
    if (
        set(allocation) != set(pairs)
        or any(not np.isfinite(w) or w <= 0 for w in allocation.values())
        or sum(allocation.values()) > 1 + 1e-12
    ):
        raise ValueError("Positive pair weights must sum to at most one")
    settlement = settlements or {}
    distributions = pd.DataFrame(0.0, index=frame.index, columns=frame.columns)
    if cash_distributions is not None:
        if (
            not isinstance(cash_distributions, pd.DataFrame)
            or not cash_distributions.index.equals(frame.index)
            or not cash_distributions.columns.is_unique
            or not set(cash_distributions.columns) <= set(frame.columns)
            or not np.isfinite(cash_distributions.to_numpy(dtype=float)).all()
            or (cash_distributions.to_numpy(dtype=float) < 0).any()
        ):
            raise ValueError(
                "Cash distributions need aligned dates, known unique columns and finite nonnegative per-share amounts"
            )
        distributions.loc[:, cash_distributions.columns] = cash_distributions
    if any(
        s not in frame or not np.isfinite(v) or v < 0 for s, v in settlement.items()
    ):
        raise ValueError("Settlement values must be known securities and nonnegative")
    positions = {k: np.zeros(2) for k in pairs}
    pending: dict[str, tuple[int, str, np.ndarray[Any, Any]]] = {}
    cash = float(c.capital)
    last = np.full(len(frame.columns), np.nan)
    equities = []
    cash_values = []
    holdings = []
    stales = []
    orders = []
    rejects = []
    distribution_flows = []

    def total_holdings() -> Any:
        total = pd.Series(0.0, index=frame.columns)
        for key, (x, y, _) in pairs.items():
            total[x] += positions[key][0]
            total[y] += positions[key][1]
        return total

    for i, (date, row) in enumerate(frame.iterrows()):
        observed = row.to_numpy(dtype=float)
        mask = np.isfinite(observed)
        last[mask] = observed[mask]
        marks = pd.Series(last, index=frame.columns)
        before = total_holdings()
        distribution = float((before * distributions.loc[date]).sum())
        cash += distribution
        distribution_flows.append(distribution)
        borrow = (
            c.annual_borrow
            / c.borrow_observations_per_year
            * sum(max(-before[s], 0) * marks[s] for s in frame if before[s] != 0)
        )
        cash -= borrow
        for key in sorted(list(pending), key=lambda k: pending[k][1] != "exit"):
            decision, kind, units = pending[key]
            x, y, _ = pairs[key]
            if not np.isfinite([row[x], row[y]]).all():
                continue
            if decision >= i:
                raise RuntimeError("Decision cannot execute on its own row")
            if kind == "exit":
                units = -positions[key]
            if not np.any(units):
                del pending[key]
                continue
            raw = np.array([row[x], row[y]], dtype=float)
            filled = raw * (1 + c.slippage * np.sign(units))
            fee = float(np.sum(np.abs(units * filled)) * c.fees)
            if kind == "entry":
                gross = sum(
                    float(
                        np.sum(
                            np.abs(
                                positions[k]
                                * np.array([marks[pairs[k][0]], marks[pairs[k][1]]])
                            )
                        )
                    )
                    for k in pairs
                    if np.any(positions[k])
                )
                current_equity = cash + float(
                    (before[before != 0] * marks[before != 0]).sum()
                )
                proposed = float(np.sum(np.abs(units * raw)))
                if (
                    gross + proposed + fee
                    > max(current_equity, 0) * c.gross_leverage + 1e-12
                ):
                    rejects.append(
                        {
                            "pair": key,
                            "decision_time": frame.index[decision],
                            "time": date,
                            "reason": "insufficient gross margin",
                        }
                    )
                    del pending[key]
                    continue
            cash -= float(units @ filled) + fee
            for j, security in enumerate([x, y]):
                orders.append(
                    {
                        "pair": key,
                        "decision_time": frame.index[decision],
                        "fill_time": date,
                        "security": security,
                        "units": float(units[j]),
                        "price": float(filled[j]),
                        "fees": float(abs(units[j] * filled[j]) * c.fees),
                        "kind": kind,
                    }
                )
            positions[key] += units
            before = total_holdings()
            del pending[key]
        # Administrative end-of-period liquidation is scheduled by the study,
        # independent of a final-row signal. Unquoted holdings require settlement.
        if i == len(frame) - 1 and c.liquidate:
            for key, (x, y, _) in pairs.items():
                if not np.any(positions[key]):
                    continue
                for j, security in enumerate([x, y]):
                    closing_units = float(-positions[key][j])
                    if closing_units == 0:
                        continue
                    is_settlement = not np.isfinite(row[security])
                    if is_settlement and security not in settlement:
                        raise ValueError(
                            f"Terminal quote unavailable for held {security}; supply official settlement"
                        )
                    price = (
                        float(settlement[security])
                        if is_settlement
                        else float(
                            row[security] * (1 + c.slippage * np.sign(closing_units))
                        )
                    )
                    fee = 0.0 if is_settlement else abs(closing_units * price) * c.fees
                    cash -= closing_units * price + fee
                    orders.append(
                        {
                            "pair": key,
                            "decision_time": None,
                            "fill_time": date,
                            "security": security,
                            "units": closing_units,
                            "price": price,
                            "fees": fee,
                            "kind": (
                                "settlement"
                                if is_settlement
                                else "scheduled_liquidation"
                            ),
                        }
                    )
                positions[key] = np.zeros(2)
        for key, (x, y, beta) in pairs.items():
            active = bool(np.any(positions[key]))
            opening = bool(entries.loc[date, key])
            closing = bool(exits.loc[date, key])
            if active and closing:
                pending[key] = (i, "exit", -positions[key].copy())
            elif not active and opening and not closing and key not in pending:
                if not np.isfinite([row[x], row[y]]).all():
                    rejects.append(
                        {
                            "pair": key,
                            "decision_time": date,
                            "time": date,
                            "reason": "missing decision quote",
                        }
                    )
                    continue
                budget = (
                    c.capital
                    * allocation[key]
                    * c.gross_leverage
                    / (1 + c.fees + c.slippage)
                )
                unit = budget / (beta * row[x] + row[y])
                pending[key] = (i, "entry", np.array([beta * unit, -unit]))
        total = total_holdings()
        held = total != 0
        value = cash + float((total[held] * marks[held]).sum())
        stale = bool((held.to_numpy() & ~mask).any())
        equities.append(value)
        cash_values.append(cash)
        holdings.append(total.to_numpy())
        stales.append(stale)
    for key, (decision, _, _) in pending.items():
        rejects.append(
            {
                "pair": key,
                "decision_time": frame.index[decision],
                "time": frame.index[-1],
                "reason": "no later executable quote",
            }
        )
    columns = [
        "pair",
        "decision_time",
        "fill_time",
        "security",
        "units",
        "price",
        "fees",
        "kind",
    ]
    return LedgerResult(
        pd.Series(equities, index=frame.index, name="equity"),
        pd.Series(cash_values, index=frame.index, name="cash"),
        pd.DataFrame(holdings, index=frame.index, columns=frame.columns),
        pd.DataFrame(orders, columns=columns),
        pd.DataFrame(rejects, columns=["pair", "decision_time", "time", "reason"]),
        pd.Series(stales, index=frame.index),
        c,
        pd.Series(distribution_flows, index=frame.index, name="cash_distributions"),
    )


def replay_vectorbt(result: LedgerResult, prices: Any) -> Any:
    """Compare real engine accounting for the independent ledger's executed units.

    This adapter does not derive the oracle's cash/P&L from engine output.
    It cannot replay official non-price settlement or borrow charges.
    """
    import vectorbt as vbt

    if (
        result.config.annual_borrow
        or (result.orders["kind"] == "settlement").any()
        or (
            result.cash_distributions is not None
            and result.cash_distributions.ne(0).any()
        )
    ):
        raise ValueError(
            "Engine replay cannot represent borrow, cash distributions or cash settlement"
        )
    if result.orders.duplicated(["fill_time", "security"]).any():
        raise ValueError(
            "Engine replay cannot net multiple same-bar fills without changing costs"
        )
    securities = list(result.orders["security"].unique())
    if not securities:
        securities = list(prices.columns[:2])
    frame = prices[securities]
    if frame.isna().any().any():
        raise ValueError("Use the authoritative ledger for missing-price execution")
    amounts = pd.DataFrame(np.nan, index=frame.index, columns=frame.columns)
    for _, order in result.orders.iterrows():
        old = amounts.loc[order["fill_time"], order["security"]]
        amounts.loc[order["fill_time"], order["security"]] = (
            0 if pd.isna(old) else old
        ) + order["units"]
    return vbt.Portfolio.from_orders(
        frame,
        size=amounts,
        size_type="amount",
        init_cash=result.config.capital,
        cash_sharing=True,
        group_by=True,
        call_seq="auto",
        fees=result.config.fees,
        slippage=result.config.slippage,
    )
