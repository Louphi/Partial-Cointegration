"""Configured one-cohort empirical entry point; requires provenance-qualified files."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pandas as pd
import numpy as np
from partial_cointegration.study import StudyConfig, formation_study
from partial_cointegration.data import validate_prices
from partial_cointegration.io import estimator_fingerprint, package_fingerprint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--config", default="experiments/configs/nasdaq-extension.json")
    p.add_argument("--trading-start", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    required = [
        "source",
        "adjustment_basis",
        "security_id_mapping",
        "corporate_action_policy",
        "delisting_policy",
        "prices_file",
        "membership_file",
        "data_kind",
    ]
    if any(not manifest.get(k) for k in required):
        raise ValueError(
            "Manifest must document source, adjustments, mapping, corporate actions, delistings and files"
        )
    if manifest["data_kind"] not in ("market", "synthetic"):
        raise ValueError("data_kind must be explicit market/synthetic")
    pricepath = manifest_path.parent / manifest["prices_file"]
    memberpath = manifest_path.parent / manifest["membership_file"]
    prices = validate_prices(pd.read_csv(pricepath, index_col="date"))
    membership = pd.read_csv(memberpath)
    start = pd.Timestamp(args.trading_start)
    tr = prices.loc[
        (prices.index >= start - pd.DateOffset(months=48)) & (prices.index < start)
    ]
    te = prices.loc[
        (prices.index >= start) & (prices.index < start + pd.DateOffset(months=6))
    ]
    if prices.index.min() > start - pd.DateOffset(months=48) + pd.Timedelta(days=7):
        raise ValueError(
            "Price feed does not cover the required 48-month formation history"
        )
    config = StudyConfig(**json.loads(Path(args.config).read_text()))
    result = formation_study(tr, te, membership, config=config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    ledger = result.pop("ledger")
    baseline_ledger = result.pop("ols_spread_baseline")
    if baseline_ledger is not None:
        baseline_ledger.equity.to_csv(out / "ols-spread-equity.csv")
    else:
        pd.Series(dtype=float, name="equity").to_csv(out / "ols-spread-equity.csv")
    baseline = result.pop("no_trade_equity")
    baseline.to_csv(out / "no-trade-equity.csv")
    if ledger is not None:
        ledger.equity.to_csv(out / "equity.csv")
        ledger.orders.to_csv(out / "orders.csv", index=False)
        ledger.rejections.to_csv(out / "rejections.csv", index=False)
    else:
        baseline.to_csv(out / "equity.csv")
        pd.DataFrame(
            columns=[
                "pair",
                "decision_time",
                "fill_time",
                "security",
                "units",
                "price",
                "fees",
                "kind",
            ]
        ).to_csv(out / "orders.csv", index=False)
        pd.DataFrame(columns=["pair", "decision_time", "reason"]).to_csv(
            out / "rejections.csv", index=False
        )
    result.update(
        ols_baseline_available=baseline_ledger is not None,
        entrypoint_sha256=hashlib.sha256(
            Path(__file__).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest(),
        requested_trading_start=str(start),
        formation_observations=len(tr),
        formation_start=str(tr.index.min()),
        formation_end=str(tr.index.max()),
        evaluation_observations=len(te),
        evaluation_start=str(te.index.min()),
        evaluation_end=str(te.index.max()),
        exchange_calendar_coverage_verified=False,
        config=asdict(config),
        estimator_fingerprint=estimator_fingerprint(),
        package_fingerprint=package_fingerprint(),
        manifest=manifest,
        input_sha256={
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [pricepath, memberpath]
        },
        historical_performance_verified=False,
    )

    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        return value

    (out / "run.json").write_text(
        json.dumps(clean(result), indent=2, default=str, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
