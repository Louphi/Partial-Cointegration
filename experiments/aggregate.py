"""Aggregate configured cohort outputs under one funding pool; no implicit leverage."""

import argparse
import json
import hashlib
from pathlib import Path
import pandas as pd
from partial_cointegration.portfolio import aggregate_cohorts, hac_mean_interval
from partial_cointegration.io import package_fingerprint


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cohorts", required=True, help="JSON list of empirical output directories"
    )
    p.add_argument("--output", required=True)
    p.add_argument("--max-active", type=int, default=6)
    args = p.parse_args()
    folders = [Path(item) for item in json.loads(Path(args.cohorts).read_text())]
    if len({str(folder.resolve()) for folder in folders}) != len(folders):
        raise ValueError(
            "Duplicate cohort directories would double-count an allocation"
        )
    records = [json.loads((folder / "run.json").read_text()) for folder in folders]
    kinds = {record["manifest"]["data_kind"] for record in records}
    names = {record["config"]["name"] for record in records}
    configurations = {
        json.dumps(record["config"], sort_keys=True) for record in records
    }
    versions = {record["package_fingerprint"] for record in records}
    if (
        len(kinds) != 1
        or len(names) != 1
        or len(configurations) != 1
        or len(versions) != 1
    ):
        raise ValueError(
            "Do not silently mix synthetic/market data or different strategy variants"
        )
    curves = {
        str(i): pd.read_csv(folder / "equity.csv", index_col=0, parse_dates=True).iloc[
            :, 0
        ]
        for i, folder in enumerate(folders)
    }
    result = aggregate_cohorts(curves, max_active=args.max_active)
    monthly = (
        result["equity"].resample("ME").last().pct_change(fill_method=None).dropna()
    )
    try:
        interval = hac_mean_interval(monthly, lag=6)
    except ValueError as exc:
        interval = {"unavailable": str(exc)}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    result["ledger"].to_csv(out / "portfolio.csv")
    result["allocations"].to_csv(out / "allocations.csv", index=False)
    (out / "aggregate.json").write_text(
        json.dumps(
            {
                "data_kind": next(iter(kinds)),
                "strategy": next(iter(names)),
                "cohorts": [str(folder) for folder in folders],
                "max_active": args.max_active,
                "package_fingerprint": package_fingerprint(),
                "cohort_package_fingerprint": next(iter(versions)),
                "config": records[0]["config"],
                "input_sha256": {
                    str(folder / file): hashlib.sha256(
                        (folder / file).read_bytes()
                    ).hexdigest()
                    for folder in folders
                    for file in ("run.json", "equity.csv")
                },
                "hac_monthly_mean": interval,
                "historical_performance_verified": False,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
