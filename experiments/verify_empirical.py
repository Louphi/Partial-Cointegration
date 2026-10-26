"""Exercise configured cohort and overlap commands with explicitly synthetic files."""

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from partial_cointegration.study import StudyConfig

root = Path(__file__).resolve().parents[1]
folder = root / ".test-deps/empirical-command-check"
folder.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20261025)
n = 1600
dates = pd.bdate_range("2018-01-01", periods=n)
x = 100 + np.cumsum(rng.normal(0, 0.2, n))
m = np.zeros(n)
for i in range(1, n):
    m[i] = 0.8 * m[i - 1] + rng.normal(0, 0.5)
permanent = np.cumsum(rng.normal(0, 0.1, n))
pd.DataFrame({"A": x, "B": x + m + permanent}, index=dates).to_csv(
    folder / "prices.csv", index_label="date"
)
pd.DataFrame(
    {
        "security_id": ["A", "B"],
        "from": ["2018-01-01"] * 2,
        "thru": [None, None],
        "sector": ["synthetic"] * 2,
    }
).to_csv(folder / "members.csv", index=False)
manifest = {
    "source": "seeded synthetic random walk plus AR/RW spread; not market data",
    "adjustment_basis": "synthetic positive levels",
    "security_id_mapping": "synthetic A/B labels",
    "corporate_action_policy": "none in synthetic DGP",
    "delisting_policy": "none in synthetic DGP",
    "prices_file": "prices.csv",
    "membership_file": "members.csv",
    "data_kind": "synthetic",
}
(folder / "manifest.json").write_text(json.dumps(manifest))
(folder / "config.json").write_text(json.dumps(asdict(StudyConfig(ranking_fraction=1))))
outputs = []
for i, date in enumerate(["2023-01-01", "2023-02-01"]):
    dest = folder / f"cohort-{i}"
    subprocess.run(
        [
            sys.executable,
            str(root / "experiments/empirical.py"),
            "--manifest",
            str(folder / "manifest.json"),
            "--config",
            str(folder / "config.json"),
            "--trading-start",
            date,
            "--output",
            str(dest),
        ],
        check=True,
        cwd=root,
    )
    outputs.append(str(dest))
(folder / "cohorts.json").write_text(json.dumps(outputs))
empty_config = asdict(StudyConfig(ranking_fraction=1, rho_min=1))
(folder / "empty-config.json").write_text(json.dumps(empty_config))
for config_name in ["empty-config.json", "config.json"]:
    subprocess.run(
        [
            sys.executable,
            str(root / "experiments/empirical.py"),
            "--manifest",
            str(folder / "manifest.json"),
            "--config",
            str(folder / config_name),
            "--trading-start",
            "2023-01-01",
            "--output",
            outputs[0],
        ],
        check=True,
        cwd=root,
    )
    if config_name == "empty-config.json":
        assert pd.read_csv(Path(outputs[0]) / "orders.csv").empty
        assert pd.read_csv(Path(outputs[0]) / "ols-spread-equity.csv").empty
        assert (pd.read_csv(Path(outputs[0]) / "equity.csv").iloc[:, 1] == 1).all()
subprocess.run(
    [
        sys.executable,
        str(root / "experiments/aggregate.py"),
        "--cohorts",
        str(folder / "cohorts.json"),
        "--output",
        str(folder / "aggregate"),
    ],
    check=True,
    cwd=root,
)
portfolio = pd.read_csv(folder / "aggregate/portfolio.csv")
assert np.isfinite(portfolio.equity).all()
assert (portfolio.cash >= -1e-12).all()
aggregate_report = json.loads((folder / "aggregate/aggregate.json").read_text())
assert "unavailable" in aggregate_report["hac_monthly_mean"]
report = {
    "status": "passed",
    "synthetic_only": True,
    "seed": 20261025,
    "cohorts": 2,
    "overlapping_cash_pool": True,
    "empty_selection_clears_prior_output": True,
    "portfolio_observations": len(portfolio),
    "historical_performance_verified": False,
    "hac_unavailable_reason": "Two six-month cohorts provide too few monthly returns",
    "fixture_directory": str(folder.relative_to(root)),
}
(root / "docs/results/empirical-workflow-verification.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report, indent=2))
