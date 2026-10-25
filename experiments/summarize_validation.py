"""Render executed scientific evidence; never promote missing evidence to a pass."""

import json
from pathlib import Path
import re
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from partial_cointegration.io import load_current_report

root = Path(__file__).resolve().parents[1]
results = root / "docs/results"
r = load_current_report(results / "scientific-validation.json")
legacy_path = results / "legacy-score-tables.json"
legacy = json.loads(legacy_path.read_text()) if legacy_path.exists() else None
raw_log = (results / "software-tests.txt").read_bytes()
testlog = raw_log.decode("utf-16" if raw_log.startswith(b"\xff\xfe") else "utf-8")
passed = re.search(r"(\d+) passed", testlog).group(1)
coverage = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+%)", testlog).group(1)
lines = [
    "# Executed correction and validation report",
    "",
    "The corrected implementation passes the executed software and synthetic size gates. "
    "This is a research implementation with bounded evidence; original-author algorithm parity "
    "and historical trading performance remain unverified.",
    "",
    "## Implemented method",
    "",
    "The default model is conditional Gaussian partial cointegration: alpha=0 is an explicit "
    "location normalization, M0=0, R0=w0 and P0=0. A covariance-propagating filter and matching "
    "innovation density replace the inherited fixed-gain/q+r score. Beta and overall variance "
    "are analytically profiled; nonlinear candidates include mixed, pure-AR and pure-RW models. "
    "Only accepted finite optimizations enter selection. Multistart searches do not prove a global maximum.",
    "",
    "Legacy scoring is separately named and calibrated. Its inherited pure-AR initialization switch, "
    "profiled optimization and normalized intercept prevent a certified reproduction claim. "
    "The mathematical specification, mode differences and migration requirements are in "
    "[model-contract.md](model-contract.md) and [migration-guide.md](migration-guide.md).",
    "",
    "## Software evidence",
    "",
    f"The full suite passed **{passed} tests with {coverage} package line coverage**. "
    "Strict mypy passed for all 13 package modules; Black and the undefined/unused-local "
    "Ruff checks passed. The refreshed editable installation passed pip's dependency check. "
    "See [software-tests.txt](results/software-tests.txt). Third-party plotting deprecation "
    "warnings remain and do not indicate failed numerical checks.",
    "",
    "Independent dense Gaussian covariance densities cover 60 fixtures, including negative/near-root "
    "rho and AR/RW boundaries. Other checks cover 100 nested-likelihood fits, forced and rare optimizer "
    "failures, compiled versus Python kernels, PSD/invalid states, missing-observation continuation, "
    "future invariance, alignment, seed reproducibility and immutable summaries. "
    "Hand-calculated fees/P&L and actual vectorbt replay check the shared-capital ledger. "
    "Cross-cohort funding has a separate hand-calculated cash oracle.",
    "",
    "Both active notebooks execute in fresh kernels and the Streamlit synthetic button flow executes "
    "without an application error. A separate synthetic command check runs two overlapping cohorts "
    "and the common-cash aggregator. See [notebook/app verification](results/notebook-app-verification.json) "
    "and [cohort verification](results/empirical-workflow-verification.json). These checks do not "
    "use genuine market data.",
    "",
    "## Corrected-model Monte Carlo evidence",
    "",
    "Seed: 20261025. Sample sizes: 100, 1,000 and 10,000. The recovery grid has 75 cells "
    "(five rho values including negative and near-root cases, five transient/permanent noise ratios, "
    "three sample sizes), 1,000 independent replications each, plus six boundary cells with 1,000 "
    "replications each. All 81,000 recovery/boundary fits returned accepted results. Accepted results "
    "may retain diagnostics from rejected internal search attempts. Weak signal and boundary estimates "
    "can still be inaccurate despite successful numerical convergence.",
    "",
    "An earlier full-study attempt exposed two failed mixed searches at n=100, rho=-0.8 and "
    "sigma_M=10. Scalar pure-AR optimization and a Powell fallback corrected that failure. "
    "Both seed cases are regression tests. The final study was restarted under the corrected numerical "
    "source fingerprint; earlier checkpoints were not pooled with it.",
    "",
    "Independent 10,000-draw calibration and 10,000-draw evaluation banks were run at each sample "
    "size: 60,000 null fits. The predeclared gate uses simultaneous 99% Bonferroni Wilson intervals "
    "over nine evaluation cells, augmented by 200 calibration resamples. Every size gate passed; "
    "no failed null fits were dropped. Rates are conditional on this geometric-x, known-initial-state "
    "DGP and estimator configuration, not universal critical values.",
    "",
    "| n | Nominal rate | Evaluation rate | Simultaneous Wilson interval | Gate |",
    "|---:|---:|---:|---|---|",
]
for n, entries in r["size"].items():
    for alpha, cell in entries.items():
        lo, hi = cell["wilson_99"]
        lines.append(
            f"| {n} | {100*float(alpha):.0f}% | "
            f"{100*cell['rejections']['conditional_rate']:.2f}% | "
            f"{100*lo:.2f}%–{100*hi:.2f}% | {cell['size_gate_passed']} |"
        )
lines += [
    "",
    "Calibration uncertainty is recorded separately in the machine-readable report and "
    "is included in the gate, not in the Wilson column above.",
    "",
    "The prespecified coverage subset uses 100 independent n=1,000 datasets with "
    "beta=1, rho=0.6 and sigma_M=sigma_R=1. Each fitted dataset has 500 independent parametric "
    "bootstrap fits (50,000 inner fits plus 100 outer fits). All intervals were available. "
    "These are percentile 95% intervals; the observed coverage and unadjusted 99% binomial "
    "intervals are below. All contain 95%, but 100 outer replications give broad uncertainty "
    "and cannot establish boundary or weak-identification coverage.",
    "",
    "| Parameter | Observed coverage | 99% Wilson interval | Contains 95% |",
    "|---|---:|---|---|",
]
for cell in r["coverage"]:
    lo, hi = cell["wilson_99"]
    lines.append(
        f"| {cell['parameter']} | {100*cell['coverage']:.0f}% | "
        f"{100*lo:.1f}%–{100*hi:.1f}% | {cell['nominal_in_interval']} |"
    )
lines += [
    "",
    "The exact study therefore contains **191,100 fits** in total. "
    "[Full results](results/scientific-validation.json) include bias, Monte Carlo standard "
    "errors, RMSE with 200-resample 99% bootstrap bands, quantiles, boundary mass, power "
    "with failure bounds, versions, configuration and checkpoint digests. Numerical recovery "
    "in a finite grid does not prove consistency or efficiency. Rho is not scored as identified "
    "when sigma_M=0.",
    "",
    "![Selected recovery curves](results/recovery-summary.png)",
    "",
    "The plot shows a selected equal-noise slice, not the entire grid. "
    "The JSON includes weak-signal cases with substantial rho error and frequent boundary estimates.",
    "",
    "## Legacy comparison and reproduction",
    "",
]
if legacy:
    failures = sum(c["failed"] for c in legacy["cells"])
    gates = all(c["passed"] for c in legacy["null_size"].values())
    lines += [
        f"Executed 51 legacy-score settings × {legacy['replications_per_cell']:,} replications "
        "plus three independent 10,000-draw null-calibration banks: **540,000 fits**. "
        f"There were {failures} failed setting fits. All independent 5% size gates passed: {gates}. "
        "See [legacy-score-tables.json](results/legacy-score-tables.json).",
        "",
        "These tables compare the inherited scorer under its stated DGP with the revised optimizer. "
        "Author-algorithm parity is explicitly false. Rho=1 is generated as its observationally "
        "equivalent combined-variance random walk; its AR/RW decomposition errors are undefined. "
        "Published point estimates without Monte Carlo uncertainty do not justify an equality gate. "
        "The full original R optimizer was not executed as an independent reference.",
        "",
    ]
else:
    lines += [
        "Legacy-score comparison is still running; its evidence is not marked complete.",
        "",
    ]
reference = json.loads((results / "r-reference-verification.json").read_text())
lines += [
    f"A later environment search found {reference['runtime']} outside PATH. "
    f"The unmodified pinned author-written R gain and steady-state scoring functions "
    f"were executed on {reference['fixtures']} fixtures. After explicitly matching initial "
    f"levels and accounting for the R first-observation density, maximum score disagreement "
    f"was {reference['max_absolute_score_error']:.2g} and gain disagreement was "
    f"{reference['max_absolute_gain_error']:.2g}. "
    "See [R reference verification](results/r-reference-verification.json) for source URLs, "
    "digests, runtime and conventions. This verifies the legacy scorer under those conventions; "
    "it does not establish original optimizer parity or validate q+r as an exact latent-state likelihood.",
    "",
    "## Data, execution and remaining gates",
    "",
    "The point-in-time selector uses formation information only, stable identifiers and immutable "
    "membership intervals. Evaluation filtering continues the saved posterior. Normalization is "
    "formation-fixed or causal rolling. Orders execute at a later observed quote; both legs share "
    "cash and capacity, costs are explicit, and missing terminal quotes require official settlement. "
    "Negative-beta trading is rejected. Gross capacity is checked on entry; intraday margin calls "
    "and financing are outside this daily-close ledger.",
    "",
    "The paper-inspired reference and Nasdaq extension are distinct frozen configurations. "
    "Cross-sectional ranking is not a calibrated individual test. The same-selection OLS level-spread "
    "comparator is not a certified cointegration test. Overlapping cohorts use one funding pool; "
    "monthly-mean HAC inference is asymptotic, uses a declared six-month lag and refuses short samples. "
    "Training selection and model risk remain outside that interval.",
    "",
    "**Unmet external gates:** no qualified raw historical adjusted-price feed, complete historical "
    "security mapping, corporate-action/delisting cash-flow record, borrow data or benchmark series "
    "was supplied. Bundled constituent coverage starts in 1995 and cannot establish earlier coverage. "
    "Consequently genuine historical returns, robustness, turnover/capacity and dependence-aware "
    "market comparisons remain unexecuted. The full Clegg 2015 source could not be verified. "
    "No historical profitability, deployment readiness or complete author-table reproduction is claimed.",
    "",
    "## Reproducibility and invalidation",
    "",
    "Audit baseline: bef2b7d (production baseline 1501aa6). Core corrected estimator: 7a321fa; "
    "workflow integration: bf1ead1. The report records the exact execution-byte source SHA "
    "and an LF-normalized SHA for portable code matching. Source/configuration changes require "
    "new calibration. The full study is checkpointed; reported invocation times may reuse "
    "checkpoints and are not total computation time. Large transient arrays remain outside Git; "
    "digest manifests are committed. A first report-save attempt failed on a NumPy boolean; "
    "the report was regenerated from intact checkpoints after serialization was corrected.",
    "",
    f"Canonical numerical-source SHA: `{r['canonical_source_hash']}`.",
    "",
    "Reproduce the commands in [README.md](../README.md), using Python 3.12 and the pinned "
    "environment in requirements-lock.txt. Run `experiments/summarize_validation.py` after "
    "the complete experiments to regenerate this report and figure. Tests, notebooks and "
    "configured empirical command checks are independent commands; do not infer their success "
    "from a Monte Carlo report alone.",
    "",
    "Prior cached estimates, critical values, tables, signal paths and backtest returns are "
    "invalidated for corrected-model conclusions. Archived research and audit evidence retain "
    "their historical status. No evidence was relabeled to make an unmet gate pass.",
    "",
]
(root / "docs/validation-report.md").write_text("\n".join(lines), encoding="utf-8")
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
for rho in [-0.8, 0, 0.6, 0.9, 0.99]:
    cells = sorted(
        [c for c in r["recovery"] if c["rho"] == rho and c["sigma_m"] == 1],
        key=lambda c: c["n"],
    )
    for ax, index in zip(axes, [0, 1]):
        ax.plot(
            [c["n"] for c in cells],
            [c["rmse"][index] for c in cells],
            marker="o",
            label=f"rho={rho}",
        )
for ax, title in zip(axes, ["Beta RMSE", "Rho RMSE"]):
    ax.set(
        xscale="log", yscale="log", xlabel="Observations", ylabel="RMSE", title=title
    )
    ax.grid(alpha=0.25)
axes[1].legend(fontsize=8)
fig.suptitle("Corrected model: equal transient/permanent noise; 1,000 draws per cell")
fig.savefig(results / "recovery-summary.png", dpi=160)
plt.close(fig)
print("Rendered validation report and recovery figure")
