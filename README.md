# Partial cointegration research

This repository fits a price-level relationship as the sum of a transient AR(1) component and permanent random-walk drift. It includes simulation, explicitly calibrated inference, causal signals, a shared-capital execution ledger, and a small research UI.

The corrected default is the **exact conditional Gaussian model**, with alpha fixed at zero as a location normalization and an explicit initial state. The previous fixed-gain/q+r scorer is retained as `mode="legacy"` for comparison. These modes produce different estimates and require separate calibration. Previous fit caches, tables and backtest returns must not be reused as corrected results.

## Setup

Supported environment: Python 3.12. Install into a virtual environment:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[research,app,engine,dev]"
```

`requirements.txt` contains core numerical pins; `pyproject.toml` separates optional research, UI, engine and validation tools. `requirements-lock.txt` records the environment used for the implementation validation.

## Quick use

```python
from partial_cointegration import Parameters, fit_pair
from partial_cointegration.simulation import simulate_pair

pair = simulate_pair(1000, Parameters(1, .6, 1, 1), seed=20261025)
fit = fit_pair(pair.x, pair.y)
print(fit.parameters)
print(fit.log_ratio)  # negative convention: smaller favors the alternative
```

Alpha is not an estimated equilibrium intercept. Rho is not identified under a pure random walk. The variance-fraction statistic describes differenced spread variance, not an OLS R-squared or profit probability. A fitted alternative or a low ranking score does not by itself establish significance or profitability.

`partial_cointegration.io.calibration_from_report(path, n=1000)` loads a source-matched full report and checks the compact score bank's digest and size gates before returning calibration. It refuses unsupported sample sizes and exposes the complete simulation design. These critical values apply to that reported null/design; matching the length of an arbitrary market sample does not establish applicability.

## Verification and experiments

```powershell
.venv/Scripts/python -m pytest -q
.venv/Scripts/python -m mypy src/partial_cointegration --strict
.venv/Scripts/python experiments/verify_notebooks.py
.venv/Scripts/python experiments/verify_empirical.py
.venv/Scripts/python experiments/validate.py --suite complete --workers 4
.venv/Scripts/python experiments/legacy_tables.py --replications 10000 --workers 4
```

The complete synthetic design is checkpointed: 1,000 replications per recovery cell; independent 10,000-draw null calibration and evaluation at each sample size; and a bootstrap-coverage subset with 500 resamples per fit. Numerical density checks are independent of the implementation's filter equations. Passing software tests does not replace those experiments.

The optional author-source scorer check is `experiments/verify_r_reference.py --rscript /path/to/Rscript`, requiring R and its zoo package. It downloads pinned source versions into ignored scratch space and records digests. It matches initialization and the first-density convention explicitly; it does not run the original optimizer. To regenerate the report and figure after experiments, capture pytest coverage output in `docs/results/software-tests.txt`, then run `experiments/summarize_validation.py` with the declared Python environment.

Legacy-score comparison tables use the inherited generation/scoring convention but a different profiled optimizer. They are **not certified reproductions of the author's original algorithm or market results**. See the validation report for what actually ran, acceptance outcomes and remaining limits.

## Data and execution

Historical raw prices are not bundled. The application no longer loads another developer's machine-specific path or an undocumented fit pickle. Start the explicit synthetic demonstration or supply configured data:

```powershell
.venv/Scripts/python -m streamlit run "PCI Framework/app.py"
```

The [data contract](docs/data-contract.md) defines price dates, stable security IDs, point-in-time membership, adjustments and delisting inputs. Formation selection never depends on future completeness. Filter means/covariances continue into evaluation; missing observations are predict-only. Decisions fill at a later observed close. One committed-capital ledger accounts for both legs, costs, margin checks, rejected orders and scheduled liquidation. Nonpositive beta is supported in estimation but currently rejected for trading.

## Research status and documentation

- [Implementation status](docs/implementation-status.md): task-by-task acceptance and evidence.
- [Validation report](docs/validation-report.md): executed experiments and limitations.
- [Migration guide](docs/migration-guide.md): intentional interface and scientific changes.
- [Mathematical contract](docs/model-contract.md): exact model, profiling, initialization and legacy mode.
- [Original audit](docs/audit/research-audit.md) and [correction plan](docs/audit/correction-plan.md): the baseline defects and planned acceptance gates.

The original research was developed by Sébastien Caron, Antoine Chartrand-deniger and Philippe Gagné. The initial reference papers and author-written R packages are registered in [audit sources](docs/audit/sources.md). Archived notebooks/results are historical evidence, not newly verified outputs. No claim of asymptotic efficiency, complete statistical validity, historical profitability or production readiness follows from the current software alone.
