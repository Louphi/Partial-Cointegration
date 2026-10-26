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

An [exploratory real-price example](docs/results/real-pair-goog-googl.json) uses GOOG/GOOGL daily Yahoo closes: formation 2019–2022, evaluation January–June 2023, fixed formation scaling and entry/exit thresholds. On $10,000, it returned −0.52% after assumed fees, slippage and borrow, completing one round trip. Pair, period and rules were fixed before computing the result. This is a retrospectively chosen historical example, not a qualified universe-wide performance study or untouched future test. Borrow availability and actual broker costs are unverified.

Run `.venv/Scripts/python experiments/real_pair_example.py` to reproduce it. The command keeps raw vendor responses, prices and full ledgers in ignored `.test-deps/real-pair-goog-googl/`, records source digests in the compact report and independently reconstructs daily cash/equity from executed orders. It uses the Close field rather than dividend-adjusted Adj Close; vendor records contain no evaluation-period dividends/splits. Deleting cached responses downloads a new vendor snapshot and can change results. No pair or threshold search is performed.

The [longer daily example](docs/results/longer-pair-goog-googl.json) evaluates 2019–2023, refitting the preceding four years every six months and rolling one capital account forward. With the same pair and rules, $10,000 became $9,234.47 (−7.66%), across 19 completed round trips and 56 rejected requests. Reproduce with `experiments/longer_pair_example.py`. This remains a retrospective, fixed-pair illustration with assumed costs.

The [one-minute prototype](docs/results/minute-pair-goog-googl.json) fits three complete sessions and evaluates two later sessions from a cached five-day public-price snapshot. It shifts vendor bar-start timestamps to completed-bar end, requires all 390 regular minutes per session, closes positions daily, and uses 252×390 borrow observations/year. It completed two round trips and returned −0.78%, with 54 rejected requests. The fixed initial-capital sizing/capacity policy materially limits executions after losses; these results do not describe all signal opportunities. Reproduce with `experiments/minute_pair_example.py`. Two sessions, session-compressed overnight filtering and assumed close-price fills cannot validate an intraday strategy; actual bid/ask quotes, latency, availability and a longer archive remain unverified. Minute parameters are refitted rather than copied from the daily model.

The same runners accept `--pair pep-ko` for PepsiCo/Coca-Cola. The [2019–2023 daily illustration](docs/results/longer-pair-pep-ko.json) completed six round trips: $10,000 became $11,226.12 (+12.26%) after assumed costs and net dividend accrual of −$88.46. Longs receive and shorts owe per-share dividend entitlements, accrued as cash on ex-dates; payment-date receivables, taxes and broker funding treatment are not modeled. Both fitting/signals and fills use Close, avoiding a hedge-unit mismatch between dividend-adjusted prices and literal prices. The [minute sample](docs/results/minute-pair-pep-ko.json) selected a pure-RW fit and remained in cash, without forcing trades. These are user-selected retrospective examples, not pair selection or profitability certification. [Graphs](docs/backtest-graphs.md) can be regenerated with `experiments/plot_pair_results.py --pair pep-ko`.

## Research status and documentation

- [Implementation status](docs/implementation-status.md): task-by-task acceptance and evidence.
- [Validation report](docs/validation-report.md): executed experiments and limitations.
- [Migration guide](docs/migration-guide.md): intentional interface and scientific changes.
- [Mathematical contract](docs/model-contract.md): exact model, profiling, initialization and legacy mode.
- [Original audit](docs/audit/research-audit.md) and [correction plan](docs/audit/correction-plan.md): the baseline defects and planned acceptance gates.

The original research was developed by Sébastien Caron, Antoine Chartrand-deniger and Philippe Gagné. The initial reference papers and author-written R packages are registered in [audit sources](docs/audit/sources.md). Superseded notebooks, caches and tables were removed from the working tree; the [cleanup record](docs/cleanup.md) explains how to recover them from Git history. The original presentation and constituent source CSV remain. No claim of asymptotic efficiency, complete statistical validity, historical profitability or production readiness follows from the current software alone.
