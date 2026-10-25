# Audit evidence and reproduction

Start with [research-audit.md](research-audit.md), then [mathematical-specification.md](mathematical-specification.md) and [correction-plan.md](correction-plan.md). The plan has 12 tasks (T00–T11), mapped to 19 findings (A01–A19). Production mathematical/strategy changes belong to the subsequent implementation phase.

## Executed evidence

| Artifact | What it establishes |
|---|---|
| `diagnostic-results.json` | Baseline/version/source hashes, independent likelihood comparisons, targeted defect probes, six smoke fits and eight reduced-notebook fits. Nonfinite values are encoded as JSON null; accompanying warnings explain them. |
| `trading-diagnostics.json` | Actual vectorbt five-day execution probe: signal-day fills, independent cash columns, functioning positive-beta exits, rejected negative sizing. |
| `baseline-tests.txt` | Six pre-existing tests run under the pinned core scientific versions. |
| `validation-protocol.md` | Numerical and experimental gates declared before diagnostic execution; expensive planned experiments are not represented as executed. |
| `sources.md` | Primary source versions, locations, access limitations and the independent critique of the legacy likelihood. |
| `requirements-audit.txt`, `requirements-trading-audit.txt` | Audit-only environment snapshots; production requirements were not modified. |

The matrix reference and dense shock-loading Gaussian density agree to the declared tolerance in all six selected cases. A near-unit-root Riccati check required increasing the internal iteration cap from 10,000 to 100,000 and using an exact-observation covariance projection to avoid accumulated roundoff; its internal stopping tolerance is 1e-12. The external 1e-9 comparison gate was not relaxed. This is an audit-harness adjustment, not a production correction.

The smoke fits and reduced notebook calculation establish operability only. No 10,000-replication paper table, formal recovery/coverage experiment, hypothesis-test size/power study or historical-market backtest was completed. S2's full text remains unavailable; R reference source was inspected but not run. Original CSV constituents were inspected; undocumented pickle caches were not executed/deserialized.

## Re-run in a fresh environment

From the repository root, using Python 3.12:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r docs/audit/requirements-audit.txt
$env:PYTHONDONTWRITEBYTECODE = '1'
.venv/Scripts/python -m unittest discover -s tests -v
.venv/Scripts/python tools/audit/diagnose.py --smoke
```

The diagnostic writes its results to `docs/audit/diagnostic-results.json`; commit a baseline copy before rerunning on a changed source revision. Wall times are machine-dependent. Source hashes are byte-level and can differ with checkout line-ending settings; compare the Git commit as well.

For real-engine synthetic probes, use a separate environment:

```powershell
python -m venv .venv-trading
.venv-trading/Scripts/python -m pip install -r docs/audit/requirements-trading-audit.txt
$env:PYTHONDONTWRITEBYTECODE = '1'
.venv-trading/Scripts/python tools/audit/trading_diagnose.py
```

The engine may compile numerical kernels on first use. These are small synthetic probes, not claims about investment performance or the full UI. `streamlit` and unavailable market data are not required for these probes.

## Scientific interpretation

The discrepancy is between the stated latent-state Gaussian model and its implemented scoring function. It is not evidence that the fixed-gain formulas are wrong, and matching the author's legacy scorer is a distinct reproduction objective. Correcting the likelihood requires recalibrating inference and rerunning affected scientific results. See the specification before implementing a fix.
