# Research audit

Date: 2026-10-25. Branch: `codex/partial-cointegration-rework`. Audited HEAD: `1501aa6bb0b246ec30a8e3b305743cc58b986b5b`; main baseline: `8d37c814cf3f20dca22cc868d938365abfe45bbb`. No production code was changed during this audit.

## Assessment

The project is a useful exploratory implementation with separable data, model, simulation and strategy components, but it is not yet a scientifically validated research package. The primary blockers are a likelihood that does not match the stated latent-state Gaussian model, an unidentified fitted intercept, and incomplete causal execution/portfolio accounting. Fixing them requires new simulation calibration and invalidates direct reuse of existing numerical results.

The existing separation of simulation generation from fitting, explicit permanent/transient components, and PR #1 regression checks provide a useful foundation. Overall review verdict: **request changes before relying on statistical or trading conclusions**.

## Baseline and reproducibility

The working tree was initially clean. No tracked AGENTS.md was found in the repository inventory. Source hashes, runtime versions and execution results are in [diagnostic-results.json](diagnostic-results.json). This audit uses the code-reviewer workflow plus independent mathematical derivation, not a style-only review.

Python 3.12.14 on Windows; core audit environment: numpy 1.26.4, pandas 2.2.0, scipy 1.12.0, statsmodels 0.14.1, tqdm 4.66.1, matplotlib 3.8.2, openpyxl 3.1.2. These match the relevant repository pins. Added xarray 2024.2.0 because simulation code imports it but requirements.txt omits it. Remaining listed application/network dependencies were not all installed in this core environment. A separate backtest probe environment is described in trading-diagnostics.json and its execution note below.

The full original requirements file is not a lockfile: it mixes pinned and unpinned packages, includes standard-library `datetime` and `importlib`, and omits xarray, vectorbt, streamlit and plotly. The app hardcodes another developer's absolute macOS data directory. `adjPrice_nasdaq100.pkl` is absent. The bundled `data.pickle` is an undocumented fit cache, not evidence of raw-price provenance; this audit does not deserialize it. Constituent CSV inspection finds 485 rows, 408 unique co_tic values, 101 open intervals and earliest start 1995-01-01. Coverage of earlier years is therefore not established by this file.

### Execution map

| Area | Execution path | Role and limitations |
|---|---|---|
| Core model | `PCI Framework/PCIPipeline.py` → `Computation` | Pair estimation, filtering, eligibility and score preparation; class combines model and orchestration. |
| Data | `PCI Framework/PCIData.py` | Constituent intervals, but annual membership includes later entrants and uses co_tic rather than an explicit point-in-time security identifier. |
| Strategy | `PCI Framework/PCIStrat.py` | vectorbt indicators and single-pair backtests. No common multi-pair capital policy is implemented. |
| App | `PCI Framework/app.py` → data/model/strategy | Streamlit UI; external price file and cached fits required; `streamlit run` is the intended launcher. |
| Simulation | `Simulation/Functions/PCISimulation.py` → `PartialCointegration.py` → `ResultReplication.py` | Synthetic generator, second estimator and summaries. `HowToUse.ipynb` calls this path. `ResultReplication.py` main defaults to 10,000 fits for one n=1000 cell, not the full documented grid. |
| Notebook copy | `PCI Framework/VBT_TMP.ipynb`, cell 0 | Defines another copy of framework classes. Later cells use hardcoded paths and cached fits. Editing the Python module does not repair this notebook's class definitions. |
| Archives | `archives/`, `Simulation/Archives/` | Earlier model, clustering, strategies and critical-value experiments; not a supported production API. Relevance must be catalogued before removal. |

### Claims recovered

The README aims to reproduce simulation tables 3–6 and interprets finite-sample mean/MSE and rejection frequencies as evidence for estimation quality and trading-screen choices. Its text says beta is OLS-estimated, although both active fit_pci implementations optimize beta jointly after the OLS initialization. The app explores a Nasdaq single-pair strategy. It does not implement a complete paper-equivalent portfolio experiment. No repository-specific verified profitability claim is established here.

## Evidence executed before corrections

Acceptance criteria were saved first in [validation-protocol.md](validation-protocol.md). Read-only diagnostics are in `tools/audit/diagnose.py` and `tools/audit/trading_diagnose.py`.

- Six pre-existing regression tests pass in the pinned core environment.
- Six deterministic likelihood cases: exact matrix recursion agrees with an independently constructed dense Gaussian density; fixed production gains agree with Riccati steady-state gains. Mixed-model scoring variance does not agree. Pure RW/AR cases agree under the matched initialization used by the diagnostic.
- Alpha-shift, degenerate-data, failed-final-optimizer, date-alignment, future-membership, state-reset, pair-index and repeated-MSE probes were executed.
- Six n=100 seeded baseline fits completed and returned finite outputs, with internal optimizer failures/warnings recorded. Both final optimizer calls reported success in these six cases; this does not contradict the separate forced-failure acceptance probe. One fit reaches rho=-1. The experiment is too small to assess recovery or inference.
- A reduced notebook calculation generated two replications per sigma_M cell {.5,1.5}, for n={100,1000}, and computed eight fits, means and MSEs. It runs, but is not a reproduction of 10,000-replication published tables. At n=1000 the two-cell mean rho values are approximately .9109 and .9001; these are descriptive outputs, not acceptance evidence for consistency.
- The actual vectorbt 0.26.2 engine was run on five synthetic daily observations. It fills entry signals on the signal date; allocates one independent unit of cash to each of three price columns (including an untraded column); closes both positive-beta legs at the supplied exit; and rejects beta=-1 with a negative-size error. See [trading-diagnostics.json](trading-diagnostics.json). Engine imports initially failed with unconstrained Plotly 7.1.0 (`heatmapgl` template incompatibility); a separate Plotly 5.18.0 audit overlay resolved that import. No production dependency file was changed.
- Full tables, test-size/power calibration, uncertainty coverage and historical market performance were **not** reproduced. Missing raw market prices block the historical experiment; expensive Monte Carlo tasks are explicitly planned rather than claimed complete.

## Findings

Line numbers below refer to the audited commit. S1–S4 refer to [sources.md](sources.md); M refers to [mathematical-specification.md](mathematical-specification.md). Each evidence key names a reproducible diagnostic, source trace or both.

### A01 — Likelihood omits hidden-state uncertainty

**Critical; mathematical/statistical defect; confidence high.** Location: PCIPipeline.py:260–281 and PartialCointegration.py:284–305. Evidence: `likelihood_comparisons`, M2–M5. At rho=.6,sigma_M=sigma_R=1, the steady-state prediction variance is 2.19224994, not 2. The fixed-gain states agree to approximately 7e-14, isolating scoring variance from filter-gain algebra. Correct scoring changes the objective by a parameter-dependent amount. Reference code shares the convention, so parity with S3/S4 does not establish exact likelihood validity. Consequence: estimates, ratios and calibrations need reevaluation. Resolution: exact conditional likelihood plus explicitly labeled legacy objective; recalibrate tests (tasks T02,T05).

### A02 — Estimated alpha is unidentified in the mixed model

**High; mathematical/statistical defect; confidence high.** Location: PCIPipeline.py:331–344,232–236 and corresponding simulation methods. Evidence: `alpha_profile` and M section 2 show errors invariant to constant shifts. Returned alpha is an arbitrary initialization convention, not an estimated equilibrium level. Exact-zero sigma_R also switches the initial state. Consequence: singular information, misleading reported parameters and inconsistent boundary comparisons. Resolution: normalize location and specify one initialization across nested models; document any separately identified intercept variant (T01,T02).

### A03 — Bounds and final convergence checks are unsound

**High; confirmed implementation defect; confidence high.** Location: PCIPipeline.py:338–357; PartialCointegration.py:55–74. Evidence: source bounds require both sigmas>=.0001 while null fitting fixes sigma_M=0; `failed_final_fit_accepted` forces success=False yet fit_pci returns a normal result tuple. Thus the purported unrestricted search excludes exact nested boundaries and may accept failed/nonfinite final fits. Resolution: explicit boundary candidates, finite checks, observable convergence status and nested-score checks (T03). This audit did not observe a positive ratio in the fourteen ordinary fits; the defect is established by bounds and forced failure, not an invented empirical violation.

### A04 — Degenerate/domain inputs are not handled deliberately

**High; confirmed implementation defect; confidence high.** Location: PCIPipeline.py:147–183,203–281 and duplicate simulation methods. Evidence: `lagvar_constant` and `lagvar_linear` return NaN rho; `zero_noise_ll` raises ZeroDivisionError; `negative_noise_filter` accepts a negative standard deviation. Domain checks need to cover rho, noise scales and positive finite innovation variance, not just input-array finiteness. Consequence: invalid starts and undefined objectives. Resolution: explicit domain/result types and supported degenerate-data policy (T03).

### A05 — Array conversion loses date alignment

**High; confirmed interface defect; confidence high.** Location: constructors PCIPipeline.py:79–86 and PartialCointegration.py:12–20. Evidence: `misaligned_input_accepted` pairs reversed-index Series and returns beta=-1 instead of rejecting the mismatch. PR #1 correctly fixes positional indexing, but silently discards index semantics. Internal selection from a common DataFrame is aligned; direct callers are not protected. Resolution: validate identical, monotone, unique Series indices before conversion; keep arrays as an explicitly positional API (T04).

### A06 — Evaluation resets the filter and normalization history

**High; confirmed implementation behavior requiring correction for continuity; confidence high.** Location: PCIStrat.py:10–14,124 onward. Evidence: `filter_reset` gives continuous first test M=-.807318 versus restarted M=0 on the same observations. Two separate f_create_Mt calls initialize the decomposition twice; indicator calls also reset rolling history. Consequence: initial evaluation signals depend on an artificial restart. Resolution: carry state/covariance and chosen training normalization/history across the boundary (T06).

### A07 — MSE computation mutates the truth specification

**High; confirmed implementation defect; confidence high.** Location: ResultReplication.py:119–166. Evidence: `mse_repeat` uses exact truth estimates; first-call MSEs are zero, while repeating the same call gives sigma_M MSE=1 in the first cell because params now holds the final cell's truth. Positional broadcasting also temporarily subtracts sigma_X from LRT before dropping LRT. Consequence: notebook order changes reported simulation summaries. Resolution: immutable named truth coordinates, explicit parameter selection, idempotency tests (T07).

### A08 — Failed estimates are silently excluded or counted as non-rejections

**High; statistical reporting defect; confidence high.** Location: ResultReplication.py:107–166,196–223. Evidence: xarray floating mean defaults skip NaNs; `missing_lrt_power` compares one rejecting LRT and one NaN with a patched fixed critical value and returns .5. Missing is thereby treated as non-rejection in the numerator while still in the denominator. No failure count accompanies the result. Consequence: conditional survivor summaries and potentially distorted rejection rates. Resolution: report failed counts/reasons, success-conditional values and unconditional worst-case bounds; calibrate only with an explicit failure policy (T05,T07).

### A09 — Signal observation and execution are not separated

**Critical for performance claims; confirmed execution design defect; confidence high.** Location: PCIStrat.py:27–36,56–65,99–118. Rolling scores include the current price; signals and sizes are passed without lag to from_signals using the same price rows. Minimal reproducer: an entry on 2023-01-02 fills on 2023-01-02 in `positive_beta.orders`. Consequence: no established attainable fill convention. Resolution: explicit signal_time/decision_time/fill_time, next-observation execution and no-final-bar fill tests (T08).

### A10 — Negative beta is allowed in estimation but not in order construction

**High; confirmed implementation defect; confidence high.** Location: PCIPipeline.py:342; PCIStrat.py:90–105. beta is unrestricted, but S1 always receives a long-entry flag and size beta/X2. Minimal reproducer: beta=-1 yields a negative amount with a long flag; the real engine raises `ValueError: Negative size is not allowed. You must express direction using signals.` Consequence: a supported model parameter cannot be traded through this interface. Resolution: initially restrict tradable pairs to beta>0, or implement a fully signed-position ledger with gross exposure limits (T08).

### A11 — Single-pair output is not a shared-capital portfolio

**High; empirical/design limitation; confidence high.** Location: PCIStrat.py:69–120 and app.py backtest invocation. The code sends all selected stock columns to from_signals but constructs signals for one pair and sets init_cash=1 without explicit cash sharing/grouping. Minimal reproducer: add an untraded third price column; inspect per-column cash and portfolio values in the real-engine probe. Consequence: aggregation/denominator can include independently funded or idle columns, and repeated stocks across pairs have no allocation policy. Resolution: explicit per-pair and aggregate ledgers, one capital budget and tested accounting (T08,T09).

### A12 — Universe selection uses membership information from later in the year

**High; confirmed methodology defect; confidence high.** Location: PCIData.py:25–30 and app.py:28. Evidence: `future_membership_selected` includes a stock entering June 2023 in a formation decision preceding January 2023. The daily helper also mutates open intervals into today's date, making reuse order-dependent. Consequence: point-in-time eligibility is not established. Resolution: membership at the formation-end timestamp, immutable intervals, stable security identifiers and data coverage checks (T04).

### A13 — PR #1 exposes missing evaluation prices without an execution policy

**High; incomplete integration after a justified fix; confidence high.** Location: PCIPipeline.py:50–65,446–470; PCIStrat.py. Existing regression confirms future NaNs no longer change training columns. However f_create_Mt now rejects nonfinite prices; the app has no documented suspension/delisting policy. Minimal reproducer: retain B with complete training and one test NaN, then pass test prices to f_create_Mt. Consequence: eliminating leakage can now stop evaluation, which is preferable to silent leakage but incomplete. Resolution: predict-only missing observations where valid, prohibit fills on missing quotes, explicitly value/liquidate delistings (T04,T06,T08).

### A14 — Current selection and score are a strategy variant

**Medium; modeling choice requiring justification; confidence high for code trace.** Location: PCIPipeline.py:405–429,474–490; app.py sliders. Source selects rho>.5 and variance fraction>.5, ranks raw LRT, then keeps ten; it has no explicit bottom-five-percent stage, sector constraint, or training-performance stage. The score is rolling centered/standardized M, not formation-fixed scale M. Ordered pairs can repeat economic exposure. S1 comparison locations are in sources.md. Consequence: do not label this paper-equivalent trading replication. Resolution: separate named legacy/reproduction and rolling-score variants; prespecify orientation, multiplicity and validation selection (T09).

### A15 — Code duplication and environment gaps prevent reliable maintenance

**High; confirmed reproducibility defect; confidence high.** Locations: both active estimator files, VBT_TMP.ipynb cell 0, requirements.txt, app.py:18–20. Framework fit_mle returns one vector while simulation fit_mle returns two, so naive deduplication changes callers. Raw price path is machine-specific and absent. Minimal reproduction: an environment built only for the declared core cannot import the simulation without xarray or the app without its undeclared dependencies. Backtest import also needed a compatible Plotly version during this audit. Resolution: one package with adapters, separate core/app dependencies, configuration and reproducible fixtures (T00,T10).

### A16 — README conclusions exceed the evidence

**High; statistical interpretation defect; confidence high.** Location: README Simulation and Table 3/4 conclusions. It describes joint-fitted beta as OLS-only, treats skewness as evidence of bias, and infers trading adequacy from finite-sample MSE. Symmetry and bias are different properties; no finite collection of MSE observations proves consistency, asymptotic normality or efficiency. At q=0/rho=1 some parameters are unidentified, rather than simply universally inconsistent. Resolution: rewrite claims around bias estimates/MC error, failure rates and identification domains (T11).

### A17 — Simulation conventions and random streams are undocumented

**Medium; modeling/reproducibility limitation; confidence high.** Location: PCISimulation.py:29–38,63–67; fit_mle random draws; ResultReplication.py:12–105. AR and RW generation start from zero before the first drawn innovation, whereas the mixed filter imposes Mhat_0=0 at the first observed sample. Global RNG controls data and optimizer restarts, so algorithm changes can change future generated datasets. Minimal reproduction: insert an extra RNG draw before generation or fitting and compare outputs. Consequence: source-table differences cannot be attributed solely to estimation quality. Resolution: explicit initial-state regimes, independent Generator/SeedSequence streams and saved seed/configuration metadata (T07).

### A18 — Subsetting pair indices breaks batch estimation

**Medium; confirmed implementation defect; confidence high.** Location: PCIPipeline.py:381–402. Evidence: `subset_pair_index` passes a valid Series slice with index 1; pairName[0] fails and the error handler raises UnboundLocalError because pairName_i was never set. Consequence: selected/reordered batches can fail before producing missing results. Resolution: row-based iteration with stable pair IDs and structured per-pair errors (T10).

### A19 — UI still permits a rolling window rejected by PR #1

**Medium; confirmed integration defect; confidence high.** Location: app.py Window Size slider and PCIPipeline.py:486. Slider permits window=1; rolling_z_score now raises ValueError for it. Existing unit test confirms rejection but no UI adjustment accompanied it. Consequence: a user-selectable option crashes the calculation. Resolution: align UI constraints, surface validation messages, add a minimal app-flow check (T10).

## Independent assessment of PR #1

Correct in the tested cases: selects M rather than R; preserves output dates; excludes future completeness from training selection; stores failed batch estimates as missing; fixes a formerly undefined failure variable; and removes a rolling Python loop. These are worth retaining.

Limits: six tests cover a pure-AR signal, pure-RW/reconstruction identities, basic splitting, mocked failures and one rolling score. They do not test mixed-model likelihood, parameter identification, inference, covariance, execution, raw-data quality, full portfolio accounting, or the copied notebook implementation. A05, A13 and A19 are integration gaps introduced/exposed by that patch. None warrants restoring the former leakage or wrong-component signal.

## Defaults and open questions

Recommended defaults: exact conditional model with alpha normalized to zero; retain a labeled legacy scorer for reproduction; bootstrap calibration for the new estimator; beta>0 trading restriction initially; next-close execution using available close data; formation-end membership; explicit missing-price handling; one portfolio capital ledger. These are proposed implementation choices, not retroactive descriptions of the paper.

Still unresolved: full Clegg 2015 source access; raw-price provenance/licensing and corporate-action/delisting fields; whether the desired deliverable is exact paper replication or a new Nasdaq strategy. Proceed with separate named experiment modes and synthetic validation until these inputs are supplied. They do not block the mathematical/software corrections.

See [correction-plan.md](correction-plan.md) for dependency order, quantitative completion gates and blocked empirical work.
