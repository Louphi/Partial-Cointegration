# Implementation status

Audit baseline: bef2b7d (production baseline 1501aa6). Branch: codex/partial-cointegration-rework. Core numerical implementation: 7a321fa; workflow integration: bf1ead1; checkpointed studies and entry points: 68ae98b; source-bound calibration loading and measured evidence: 2dc1f57. The final delivery commit contains the completed legacy study and documentation. Original audit artifacts remain historical. Final software evidence: 193 passing tests, 96% package coverage, strict type/style checks; 731,100 final-design fits with no failed fits.

| Task | Dependencies | Acceptance | Status | Implementation/evidence |
|---|---|---|---|---|
| T00 | none | Supported environment, imports, tests | Complete | Fresh Python 3.12 environment, pinned extras/lock, editable install, pip check, engine replay and UI/notebook execution. |
| T01 | T00 | Explicit normalization, initial state, mode contracts | Contract complete; S2 verification unavailable | Exact conditional alpha=0 and common boundary prior; separate legacy scorer. |
| T02 | T01 | >=50 independent dense-reference fixtures; boundary/PSD tests | Complete for stated model | 60 independent dense-Gaussian fixtures; Python/compiled comparisons; domain/state tests; 60 executed author-written R gain/scorer fixtures under matched conventions. |
| T03 | T02 | Structured failures, reproducible fits, >=100 nested comparisons | Complete within multistart scope | 100 nested comparisons, forced failures and two rare failure-seed regressions; 81,000 accepted final recovery/boundary fits. No global-optimum proof. |
| T04 | T00,T01 | Aligned inputs, point-in-time membership, future invariance | Software complete; provenance blocked | Alignment, membership, missing-quote and future-invariance tests. Genuine data/mapping/calendar/action/delisting verification remains external. |
| T05 | T03 | Independent 10k calibration/evaluation draws and size gates | Complete for specified RW null/design | 10,000+10,000 independent null fits at each of three n; all nine familywise uncertainty-aware gates passed. No universal null/nuisance claim. |
| T06 | T02,T04 | Continuation, missing observations, causal normalization | Complete | Posterior mean/covariance and history continue; missing observations predict only; timestamp and identity checks. |
| T07 | T03,T05 | Immutable summaries, full recovery/coverage/reproduction experiments | Declared computational studies complete; author-table certification unverified | 75+6 cells ×1,000; 100 outer fits ×500 bootstrap fits; all prespecified subset gates passed. Legacy comparison: 51 cells ×10,000 plus 30,000 independent calibration fits; all three size gates passed. Executed R scorer parity under matched conventions; full original optimizer/table parity not certified. |
| T08 | T03,T04,T06 | Next-observation fills, ledger oracle, shared cash and costs | Complete for daily-close scope | Hand-calculated cash/P&L, actual vectorbt replay, later fills, costs/borrow, rejections, settlement, overlap funding. Intraday margin calls/financing outside scope; beta<=0 trading rejected. |
| T09 | T04,T05,T07,T08 | Prespecified variants; empirical evaluation | Software/design complete; qualified market evaluation blocked | Frozen configs, causal selection, OLS level-spread/no-trade baselines, cash-funded overlap and declared-lag HAC. Two-cohort synthetic command flow passed. A subsequent GOOG/GOOGL fixed-pair public-price illustration ran on 2019–2023 history with an independent cash oracle; it does not establish provenance-qualified universe-wide robustness/capacity/dependence. See results/real-pair-goog-googl.json. |
| T10 | T02,T03,T04,T06,T08 | Wrappers, app, fresh-kernel notebooks, pair-index/UI regression | Complete for active interfaces | Deliberate adapter shapes; batch index/failure tests; standalone app flow; two notebooks executed in fresh kernels. Superseded archives remain recoverable in Git history; see cleanup.md. |
| T11 | upstream | Validation/migration docs and all executed gates mapped | Complete for bounded synthetic-research delivery | Validation/migration/data/model docs, measured manifests, compact auditable null banks, twelve independently checked cutoffs/counts, README claim corrections and updated draft PR. Empirical/source limitations remain explicit; do not merge. |

## Findings A01–A19

| Finding | Correction/evidence |
|---|---|
| A01 | Covariance-aware density, independent dense likelihood checks and recalibrated inference. |
| A02 | Explicit alpha normalization; common exact boundary prior. |
| A03 | Admissible nested candidates, finite convergence checks and rejected-search diagnostics. |
| A04 | Domain/degenerate-input errors, boundary candidates and rare-search regression tests. |
| A05 | Labeled dates validated before conversion; mismatches/duplicates rejected. |
| A06 | Posterior and causal normalization history continued into evaluation. |
| A07 | Copied named truth and pure/idempotent MSE summaries. |
| A08 | Direct failures, batch NaNs/errors, explicit denominators and unconditional rejection bounds. |
| A09 | Decision/later-fill timestamps and independent execution ledger. |
| A10 | Negative beta estimable; unsupported beta<=0 trading rejected. |
| A11 | Shared cash/capacity within portfolios and across funded cohorts. |
| A12 | Immutable formation-end membership excludes future entrants. |
| A13 | Predict-only missing observations, no fabricated fills, official terminal settlement required. |
| A14 | Distinct named variants; ranking separated from significance; original parity unverified. |
| A15 | One active package; facades/app/notebooks migrated; supported environment and R reference execution. |
| A16 | Unsupported efficiency/profitability claims removed; measured evidence and limits separated. |
| A17 | Seeded DGPs and independent RNG streams; initialization/design/source metadata. |
| A18 | Explicit pair identities preserve reordered/subset batches. |
| A19 | UI/calculation window>=2 and executed standalone app flow. |

Missing external inputs: raw adjusted-price history, point-in-time mapping, corporate-action/delisting records, borrow/calendar/benchmark data; full Clegg 2015 text. These block genuine empirical/source verification only. All corrected-model computational experiments above ran at full declared size. See validation-report.md for exact rates, uncertainty, early failures, version/configuration identity and invalidated prior conclusions.
