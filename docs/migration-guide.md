# Migration from the inherited framework

Scientific outputs must be regenerated. Keep original reports in Git history with their original commit/configuration; do not rename them as corrected results.

| Previous behavior/interface | Corrected behavior |
|---|---|
| Free alpha optimized with a flat mixed-model likelihood | Alpha fixed at zero and labeled as normalization; initial permanent level absorbs location. |
| Fixed-gain likelihood using sigma_M^2+sigma_R^2 | Default exact conditional likelihood with state covariance; explicit legacy comparison mode remains. |
| Unchecked final optimizer tuple | `fit_pair` returns FitResult with accepted candidate and diagnostics; invalid/no accepted fit raises FitError. |
| Framework fit_mle returns one vector | Compatibility wrapper retains one rho/sigma_M/sigma_R vector. |
| Simulation fit_mle returns two vectors | Simulation adapter retains alternative/null vectors; both delegate active mathematics to the package. |
| Failed fit silently represented as ordinary output | Direct fits raise; batch adapters preserve NaN rows and expose `last_batch_errors`. Modern result summaries include failure counts. |
| Pandas indices discarded before alignment checks | Mismatched, duplicate or decreasing labeled indices are rejected. |
| Annual membership includes future entrants | Year helper uses formation-end membership preceding that trading year; prefer explicit `members_at(timestamp)`. |
| Ordered pair permutations | One deterministic unordered orientation; this changes the universe and is documented as a strategy variant. |
| Filter/rolling history restarted at evaluation | Means/covariance and history continue; missing observations predict only. |
| Permanent R accidentally used for trading | Transient M is used. |
| Same-row signal/fill and per-column cash | Later-quote execution and one shared portfolio ledger. |
| Negative beta sent as negative long size | Estimation allows it; the initial trading layer explicitly rejects beta<=0. |
| `pf` is necessarily a vectorbt Portfolio | Compatibility strategy returns a LedgerResult with `.value()`, `.stats()`, `.orders`, `.rejections`, `.holdings`; optional `replay_vectorbt` checks complete-price ledgers. Engine-specific `.vbt`/trade-record calls require migration. |
| f_compute_powers_lrt silently runs calibration again | Supply independent `critical_values` with matching estimator fingerprint, mode, x design and initialization. Missing fits are counted and bounded. |
| f_compute_mse mutates a six-element truth list | Truth is copied into named coordinates; repeated calls are idempotent. |
| Global random state controls generation/restarts | Explicit seeds and independent SeedSequence streams. Repeated same configuration reproduces the same data. |
| Default window can be one | Window>=2 enforced in both UI and calculations. |
| Hardcoded local data and implicit pickle loading | CSV/parquet configuration; historical pickle loading only through explicit trusted_pickle=True. Fit-cache provenance cannot be inferred from an old pickle. |

## Installation and examples

Install the package in Python 3.12 using the documented extras. The two active notebooks now import the package and contain small synthetic examples, with stale outputs cleared. Their full-kernel execution is recorded in `docs/results/notebook-app-verification.json`. Archives remain historical and are not supported as active APIs.

The named dataclasses Parameters, FitOptions, FitResult and FilterState replace anonymous tuples in new work. Old import paths still work after package installation. Use `.legacy_tuple()` only at the compatibility boundary. Exact and legacy modes require separate caches/calibration; never combine likelihoods from different modes or initializations in one ratio.

## Data and study inputs

Follow `docs/data-contract.md`. Existing `co_tic` is not assumed to be a complete historical identifier mapping. The 1995 start of bundled constituent coverage must be respected. Raw historical adjusted prices, corporate-action/delisting and benchmark fields are not supplied by this migration.

`experiments/configs/paper-inspired-reference.json` preserves some published-style choices while intentionally adopting corrected execution and allocation. It is not an exact replication configuration. The Nasdaq extension and the same-selection OLS-spread comparator are labeled separately. No-trade is the pure-RW benchmark.

## Invalidated conclusions

Prior point estimates, likelihood ratios, critical values, power tables, signal paths and backtest returns are not validated outputs of the corrected method. Skewness does not establish bias; empirical MSE does not prove consistency or efficiency; a selected pair does not establish profitability. Read the final validation report before treating any statistical regime as supported.
