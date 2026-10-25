# Code and model review — first pass

This change repairs the active pipeline while preserving its public class and method names.

## Fixed

- `f_create_Mt` selected the second Kalman output (R, the permanent random walk), although trading indicators expect the first output (M, the transient component). It now selects M and retains price dates in both returned frames.
- Training stocks were filtered for completeness over both training and future test observations. Universe completeness now uses training data alone; missing test prices remain visible for explicit downstream handling.
- Failed pair estimates were initialized as zeros, and total optimizer failure raised an unrelated undefined-variable error. Failures now remain NaN.
- Both active estimator implementations now convert input series to numeric arrays, validate paired inputs, and avoid pandas label/position ambiguity inside the filter.
- Rolling normalization uses pandas rolling calculations instead of a Python window loop, retaining population-window behavior with sample standard deviation (ddof=1), a NumPy return value, and missing scores for constant windows.

## Validation

Run `python -m unittest discover -s tests -v` with numpy, pandas, scipy, statsmodels and tqdm installed. Six tests cover future-data isolation, transient-component extraction, date preservation, random-walk boundary behavior, decomposition reconstruction, failed fits, and rolling-score causality. These tests do not fetch market data or place trades.

Local validation used Python 3.12, numpy 2.5.4, pandas 3.1.0, scipy 1.18.1, statsmodels 0.15.0 and tqdm 4.70.1. The repository's older pinned environment has not yet been exercised.

## Open model and backtest work

- The framework and simulation estimator duplicate most calculations and have different `fit_mle` return contracts. Consolidation needs a compatibility layer for existing notebooks.
- The likelihood uses sigma_M squared plus sigma_R squared as innovation variance. Its equivalence to the intended steady-state filter likelihood needs checking against the original research/reference implementation before changing estimates or claiming statistical validation.
- Lag-variance initialization can divide by zero or yield invalid bounds; random restarts are unseeded, and final PCI fits do not check optimizer convergence.
- The full-model volatility bounds exclude zero, although the null and pure-AR models lie on those boundaries. Nested-model likelihood ordering and the existing negative log-ratio convention require dedicated tests.
- Out-of-sample filtering and rolling normalization restart instead of carrying training history into evaluation.
- Trading signals execute at the same price observation used to calculate them. Execution timing, negative hedge ratios, portfolio cash sharing, and missing-price policy need explicit decisions and end-to-end tests.
- Annual constituent selection includes stocks joining later that year; point-in-time membership must be enforced before a historical performance claim.

The existing separation between data, estimation and strategy is useful; it provides clear places to add these checks without rewriting research notebooks immediately. Overall review verdict: further changes required before relying on backtest performance. No performance improvement or complete model verification is claimed by this patch.
