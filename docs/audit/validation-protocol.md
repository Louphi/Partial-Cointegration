# Audit execution protocol (declared before diagnostic execution)

Baseline: 1501aa6bb0b246ec30a8e3b305743cc58b986b5b. No production edits.

Fast numerical checks: compare independent matrix Kalman filter and dense Gaussian covariance likelihood at atol=1e-9, rtol=1e-10 for modest, well-conditioned deterministic examples. Compare steady-state gains to converged Riccati gains at atol=1e-9. Require reconstruction error <1e-10. Alpha-shift invariance is detected at atol=1e-9. Zero total noise must eventually raise an explicit domain error; existing nonfinite behavior is recorded as a finding, not hidden.

Baseline smoke experiment: seed 20261025; n=100; three simulated pairs per rho in {0.6, 0.9}, sigma_M=sigma_R=1, beta=1, alpha=0, sigma_X=.0236. Record every fit, wall time, warnings, optimizer success, estimate and likelihood. This small experiment measures operability and exposes failures; it is not a test of consistency, recovery accuracy, or hypothesis-test size/power.

Full validation design (not executed in this audit): independent calibration and evaluation seeds; 10,000 null draws for each calibration cell and 10,000 evaluation draws. For nominal alpha in {.01,.05,.10}, require the nominal probability to lie within a predeclared 99% Wilson evaluation interval augmented by independent calibration uncertainty (estimate from 200 calibration resamples); flag rather than tune failures. Adjust simultaneous comparisons with Holm or report familywise simulation envelopes. Report failed fits separately and both worst-case rejection bounds; no dropping failures silently.

Recovery experiment: at least 1,000 replications/cell; n in {100,1000,10000}, rho in {-.8,0,.6,.9,.99}, sigma_M/sigma_R in {.1,.5,1,2,10}, plus boundary cells. Record bias with MC standard error, RMSE with bootstrap interval, quantiles, convergence rate and boundary mass. Predeclare valid interior regime separately; target >=99% finite converged fits in ordinary interior cells, with exceptions investigated. Do not require rho recovery under sigma_M=0 or decomposition recovery at rho=1. Consistency is not proved by a decreasing empirical MSE.

Uncertainty: parametric-bootstrap confidence intervals, at least 500 resamples/fit on a prespecified subset; assess nominal 95% coverage with 99% binomial intervals over independent replications. Hessian intervals are diagnostic only near boundaries.

Reproduction: retain a legacy reference-objective mode and exact-model mode. Original paper tables are legacy-method comparisons; corrected-method estimates are new results. Match design/initialization before comparing numbers. Published point estimates without MC standard errors cannot support exact-equality acceptance.
