# Corrected model contract, version 0.2

The exact conditional specification is M1–M3 in the audit mathematical document. The implemented default fixes alpha=0, M0=0, R0=y0-beta*x0, P0=0 for **all** pure-RW, pure-AR and mixed candidates. The conditional density excludes the initial observation. Inputs in labeled Series must have identical unique increasing indices; NumPy arrays are explicitly positional. No log-price/share-hedge equivalence is assumed.

Noise standard deviations are nonnegative with positive total variance. Stationary transients require abs(rho)<1. Numerical estimation searches rho in [-.9999,.9999], an explicit configurable bound, and evaluates exact zero-noise boundaries. Rho is unidentified when sigma_M=0; at rho=1 the component split is unidentified, so that boundary is represented by a combined-variance random walk in comparison experiments. Alpha is a location convention, never an identified fitted parameter.

## Concentrated estimation

For fixed rho and fraction f=q/(q+r), the innovation filter is linear in the observed spread. Write innovation vectors as v_y-beta*v_x and innovation variance as tau_squared*F_base, with tau_squared=q+r. Let A=sum(v_x^2/F_base), B=sum(v_x*v_y/F_base). Conditional GLS gives beta_hat=B/A, and tau_squared_hat=sum((v_y-beta_hat*v_x)^2/F_base)/(n-1). Substitute these into M3 and minimize over rho,f. This is analytic profiling of the same joint conditional likelihood, not OLS beta held fixed. A second residual pass avoids cancellation in the profiled sum of squares.

Internal scale transformations add the appropriate (n-1)*log(y_scale) likelihood offset and convert beta/noise scales back to price units. Numerical profiling agrees with the full filter at the resulting parameters, and exact filtering agrees with independent dense Gaussian densities.

Pure-RW beta/scale are analytic. Pure-AR rho is searched using a coarse grid and bounded scalar refinements. Mixed candidates use deterministic and independently seeded starts; a derivative-free bounded fallback is used if all mixed line searches fail. Only successful finite candidates are accepted. Failed attempts remain in diagnostics. The best accepted candidate includes both nested boundaries; optimizer success does not prove a global optimum.

The first recovery pilot exposed two mixed line-search failures at a valid pure-AR boundary. The final solver accepts a separately converged fallback rather than relabeling failed output as success. Regression seeds preserve these cases. The study was restarted under a new source fingerprint; the preliminary failures were not silently removed from the same result bank.

## Legacy mode

`mode="legacy"` retains fixed gains, q+r scoring, and the inherited pure-AR initialization switch. Alpha remains explicitly normalized. It is a comparison objective, not the exact latent-model Gaussian likelihood. Profiling, boundary candidate handling and convergence checks differ from the original optimizer. Legacy score tables must disclose those differences and cannot be called certified author-algorithm reproductions. Old critical values never apply to exact mode.

## State and timing

FilterState contains the posterior mean/covariance, model parameters, mode, version and timestamp. Continuation checks model identity and ordering. JSON serialization is explicit; filtering validates saved-state shape/PSD. Missing residuals propagate prediction mean/covariance without a measurement update or density contribution. Missing quotes never produce a trading decision/fill. Rolling history may contain causal predictions; decisions themselves are masked when observations are absent.

A close-t decision can execute only at a later observed close. Units are set from decision-time prices; a later price jump can cause a margin rejection rather than a fictitious retroactive resize. All pair allocations share one committed capital pool; short proceeds do not create extra gross capacity. Costs and per-row borrow charges are explicit. Missing terminal quotes require official settlement. The configured last-period close is an administrative liquidation, not a signal computed from that same final price.

## Inference and scientific scope

Lambda=ell_null-ell_alternative is negative-convention; D=-2*Lambda is equivalent. Bottom-fraction ranking is not a calibrated significance test. Use independent null calibration/evaluation under matching n, x design, initialization, objective, options and source fingerprint. Nonregular boundaries invalidate an automatic Wilks chi-square assertion. Failure counts and unconditional rejection bounds are reported. Bootstrap coverage and Monte Carlo size gates are empirical checks, not proofs of asymptotic properties.

The committed compact null-score banks permit direct quantile/rejection-count audits. `calibration_from_report` verifies source identity, full-study status, sample-size gates, bank digest and replication count, then returns read-only scores with DGP/source metadata. The lower-level Calibration class assumes the caller supplies an honest complete design/options identity; use the guarded report loader for published evidence.

The market feed, security mapping and delisting/corporate-action inputs remain external. The study runner supports configured cohorts; `aggregate_cohorts` funds overlapping cohorts from one cash pool and reports a declared-lag asymptotic HAC interval when enough monthly observations exist. Historical validation of that workflow still requires qualified real data. The paper-inspired configuration is a distinct corrected reference variant. No result is a promise of profitable trading.
