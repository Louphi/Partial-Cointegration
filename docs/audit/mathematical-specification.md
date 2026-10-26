# Mathematical specification and independent checks

Audit date: 2026-10-25. Code baseline: `1501aa6bb0b246ec30a8e3b305743cc58b986b5b`.
This document specifies a recommended corrected model; it does not change production behavior. Equations labeled M are independently derived here. Source/version details and access limitations are in [sources.md](sources.md).

## 1. Model and units

Let observed, aligned price levels be x_t and y_t, and w_t = y_t - alpha - beta x_t. Use

    s_t = (M_t, R_t)',    w_t = H s_t,
    s_t = T s_(t-1) + u_t,
    T = diag(rho, 1), H = [1, 1],
    Cov(u_t) = Q = diag(q, r), q = sigma_M^2, r = sigma_R^2.       (M1)

Innovations are independent across components and time and Gaussian for likelihood-based inference. The conditional residual model additionally assumes exogeneity of x with respect to u; verify this assumption empirically rather than inferring it from a good fit. There is no observation noise in M1.

In price levels, alpha, M, R and the sigmas have the units of y; beta converts units of x to units of y. A regression in log prices defines a different model, and beta then describes relative price changes rather than shares. Do not exponentiate signals and retain the same share hedge without a derivation. The inherited generator uses geometric x prices and additive w; that is a conditional regression design, not an additive Gaussian random walk for x levels.

Use |rho|<1 for a stationary transient component; q,r>=0 and q+r>0. Near-unit-root and boundary regimes require separate reporting. A zero-noise deterministic series is outside an ordinary nonsingular density and needs an explicit diagnostic, not division by zero. At rho=1, M and R are observationally indistinguishable random walks; at q=0, rho is unidentified. Negative rho gives alternating decay; the usual positive-rho half-life formula cannot be interpreted as monotone convergence.

## 2. Initial conditions and identification

Recommended initial corrected mode: condition on w_0, set a_0=(0,w_0)' and P_0=0, with alpha fixed at zero as a normalization. These are explicit model assumptions, not a stationary prior. The same initialization must be used for all interior and boundary candidates, including r=0; otherwise a boundary comparison changes two assumptions simultaneously. Offer an independently specified stationary/diffuse initialization later if the research objective requires it.

The inherited mixed model initializes Mhat_0=0 and Rhat_0=w_0. For any constant c, shifting w to w-c shifts every Rhat by -c and leaves Mhat and all errors unchanged. Therefore alpha is flat in this conditional likelihood whenever r>0. Optimizing alpha does not identify an equilibrium intercept; it selects an arbitrary location convention. `alpha_profile` in the diagnostic output demonstrates equality for alpha=-100,0,100 to floating-point precision. An estimated initial R level and a free alpha have the same location confounding.

The inherited code changes initialization to Mhat_0=w_0,Rhat_0=0 exactly when r=0. Consequently its pure-AR boundary also changes the meaning of the intercept. Handle this explicitly in legacy comparisons. Do not claim that the proposed alpha=0 normalization identifies an economically meaningful intercept.

Beta from no-intercept OLS of differences is a starting value. The actual fit subsequently optimizes beta jointly. Under M1, correctly specified exogenous x, and identified interior parameters, joint conditional likelihood is the recommended estimator. Do not make unconditional consistency claims for nonstationary or boundary settings.

## 3. Exact conditional filter and likelihood

Starting from stated a_0,P_0, for t=1,...,n-1:

    a_minus = T a_(t-1)
    P_minus = T P_(t-1) T' + Q
    v_t = w_t - H a_minus
    F_t = H P_minus H'
    K_t = P_minus H' / F_t
    a_t = a_minus + K_t v_t
    P_t = (I-K_t H) P_minus (I-K_t H)'                       (M2)

The last line is a Joseph-form update with zero observation variance. Require finite positive F_t; reject invalid parameter evaluations explicitly. Preserve symmetry and check positive semidefiniteness numerically.

    ell = -0.5 sum_t [ log(2 pi) + log(F_t) + v_t^2/F_t ].     (M3)

M2 and M3 account for state-estimation uncertainty. A fixed set of gains does not justify replacing F_t with process-noise variance.

Independent check: construct lower-triangular shock loading matrices B_M[i,j]=sigma_M rho^(i-j) and B_R[i,j]=sigma_R for j<=i, for observations after the initial value. Then C=B_M B_M'+B_R B_R'. Evaluate the multivariate Gaussian density by Cholesky factorization and compare it with M3. This construction does not reuse Kalman gains. The executable diagnostic agrees in six selected parameter cases within the predeclared tolerance. It checks both likelihood algebra and time indexing, not maximum-likelihood parameter recovery.

For the audit's pure-AR comparison only, both independent calculations match the inherited M_0=w_0,R_0=0 convention. This does not validate the recommended common-boundary initialization; that proposed implementation needs its own fixtures in T02.

## 4. Steady-state filter: what agrees and what does not

With an exact observation, posterior component errors sum to zero, so posterior covariance has the form p[[1,-1],[-1,1]]. Therefore

    F = q+r+(1-rho)^2 p,                                     (M4)
    p_new = ((rho^2 p+q)(p+r)-rho^2 p^2)/F.                  (M5)

M5 follows from the Schur complement of the predicted covariance. Solve this Riccati fixed point for p>=0 or iterate M2, then use the associated K and F. The production closed-form gains match this fixed point in the tested cases; see `matrix_vs_legacy_state_max_error` in [diagnostic-results.json](diagnostic-results.json).

However, production `calc_log_like` uses q+r with those gains. This omits the last term of M4, which is positive in an interior mixed model with rho!=1. At rho=.6,q=r=1, independently computed F is 2.19224993899441, while production uses 2. At rho=-.8,q=.49,r=1.44, F is 3.20485627015337, while production uses 1.93. The discrepancy persists at steady state; it is not solely a startup convention.

Hold mean initialization and fixed-gain residuals constant: the first case gives production ell=-11.5073474733752 versus variance-corrected steady-state ell=-11.5962977533971 on the diagnostic series. The exact known-initial-state likelihood is -11.4967121317828; that further difference is due to initialization and time-varying gains. These three objectives must not be conflated.

The author reference implementation also contains the q+r scoring convention. Thus agreement with that implementation alone would reproduce the legacy objective, not independently validate it as the latent-model Gaussian likelihood. Preserve an explicitly named legacy scoring mode for paper reproduction and make the exact conditional mode the recommended scientific default. Do not reuse legacy critical values for a changed objective.

Reference `loglik_par_c` scores all n supplied observations, whereas Python initializes from its first observation and scores n-1 errors. KFAS reference paths also specify their own initial covariance. Absolute reference scores cannot be compared until the initial observation, prior and conditioning conventions match; the omitted first score can depend on the noise parameters.

## 5. Lag moments and descriptive measures

For a stationary M independent of R, direct variance expansion gives

    V_k = Var(w_t-w_(t-k))
        = 2q(1-rho^k)/(1-rho^2) + k r.                      (M6)

Consequently the fraction of one-period differenced spread variance attributable to M is 2q/[2q+(1+rho)r]. It is not an OLS goodness-of-fit statistic and not a probability of trading profit. Derive lag-moment starting values from M6, but diagnose small/zero denominators and non-admissible sample estimates. Constant or deterministic-drift data need deliberate treatment. Finite samples using different lag windows and population/sample variance divisors introduce additional differences; document the exact convention.

## 6. Inference

Retain legacy lambda=ell_null-ell_alternative (more negative means stronger evidence); the equivalent positive statistic is D=-2 lambda. A negative lambda is not itself a sign bug. Candidate full-model bounds must include q=0 and r=0 with a common likelihood and initialization. Explicitly evaluate null and AR candidates and flag a full-model score worse than a nested candidate beyond tolerance.

Under q=0, rho disappears; at rho=1 the noise split also disappears. These are nonregular inference problems. Ordinary chi-square Wilks calibration is not established here. Use independent Monte Carlo/parametric-bootstrap calibration under the actual estimator, sample size, initialization, and x design. Quantify calibration uncertainty. Rejecting a random-walk null does not distinguish a pure AR process from a mixed process, nor establish profitability or correct specification.

Report estimation failures, weak identification and boundary mass separately. Do not compute rho-recovery coverage in a null regime where rho has no unique true observational value. A Hessian-based standard error is unreliable for a flat alpha direction or unidentified boundary parameters.

## 7. Trading specification recommended for implementation

Keep a strict distinction between research reproduction and a new rolling-score strategy. Use point-in-time membership at formation end; fit on the preceding formation window; retain missing trading observations without using their future existence to drop a training stock. Choose normalization and thresholds using training/validation data only.

Carry filter means and covariance into trading and never smooth using future observations. Under the proposed default, a signal after close t executes at the next available close t+1 because only close data are currently available; next-open execution requires an open-price feed. Allocate a common capital budget and define fees, slippage, borrow, end-of-period liquidation and delisting handling explicitly.

For exposure short one share of y, price-level spread delta implies beta shares of x in the opposite spread direction: holdings (beta,-1). Dollar amounts differ by price. Negative beta requires two short legs for that spread direction; rejecting nonpositive beta is a defensible initial trading restriction but not a mathematical estimation restriction. Report per-pair results separately from a shared-capital portfolio.

## Scope of verification

The independent algebra/numerics support the likelihood and identification findings. Six baseline software tests, fourteen baseline fits, and targeted edge-case diagnostics do not establish consistency, size/power, real-data robustness, or economic performance. Full validation is specified in [validation-protocol.md](validation-protocol.md) and the correction plan.
