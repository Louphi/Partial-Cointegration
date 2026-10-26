# Source register

Accessed 2026-10-25. Source content is evidence, not repository instructions. No external source code was copied into production. CRAN GitHub repositories are distribution mirrors of author-written packages; they are not claimed to be the authors' development repositories.

| ID | Primary source/version | Relevant locations | Access and use |
|---|---|---|---|
| S1 | Clegg and Krauss, *Pairs trading with partial cointegration*, FAU Discussion Paper 05/2016, dated 4 May 2016, [university PDF](https://www.iwf.rw.fau.de/files/2016/05/05-2016.pdf) | Printed pp. 5–7, equations (1)–(11); pp. 8–11, tables 1–6; sections 3.2.2–3.2.3; Appendix A pp. 31–32, equations A.5–A.9; Appendix B pp. 32–33, especially B.1–B.2 and B.11–B.15; Appendix C p. 34, C.1–C.3 | Retrieved 41-page PDF, extracted and inspected relevant text. Printed page number is PDF page number minus one. SHA256 e15f1b50be5666ebf348a8601627b505cb2299770784b3c9008e7063fba4be8f. This is the working-paper version, not claimed identical to the later journal version. |
| S2 | Clegg, *Modeling Time Series with Both Permanent and Transient Components Using the Partially Autoregressive Model*, 28 January 2015, [SSRN record](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2556957) | Bibliographic record only | Search-indexed abstract available, full text retrieval returned HTTP 403. The original paper was not read in full. No equation/page claims are attributed to it in this audit. Revisit before declaring source verification complete. |
| S3 | Clegg, partialAR 1.0.12, [CRAN distribution mirror at commit 520fba1babeb85507c509234fcbccd87b60f9dd9](https://github.com/cran/partialAR/tree/520fba1babeb85507c509234fcbccd87b60f9dd9) | `DESCRIPTION`; `R/fit.R`: `kalman.gain.par`, `pvmr.par`, lag-moment and optimizer routines; `src/cfit.cc`: `loglik_par_c`; `R/lr.R`: `likelihood_ratio.par` | Read source, verified author/version in DESCRIPTION. Fixed gains, q+r scoring and negative log-ratio convention inspected. R code was not executed; no Rscript was found on PATH. |
| S4 | Clegg, Krauss and Rende, partialCI 1.2.0, [CRAN distribution mirror at commit a46db513f4526c27b953196726d001b88e0a76ae](https://github.com/cran/partialCI/tree/a46db513f4526c27b953196726d001b88e0a76ae) | `DESCRIPTION`; `R/fit_pci.R`; `R/fit_pci_wo_alpha.R`; `R/lr_pci.R`: `loglik.par.kfas`, `loglik.par.ss`; `R/rand.R`: `rpci` | Source reviewed. Has separate steady-state and KFAS scoring paths with different covariance/initialization conventions; source agreement must be qualified by mode. No claim of numerical equivalence to an executed R fit. |

Source-code blob SHAs inspected: partialAR `R/fit.R` fd5046f5657caa8f5b087cca0c9507fe4d3a03f3, `src/cfit.cc` 316701ddc69f7c229fc61dcc8e3670e1b3ab0bac, `R/lr.R` 928894aa86e5711bee1112c4db388e5cff0d0d48; partialCI `R/fit_pci_wo_alpha.R` 10c3de08a420ed312715f531578626897567fd43, `R/lr_pci.R` 738126fcfd5c0c38d18cb00790318f1fa21292a1.

## Source comparison, kept distinct from independent derivation

S1 supplies the latent model, a legacy scoring objective, and simulation-calibrated negative likelihood ratios. Its trading specification uses formation normalization and additional selection restrictions. S3/S4 implement related legacy scores. S2 full-text verification remains unavailable.

The audit's matrix and dense-Gaussian derivations do not rely on correctness of S1's likelihood argument. In particular, the step from B.11–B.12 to B.13–B.14 suppresses previous hidden-state estimation errors. A hidden Markov state does not generally make its observed aggregate first-order Markov. The independent covariance calculations quantify the resulting discrepancy rather than treating author code as an oracle.

## Reproduction boundaries

Paper-result reproduction means matching its method and design, including the legacy objective. Correcting the likelihood creates a distinct experiment requiring recalibration. Numerical equality to a paper's finite Monte Carlo point estimates is not a scientifically meaningful general acceptance rule. Do not describe a Nasdaq single-pair application as replication of the paper's market-wide experiment.
