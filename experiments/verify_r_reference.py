"""Execute pinned author-written R scoring functions; no author optimizer parity claim."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.request import urlopen
import numpy as np
import pandas as pd
from partial_cointegration import Parameters
from partial_cointegration.filtering import _gains, filter_residuals

p = argparse.ArgumentParser()
p.add_argument("--rscript", required=True)
args = p.parse_args()
root = Path(__file__).resolve().parents[1]
folder = root / ".test-deps/r-reference"
folder.mkdir(parents=True, exist_ok=True)
sources = {
    "fit_par.R": "https://raw.githubusercontent.com/cran/partialAR/520fba1babeb85507c509234fcbccd87b60f9dd9/R/fit.R",
    "lr_pci.R": "https://raw.githubusercontent.com/cran/partialCI/a46db513f4526c27b953196726d001b88e0a76ae/R/lr_pci.R",
}
digests = {}
for name, url in sources.items():
    payload = urlopen(url, timeout=30).read()
    (folder / name).write_bytes(payload)
    digests[name] = hashlib.sha256(payload).hexdigest()
rows = []
expected = {}
rng = np.random.default_rng(20261025)
for n in [5, 10, 30]:
    for rho in [-0.8, 0, 0.6, 0.99]:
        for sm, sr in [(1, 1), (0.2, 1), (1, 0), (0, 1), (10, 0.1)]:
            identity = len(expected)
            w = 2.4 + np.cumsum(rng.normal(size=n))
            parameters = Parameters(1, rho, sm, sr)
            ours = filter_residuals(w, parameters, mode="legacy")
            # The author function includes the first density. Pure-AR legacy
            # uses a different initial permanent level; request that level explicitly.
            first_density = 0.5 * np.log(2 * np.pi * (sm**2 + sr**2))
            if sr == 0:
                first_density += w[0] ** 2 / (2 * sm**2)
            expected[identity] = (
                -ours.log_likelihood + first_density,
                _gains(parameters),
            )
            rows.extend(
                {"id": identity, "rho": rho, "sm": sm, "sr": sr, "w": value}
                for value in w
            )
pd.DataFrame(rows).to_csv(folder / "fixtures.csv", index=False)
rsource = """args <- commandArgs(trailingOnly=TRUE)
setwd(args[1])
library(zoo)
source("fit_par.R")
source("lr_pci.R")
d <- read.csv("fixtures.csv")
out <- lapply(split(d, d$id), function(z) {
  rho <- z$rho[1]; sm <- z$sm[1]; sr <- z$sr[1]
  r0 <- if (sr == 0) 0 else z$w[1]
  score <- loglik.par.ss(z$w, rho, sm, sr, M0=0, R0=r0)
  gains <- kalman.gain.par(rho, sm, sr)
  data.frame(id=z$id[1], nll=score, km=gains[1], kr=gains[2])
})
write.csv(do.call(rbind, out), "scores.csv", row.names=FALSE)
writeLines(R.version.string, "runtime.txt")
writeLines(as.character(packageVersion("zoo")), "zoo-version.txt")
"""
(folder / "check.R").write_text(rsource)
subprocess.run(
    [args.rscript, "--vanilla", str(folder / "check.R"), str(folder)], check=True
)
actual = pd.read_csv(folder / "scores.csv")
max_score_error = 0.0
max_gain_error = 0.0
for row in actual.itertuples():
    score, gains = expected[row.id]
    np.testing.assert_allclose(row.nll, score, atol=1e-9, rtol=1e-10)
    np.testing.assert_allclose([row.km, row.kr], gains, atol=1e-12, rtol=1e-12)
    max_score_error = max(max_score_error, abs(row.nll - score))
    max_gain_error = max(
        max_gain_error, float(np.max(np.abs(np.array([row.km, row.kr]) - gains)))
    )
report = {
    "status": "passed",
    "fixtures": len(actual),
    "seed": 20261025,
    "runtime": (folder / "runtime.txt").read_text().strip(),
    "zoo_version": (folder / "zoo-version.txt").read_text().strip(),
    "source_urls": sources,
    "source_sha256": digests,
    "max_absolute_score_error": max_score_error,
    "max_absolute_gain_error": max_gain_error,
    "comparison": "Author steady-state scorer with explicit matching initial level; first density retained in R and added to conditional Python score",
    "author_optimizer_executed": False,
    "full_algorithm_parity": False,
    "exact_model_parity_claimed": False,
}
(root / "docs/results/r-reference-verification.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report, indent=2))
