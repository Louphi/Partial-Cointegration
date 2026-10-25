"""Full-size legacy-score comparison tables; never advertised as exact author parity.

Matches inherited geometric-x/legacy initialization DGP and 10k replications/cell.
For rho=1, uses its observationally equivalent combined-variance random walk;
component recovery in that row is explicitly unidentified.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from partial_cointegration import Parameters, FitOptions, fit_pair, FitError
from partial_cointegration.simulation import simulate_pair, summary
from partial_cointegration.inference import rejection_summary, wilson_interval
from scipy.stats import norm


def checkpoint_valid(path, shape):
    if not path.exists():
        return False
    from zipfile import BadZipFile

    try:
        with np.load(path) as data:
            if data["records"].shape == shape:
                return True
    except (OSError, ValueError, EOFError, BadZipFile):
        pass
    path.rename(path.with_suffix(".interrupted"))
    return False


def save_checkpoint(path, **data):
    temporary = path.with_suffix(".working.npz")
    np.savez_compressed(temporary, **data)
    temporary.replace(path)


def job_work(job):
    cell, n, rho, sm, start, count, root_seed, cache = job
    path = Path(cache) / f"{cell}-{start}.npz"
    if checkpoint_valid(path, (count, 5)):
        return cell, start, str(path)
    rows = np.full((count, 5), np.nan)
    failures = []
    for j in range(count):
        rep = start + j
        seed, fitseed = map(
            int, np.random.SeedSequence([root_seed, 77, cell, rep]).generate_state(2)
        )
        p = (
            Parameters(1, rho, sm, 1)
            if rho < 1
            else Parameters(1, 0, 0, np.sqrt(sm**2 + 1))
        )
        data = simulate_pair(n, p, seed=seed, initialization="legacy")
        try:
            fitted = fit_pair(
                data.x, data.y, mode="legacy", options=FitOptions(seed=fitseed)
            )
            p = fitted.parameters
            rows[j] = [p.beta, p.rho, p.sigma_m, p.sigma_r, fitted.log_ratio]
        except (FitError, ValueError) as e:
            failures.append({"replication": rep, "error": str(e)})
    save_checkpoint(path, records=rows, failures=json.dumps(failures))
    return cell, start, str(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replications", type=int, default=10000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=20261025)
    args = p.parse_args()
    settings = sorted(
        set(
            [
                (n, 0.9, float(sm))
                for n in [100, 1000, 10000]
                for sm in np.arange(0, 2.01, 0.25)
            ]
            + [
                (n, float(rho), 1.0)
                for n in [100, 1000, 10000]
                for rho in np.arange(0.6, 1.001, 0.05)
            ]
        )
    )
    # Normalize decimal grid endpoints to represent the intended boundary exactly.
    settings = [(n, round(rho, 2), round(sm, 2)) for n, rho, sm in settings]
    settings = sorted(set(settings))
    sources = ["model.py", "estimation.py", "likelihood.py", "simulation.py"]
    sha = hashlib.sha256(
        b"".join((ROOT / "src/partial_cointegration" / f).read_bytes() for f in sources)
    ).hexdigest()
    key = hashlib.sha256(
        (
            sha
            + str(args.replications)
            + repr(settings)
            + ("" if args.seed == 20261025 else str(args.seed))
        ).encode()
    ).hexdigest()[:16]
    cache = ROOT / ".test-deps/paper-study" / key
    cache.mkdir(parents=True, exist_ok=True)
    jobs = []
    groups = {i: [] for i in range(len(settings))}
    for i, (n, rho, sm) in enumerate(settings):
        for start in range(0, args.replications, 100):
            jobs.append(
                (
                    i,
                    n,
                    rho,
                    sm,
                    start,
                    min(100, args.replications - start),
                    args.seed,
                    str(cache),
                )
            )
    calibration_groups = {n: [] for n in [100, 1000, 10000]}
    for n in calibration_groups:
        cell = settings.index((n, 0.9, 0.0))
        for offset in range(0, args.replications, 100):
            jobs.append(
                (
                    cell,
                    n,
                    0.9,
                    0.0,
                    args.replications + offset,
                    min(100, args.replications - offset),
                    args.seed,
                    str(cache),
                )
            )
    started = time.perf_counter()
    print(
        f"{len(settings)} legacy-score cells, {len(jobs)} checkpoint batches",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, future in enumerate(
            as_completed([pool.submit(job_work, job) for job in jobs])
        ):
            cell, start, path = future.result()
            if start >= args.replications:
                calibration_groups[settings[cell][0]].append((start, path))
            else:
                groups[cell].append((start, path))
            if (i + 1) % 100 == 0:
                print(
                    f"Paper-score batches {i+1}/{len(jobs)}; {time.perf_counter()-started:.0f}s",
                    flush=True,
                )
    arrays = {
        cell: np.concatenate([np.load(path)["records"] for _, path in sorted(parts)])
        for cell, parts in groups.items()
    }
    calibration_arrays = {
        n: np.concatenate([np.load(path)["records"][:, 4] for _, path in sorted(parts)])
        for n, parts in calibration_groups.items()
    }
    null_path = ROOT / "docs/results/legacy-null-banks.npz"
    banks = {f"calibration_n{n}": values for n, values in calibration_arrays.items()}
    banks.update(
        {
            f"evaluation_n{n}": arrays[settings.index((n, 0.9, 0.0))][:, 4]
            for n in calibration_arrays
        }
    )
    np.savez_compressed(
        null_path,
        **banks,
        source_hash=sha,
        seed=args.seed,
        initialization="legacy",
        mode="legacy",
        replications=args.replications,
    )
    reports = []
    size_report = {}
    rng = np.random.default_rng(99177)
    for i, (n, rho, sm) in enumerate(settings):
        report = summary(arrays[i][:, :4], np.array([1, rho, sm, 1.0]))
        nullcell = next(
            k for k, setting in enumerate(settings) if setting == (n, 0.9, 0.0)
        )
        null = arrays[nullcell][:, 4]
        calibration = calibration_arrays[n]
        finite = calibration[np.isfinite(calibration)]
        cutoff = float(np.quantile(finite, 0.05))
        if n not in size_report:
            size = rejection_summary(null, cutoff)
            interval = wilson_interval(
                size["rejected"], len(null), z=float(norm.ppf(1 - 0.01 / (2 * 3)))
            )
            bootcuts = [
                np.quantile(rng.choice(finite, len(finite), replace=True), 0.05)
                for _ in range(200)
            ]
            rates = np.array(
                [np.mean(null[np.isfinite(null)] < value) for value in bootcuts]
            )
            uncertainty = float(
                np.max(
                    np.abs(
                        np.quantile(rates, [0.005, 0.995]) - size["conditional_rate"]
                    )
                )
            )
            size_report[n] = {
                "cutoff": cutoff,
                "rejections": size,
                "wilson_99_familywise": interval,
                "calibration_rate_uncertainty": uncertainty,
                "calibration_failures": int((~np.isfinite(calibration)).sum()),
                "passed": bool(
                    interval[0] - uncertainty <= 0.05 <= interval[1] + uncertainty
                    and np.isfinite(calibration).all()
                    and np.isfinite(null).all()
                ),
            }
        # A separate null bank is required for valid power calibration. These
        # descriptive quantiles are reported, not used to certify test size.
        report.update(
            n=n,
            rho=rho,
            sigma_m=sm,
            sigma_r=1.0,
            identified_components=rho < 1 and sm > 0,
            power_at_five_percent=rejection_summary(arrays[i][:, 4], cutoff),
            independent_null_quantiles={
                str(a): float(np.quantile(finite, a)) for a in [0.01, 0.05, 0.1]
            },
        )
        power = report["power_at_five_percent"]
        report["power_wilson_99_conditional"] = (
            wilson_interval(power["rejected"], power["successful"])
            if power["successful"]
            else None
        )
        valid_estimates = arrays[i][np.isfinite(arrays[i][:, :4]).all(axis=1), :4]
        report["mse_mcse"] = (
            (
                np.std(
                    (valid_estimates - np.array([1, rho, sm, 1])) ** 2, axis=0, ddof=1
                )
                / np.sqrt(len(valid_estimates))
            ).tolist()
            if len(valid_estimates) > 1
            else None
        )
        if rho == 1 or sm == 0:
            unidentified = [1, 2, 3] if rho == 1 else [1]
            for metric in ["bias", "mse", "rmse", "bias_mcse", "mse_mcse"]:
                if report.get(metric) is not None:
                    for index in unidentified:
                        report[metric][index] = None
        reports.append(report)
    out = {
        "mode": "legacy",
        "seed": args.seed,
        "report_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dgp_initialization": "legacy",
        "replications_per_cell": args.replications,
        "source_hash": sha,
        "canonical_source_hash": hashlib.sha256(
            b"".join(
                (ROOT / "src/partial_cointegration" / f)
                .read_bytes()
                .replace(b"\r\n", b"\n")
                for f in sources
            )
        ).hexdigest(),
        "synthetic_only": True,
        "author_algorithm_parity": False,
        "reason": "Profiled optimization, explicit boundary candidates and intercept normalization differ from the original optimizer; reference scoring modes also differ in initial observation treatment. These are full-size legacy-score comparison tables, not certified reproductions of the author tables. Independent calibration/evaluation is used for this comparison scorer; it does not certify the original author algorithm.",
        "seconds": time.perf_counter() - started,
        "runtime_scope": "this invocation; checkpoint reuse is possible; not total CPU time",
        "independent_calibration_reps_per_n": args.replications,
        "null_size": size_report,
        "null_score_bank_file": null_path.name,
        "null_score_bank_sha256": hashlib.sha256(null_path.read_bytes()).hexdigest(),
        "cells": reports,
        "raw_manifest": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in cache.glob("*.npz")
        },
    }
    (ROOT / "docs/results/legacy-score-tables.json").write_text(
        json.dumps(out, indent=2, allow_nan=False) + "\n"
    )
    print("Legacy-score tables saved", flush=True)


if __name__ == "__main__":
    main()
