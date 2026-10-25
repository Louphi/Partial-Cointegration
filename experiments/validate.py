"""Checkpointed scientific validation; synthetic data are labeled as such.

Run `python experiments/validate.py --suite complete --workers 4`.
The complete design runs 75 recovery cells x 1000 replications, independent
10000+10000 RW calibration/evaluation draws at each n, and a 100 x 500
bootstrap coverage subset. No market-performance claims are generated.
"""

from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from scipy.stats import norm

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from partial_cointegration import Parameters, fit_pair, FitOptions, FitError
from partial_cointegration.simulation import simulate_pair, summary
from partial_cointegration.inference import rejection_summary, wilson_interval


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


def worker(job):
    suite, cell, n, rho, sm, sr, start, count, root_seed, mode, cache = job
    path = Path(cache) / f"{suite}-{cell}-{start}.npz"
    if checkpoint_valid(path, (count, 6)):
        return suite, cell, start, str(path)
    records = np.full((count, 6), np.nan)
    failures = []
    for j in range(count):
        idx = start + j
        seed_parts = np.random.SeedSequence(
            [root_seed, suite, cell, idx]
        ).generate_state(2)
        seed, fitseed = map(int, seed_parts)
        pair = simulate_pair(n, Parameters(1, rho, sm, sr), seed=seed)
        try:
            fitted = fit_pair(
                pair.x, pair.y, mode=mode, options=FitOptions(seed=fitseed)
            )
            p = fitted.parameters
            records[j] = [
                p.beta,
                p.rho,
                p.sigma_m,
                p.sigma_r,
                fitted.log_ratio,
                sum(not d.success for d in fitted.diagnostics),
            ]
        except (FitError, ValueError, FloatingPointError) as e:
            failures.append({"replication": idx, "error": str(e)})
    save_checkpoint(path, records=records, failures=json.dumps(failures))
    return suite, cell, start, str(path)


def coverage_worker(job):
    idx, n, root_seed, resamples, cache = job
    path = Path(cache) / f"coverage-{idx}.npz"
    if path.exists():
        return str(path)
    truth = np.array([1, 0.6, 1, 1])
    outerseed = int(np.random.SeedSequence([root_seed, 40, idx]).generate_state(1)[0])
    pair = simulate_pair(n, Parameters(*truth), seed=outerseed)
    try:
        fit = fit_pair(pair.x, pair.y, options=FitOptions(seed=outerseed))
    except (FitError, ValueError):
        save_checkpoint(
            path,
            covered=np.full(4, np.nan),
            intervals=np.full((2, 4), np.nan),
            failed=resamples,
        )
        return str(path)
    draws = np.full((resamples, 4), np.nan)
    for k in range(resamples):
        seed = int(np.random.SeedSequence([root_seed, 41, idx, k]).generate_state(1)[0])
        generated = simulate_pair(n, fit.parameters, seed=seed)
        try:
            f = fit_pair(generated.x, generated.y, options=FitOptions(seed=seed))
            p = f.parameters
            draws[k] = [p.beta, p.rho, p.sigma_m, p.sigma_r]
        except (FitError, ValueError):
            pass
    valid = np.isfinite(draws).all(axis=1)
    if valid.all():
        intervals = np.quantile(draws, [0.025, 0.975], axis=0)
        covered = ((truth >= intervals[0]) & (truth <= intervals[1])).astype(float)
    else:
        intervals = np.full((2, 4), np.nan)
        covered = np.full(4, np.nan)
    save_checkpoint(
        path, covered=covered, intervals=intervals, failed=int((~valid).sum())
    )
    return str(path)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", choices=["pilot", "complete"], default="pilot")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=20261025)
    p.add_argument("--output", default="docs/results/scientific-validation.json")
    args = p.parse_args()
    full = args.suite == "complete"
    reps = 1000 if full else 20
    nullreps = 10000 if full else 100
    nvalues = [100, 1000, 10000] if full else [100, 1000]
    recovery = [
        (n, rho, ratio, 1.0)
        for n in nvalues
        for rho in [-0.8, 0.0, 0.6, 0.9, 0.99]
        for ratio in [0.1, 0.5, 1.0, 2.0, 10.0]
    ]
    boundary = [(n, 0.6, 0.0, 1.0) for n in nvalues] + [
        (n, 0.6, 1.0, 0.0) for n in nvalues
    ]
    config = {
        "suite": args.suite,
        "seed": args.seed,
        "recovery_reps": reps,
        "null_reps": nullreps,
        "n": nvalues,
        "rho": [-0.8, 0, 0.6, 0.9, 0.99],
        "ratio": [0.1, 0.5, 1, 2, 10],
        "mode": "exact",
        "fit_options": asdict(FitOptions()),
        "coverage_outer": 100 if full else 3,
        "coverage_bootstrap": 500 if full else 20,
    }
    files = [
        "model.py",
        "data.py",
        "likelihood.py",
        "estimation.py",
        "simulation.py",
        "inference.py",
    ]
    source_hash = hashlib.sha256(
        b"".join((ROOT / "src/partial_cointegration" / f).read_bytes() for f in files)
    ).hexdigest()
    key = hashlib.sha256(
        (json.dumps(config, sort_keys=True) + source_hash).encode()
    ).hexdigest()[:16]
    cache = ROOT / ".test-deps/studies" / key
    cache.mkdir(parents=True, exist_ok=True)
    jobs = []
    groups = {}
    for suite, cells, num in [
        (1, recovery, reps),
        (2, [(n, 0.0, 0.0, 1.0) for n in nvalues], nullreps),
        (3, [(n, 0.0, 0.0, 1.0) for n in nvalues], nullreps),
        (4, boundary, reps),
    ]:
        for cell, (n, rho, sm, sr) in enumerate(cells):
            groups[(suite, cell)] = []
            for start in range(0, num, 100):
                jobs.append(
                    (
                        suite,
                        cell,
                        n,
                        rho,
                        sm,
                        sr,
                        start,
                        min(100, num - start),
                        args.seed,
                        "exact",
                        str(cache),
                    )
                )
    begun = time.perf_counter()
    done = 0
    print(
        f"{len(jobs)} checkpoint batches; source {source_hash}; cache {key}", flush=True
    )
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, j) for j in jobs]
        for future in as_completed(futures):
            suite, cell, start, path = future.result()
            groups[(suite, cell)].append((start, path))
            done += 1
            if done % 50 == 0 or done == len(jobs):
                print(
                    f"Batches {done}/{len(jobs)}; elapsed {time.perf_counter()-begun:.1f}s",
                    flush=True,
                )
    arrays = {
        k: np.concatenate([np.load(path)["records"] for _, path in sorted(v)])
        for k, v in groups.items()
    }
    calibration = {}
    size = {}
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, 900]))
    for cell, n in enumerate(nvalues):
        cal = arrays[(2, cell)][:, 4]
        ev = arrays[(3, cell)][:, 4]
        valid = cal[np.isfinite(cal)]
        entries = {}
        for alpha in [0.01, 0.05, 0.1]:
            cutoff = float(np.quantile(valid, alpha))
            rep = rejection_summary(ev, cutoff)
            interval = wilson_interval(
                rep["rejected"],
                len(ev),
                z=float(norm.ppf(1 - 0.01 / (2 * 3 * len(nvalues)))),
            )
            # Quantile-calibration uncertainty is propagated into the rejection
            # rate on an independent evaluation sample, not added in score units.
            bootcuts = [
                np.quantile(rng.choice(valid, len(valid), replace=True), alpha)
                for _ in range(200)
            ]
            rates = np.array([np.mean(ev[np.isfinite(ev)] < c) for c in bootcuts])
            adjustment = float(
                np.max(
                    np.abs(np.quantile(rates, [0.005, 0.995]) - rep["conditional_rate"])
                )
            )
            gate = (
                interval[0] - adjustment <= alpha <= interval[1] + adjustment
                and np.isfinite(cal).all()
                and np.isfinite(ev).all()
            )
            entries[str(alpha)] = {
                "cutoff": cutoff,
                "rejections": rep,
                "wilson_99": interval,
                "calibration_rate_uncertainty": adjustment,
                "size_gate_passed": gate,
            }
        calibration[str(n)] = {
            "failed": int((~np.isfinite(cal)).sum()),
            "quantiles": {a: r["cutoff"] for a, r in entries.items()},
            "replications": len(cal),
        }
        size[str(n)] = entries
    cells = []
    for i, (n, rho, sm, sr) in enumerate(recovery):
        a = arrays[(1, i)]
        result = summary(a[:, :4], np.array([1, rho, sm, sr]))
        cutoff = calibration[str(n)]["quantiles"]["0.05"]
        valid_values = a[np.isfinite(a[:, :4]).all(axis=1), :4]
        if len(valid_values):
            boot_rng = np.random.default_rng(
                np.random.SeedSequence([args.seed, 901, i])
            )
            indices = boot_rng.integers(
                0, len(valid_values), size=(200, len(valid_values))
            )
            truth = np.array([1, rho, sm, sr])
            rmse_draws = np.sqrt(np.mean((valid_values[indices] - truth) ** 2, axis=1))
            result["rmse_bootstrap_99"] = np.quantile(
                rmse_draws, [0.005, 0.995], axis=0
            ).tolist()
            result["boundary_mass"] = {
                "sigma_m_zero": float(np.mean(valid_values[:, 2] == 0)),
                "sigma_r_zero": float(np.mean(valid_values[:, 3] == 0)),
                "rho_at_limit": float(
                    np.mean(np.abs(valid_values[:, 1]) >= 0.9999 - 1e-8)
                ),
            }
        result.update(
            n=n,
            rho=rho,
            sigma_m=sm,
            sigma_r=sr,
            power=rejection_summary(a[:, 4], cutoff),
            successful_fit_gate=result["failed"] / reps <= 0.01,
        )
        power = result["power"]
        result["power_wilson_99_conditional"] = (
            wilson_interval(power["rejected"], power["successful"])
            if power["successful"]
            else None
        )
        cells.append(result)
    boundaries = [
        dict(
            n=n,
            rho=rho,
            sigma_m=sm,
            sigma_r=sr,
            **summary(arrays[(4, i)][:, :4], np.array([1, rho, sm, sr])),
        )
        for i, (n, rho, sm, sr) in enumerate(boundary)
    ]
    for item in boundaries:
        item["rho_identified"] = item["sigma_m"] > 0
        if not item["rho_identified"]:
            for metric in ["bias", "mse", "rmse", "bias_mcse"]:
                if item.get(metric) is not None:
                    item[metric][1] = None
    print("Starting bootstrap coverage subset", flush=True)
    coverage_jobs = [
        (i, 1000, args.seed, config["coverage_bootstrap"], str(cache))
        for i in range(config["coverage_outer"])
    ]
    coverage = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(coverage_worker, j) for j in coverage_jobs]
        for i, future in enumerate(as_completed(futures)):
            path = future.result()
            coverage.append(np.load(path)["covered"])
            if (i + 1) % 10 == 0:
                print(f"Coverage {i+1}/{len(coverage_jobs)}", flush=True)
    coverage = np.array(coverage)
    coverage_report = []
    for j, param in enumerate(["beta", "rho", "sigma_m", "sigma_r"]):
        valid = np.isfinite(coverage[:, j])
        count = int(np.nansum(coverage[:, j]))
        interval = (
            wilson_interval(count, int(valid.sum()))
            if valid.any()
            else (np.nan, np.nan)
        )
        coverage_report.append(
            dict(
                parameter=param,
                total=len(coverage),
                failed=int((~valid).sum()),
                coverage=count / int(valid.sum()) if valid.any() else np.nan,
                wilson_99=interval,
                nominal_in_interval=interval[0] <= 0.95 <= interval[1],
            )
        )
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except subprocess.CalledProcessError:
        commit = "unavailable"
    report = {
        "config": config,
        "source_hash": source_hash,
        "canonical_source_hash": hashlib.sha256(
            b"".join(
                (ROOT / "src/partial_cointegration" / f)
                .read_bytes()
                .replace(b"\r\n", b"\n")
                for f in files
            )
        ).hexdigest(),
        "hash_contract": "source_hash records execution bytes; canonical_source_hash normalizes CRLF to LF only",
        "commit_at_report": commit,
        "cache": str(cache.relative_to(ROOT)),
        "seconds": time.perf_counter() - begun,
        "runtime_scope": "this invocation; checkpoint reuse is possible; not total CPU time",
        "versions": {
            p: importlib.metadata.version(p)
            for p in ["numpy", "pandas", "scipy", "numba"]
        },
        "calibration": calibration,
        "size": size,
        "recovery": cells,
        "boundaries": boundaries,
        "coverage": coverage_report,
        "synthetic_only": True,
        "simultaneous_size_gate": "Bonferroni 99% Wilson intervals over 3 levels x all sample sizes, augmented by 200 independent calibration resamples",
        "report_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "full_design": full,
        "raw_manifest": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(cache.glob("*.npz"))
        },
    }
    out = ROOT / args.output
    out.parent.mkdir(exist_ok=True, parents=True)
    out.write_text(json.dumps(clean(report), indent=2, allow_nan=False) + "\n")
    print(f"Saved {out}; elapsed {time.perf_counter()-begun:.1f}s", flush=True)


if __name__ == "__main__":
    main()
