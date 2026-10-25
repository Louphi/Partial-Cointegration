"""Explicit state serialization and source-keyed calibration loading."""

from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, cast
import numpy as np
from .model import FilterState, Parameters, Mode, MODEL_VERSION, FitError
from .inference import Calibration


def estimator_fingerprint() -> str:
    """Numerical-source SHA with CRLF normalized to LF for portable checkouts."""
    folder = Path(__file__).parent
    files = [
        "model.py",
        "data.py",
        "likelihood.py",
        "estimation.py",
        "simulation.py",
        "inference.py",
    ]
    return hashlib.sha256(
        b"".join((folder / name).read_bytes().replace(b"\r\n", b"\n") for name in files)
    ).hexdigest()


def state_to_dict(state: FilterState) -> dict[str, Any]:
    """Serialize a model-bound posterior state without executable pickle data."""
    return {
        "mean": state.mean.tolist(),
        "covariance": state.covariance.tolist(),
        "parameters": {
            key: float(value) for key, value in asdict(state.parameters).items()
        },
        "mode": state.mode,
        "timestamp": state.timestamp,
        "version": state.version,
    }


def package_fingerprint() -> str:
    """Identify all package sources, including data, signal and ledger behavior."""
    return hashlib.sha256(
        b"".join(
            file.name.encode() + b"\0" + file.read_bytes().replace(b"\r\n", b"\n")
            for file in sorted(Path(__file__).parent.glob("*.py"))
        )
    ).hexdigest()


def state_from_dict(value: dict[str, Any]) -> FilterState:
    """Read an explicit state; filtering revalidates its shapes/PSD/timestamp."""
    if value.get("version") != MODEL_VERSION or value.get("mode") not in (
        "exact",
        "legacy",
    ):
        raise ValueError("Unsupported saved-state model/version")
    return FilterState(
        np.asarray(value["mean"], dtype=float),
        np.asarray(value["covariance"], dtype=float),
        Parameters(**value["parameters"]),
        cast(Mode, value["mode"]),
        value.get("timestamp"),
    )


def load_current_report(path: str | Path) -> dict[str, Any]:
    """Load complete source-matched evidence; inspect statistical acceptance gates separately."""
    report: dict[str, Any] = json.loads(Path(path).read_text())
    if estimator_fingerprint() not in (
        report.get("source_hash"),
        report.get("canonical_source_hash"),
    ) or not report.get("full_design"):
        raise ValueError("Scientific report is stale or only a pilot")
    return report


def calibration_from_report(path: str | Path, *, n: int) -> Calibration:
    """Load a source-matched, size-checked calibration with its full DGP identity.

    This supports the reported sample sizes and null design only. It does not
    validate a new market-data design merely because its length is the same.
    """
    report = load_current_report(path)
    entries = report["size"].get(str(n))
    if not entries:
        raise ValueError("No calibrated evidence for that sample size")
    if not all(cell["size_gate_passed"] for cell in entries.values()):
        raise FitError("Calibration size gate failed; inspect the scientific report")
    folder = Path(path).resolve().parent
    bank_path = (folder / report["null_score_bank_file"]).resolve()
    if bank_path.parent != folder:
        raise ValueError("Score bank must be a sibling of its report")
    if (
        hashlib.sha256(bank_path.read_bytes()).hexdigest()
        != report["null_score_bank_sha256"]
    ):
        raise ValueError("Calibration score-bank integrity check failed")
    with np.load(bank_path) as bank:
        if str(bank["source_hash"]) != report["source_hash"]:
            raise ValueError("Score bank and report source identities differ")
        scores = np.asarray(bank[f"calibration_n{n}"], dtype=float).copy()
    if scores.ndim != 1 or len(scores) != report["calibration"][str(n)]["replications"]:
        raise ValueError("Score bank has an unexpected replication count")
    scores.setflags(write=False)
    config = report["config"]
    return Calibration(
        scores,
        n,
        cast(Mode, config["mode"]),
        config["seed"],
        json.dumps(report["dgp"], sort_keys=True),
        {**config["fit_options"], "source_hash": estimator_fingerprint()},
    )
