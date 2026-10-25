"""Explicit state serialization and source-keyed calibration loading."""

from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, cast
import numpy as np
from .model import FilterState, Parameters, Mode, MODEL_VERSION


def estimator_fingerprint() -> str:
    """Conservative byte-level fingerprint of the estimator's numerical sources."""
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
        b"".join((folder / name).read_bytes() for name in files)
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
    if report.get("source_hash") != estimator_fingerprint() or not report.get(
        "full_design"
    ):
        raise ValueError("Scientific report is stale or only a pilot")
    return report
