"""Simulation adapter retaining the historical two-vector fit_mle contract."""

from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "PCI Framework"))
from PCIPipeline import Computation


class partial_cointegration(Computation):
    def fit_mle(self, W, tol=1e-10):
        alternative = super().fit_mle(W, tol)
        diff = np.diff(np.asarray(W, dtype=float))
        null = np.array([0.0, 0.0, np.sqrt(np.mean(diff**2))])
        return [alternative, null]
