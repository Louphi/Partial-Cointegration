"""Audit published critical values and independent rejection counts from score banks."""

import hashlib
import json
from pathlib import Path
import numpy as np
from partial_cointegration.io import load_current_report

root = Path(__file__).resolve().parents[1]
folder = root / "docs/results"
checked = []
for mode, name in [
    ("exact", "scientific-validation.json"),
    ("legacy", "legacy-score-tables.json"),
]:
    path = folder / name
    if mode == "exact":
        report = load_current_report(path)
    else:
        report = json.loads(path.read_text())
    bank_path = folder / report["null_score_bank_file"]
    assert (
        hashlib.sha256(bank_path.read_bytes()).hexdigest()
        == report["null_score_bank_sha256"]
    )
    with np.load(bank_path) as bank:
        assert str(bank["source_hash"]) == report["source_hash"]
        for n in [100, 1000, 10000]:
            calibration = bank[f"calibration_n{n}"]
            evaluation = bank[f"evaluation_n{n}"]
            assert len(calibration) == len(evaluation) == 10000
            assert np.isfinite(calibration).all() and np.isfinite(evaluation).all()
            assert not np.array_equal(calibration, evaluation)
            for alpha in [0.01, 0.05, 0.1] if mode == "exact" else [0.05]:
                entry = (
                    report["size"][str(n)][str(alpha)]
                    if mode == "exact"
                    else report["null_size"][str(n)]
                )
                cutoff = float(np.quantile(calibration, alpha))
                assert cutoff == entry["cutoff"]
                assert (
                    int(np.sum(evaluation < cutoff)) == entry["rejections"]["rejected"]
                )
                checked.append(
                    {"mode": mode, "n": n, "alpha": alpha, "status": "passed"}
                )
(folder / "null-bank-verification.json").write_text(
    json.dumps(checked, indent=2) + "\n"
)
print(f"Verified {len(checked)} critical values and independent rejection counts")
