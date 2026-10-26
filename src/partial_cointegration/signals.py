"""Causal filtering and explicitly named normalization/decision variants."""

from __future__ import annotations
from typing import Any, Literal
import numpy as np
import pandas as pd
from .filtering import filter_residuals
from .model import Parameters


def score_components(
    training: Any,
    testing: Any,
    parameters: Parameters,
    *,
    variant: Literal["formation", "rolling"] = "formation",
    window: int = 30,
) -> dict[str, Any]:
    """Continue fitted states into evaluation; normalization uses only past rows."""
    if window < 2 or variant not in ("formation", "rolling"):
        raise ValueError("Choose formation/rolling and a window >=2")
    if len(training) < 2 or len(testing) == 0 or training.index[-1] >= testing.index[0]:
        raise ValueError("Training must strictly precede nonempty evaluation")
    tr = filter_residuals(training.to_numpy(), parameters, timestamps=training.index)
    te = filter_residuals(
        testing.to_numpy(), parameters, state=tr.final_state, timestamps=testing.index
    )
    train_m = pd.Series(tr.m, index=training.index)
    test_m = pd.Series(te.m, index=testing.index)
    if variant == "formation":
        scale = float(train_m.std(ddof=1))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Formation component has no usable scale")
        ztrain = train_m / scale
        ztest = test_m / scale
    else:
        combined = pd.concat([train_m, test_m])
        roll = combined.rolling(window)
        z = (combined - roll.mean()) / roll.std(ddof=1).replace(0, np.nan)
        ztrain = z.iloc[: len(training)]
        ztest = z.iloc[len(training) :]
    # Predict-only state estimates exist during missing quotes, but are not
    # executable signals. Preserve rolling history while masking those decisions.
    ztest = ztest.where(testing.notna())
    ztrain = ztrain.where(training.notna())
    return {
        "training_m": train_m,
        "testing_m": test_m,
        "training_score": ztrain,
        "testing_score": ztest,
        "state": te.final_state,
    }


def decisions(scores: Any, entry: float = 1.0, exit: float = -0.5) -> tuple[Any, Any]:
    """Stateful long-x/short-y decisions; an invalid quote never creates a decision."""
    if not np.isfinite([entry, exit]).all() or entry <= exit:
        raise ValueError("Entry must exceed exit")
    frame = scores.to_frame() if isinstance(scores, pd.Series) else scores
    opens = pd.DataFrame(False, index=frame.index, columns=frame.columns)
    closes = opens.copy()
    for col in frame.columns:
        active = False
        for i, value in enumerate(frame[col]):
            if not np.isfinite(value):
                continue
            if not active and value > entry:
                opens.iloc[i, frame.columns.get_loc(col)] = True
                active = True
            elif active and value < exit:
                closes.iloc[i, frame.columns.get_loc(col)] = True
                active = False
    return opens, closes
