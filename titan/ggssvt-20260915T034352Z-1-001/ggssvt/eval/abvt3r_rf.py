"""CropCraft-side evaluator for the legacy ABVT3R RF artifact."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def evaluate_artifact(path: Path) -> dict:
    """Attempt a safe load and return an actionable portability diagnostic."""
    try:
        np.load(path, allow_pickle=True)
    except Exception as exc:  # noqa: BLE001 - preserve loader diagnostics
        return {
            "status": "blocked",
            "artifact": str(path),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "next_step": "Retrain/export with portable sklearn/joblib, or provide the original estimator module.",
        }
    return {"status": "loaded", "artifact": str(path)}


__all__ = ["evaluate_artifact"]
