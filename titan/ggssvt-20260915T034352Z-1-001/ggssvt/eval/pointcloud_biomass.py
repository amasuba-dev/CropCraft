"""Biomass probes for metric, pot-excluded point clouds.

The probe is intentionally small and auditable: it voxelises each aligned
point cloud above the estimated pot rim, then evaluates allometric and ridge
regressors with leave-one-plant-out validation.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ..config import VOXEL_SIZE_M, WORK_DIR
from ..data.preprocess import load_cached, usable_plant_ids


def _features(path: Path, pot_height_m: float) -> np.ndarray:
    values = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip() == "end_header":
                break
        for line in stream:
            fields = line.split()
            if len(fields) >= 3:
                values.append([float(fields[0]), float(fields[1]), float(fields[2])])
    points = np.asarray(values, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    above = points[points[:, 2] > pot_height_m] if points.size else points
    if above.size == 0:
        return np.zeros(5, dtype=np.float64)
    keys = np.floor(above / VOXEL_SIZE_M).astype(np.int64)
    occupied = np.unique(keys, axis=0).shape[0]
    z = above[:, 2]
    radial = np.linalg.norm(above[:, :2], axis=1)
    height = float(z.max() - pot_height_m)
    footprint = float(np.unique(keys[:, :2], axis=0).shape[0]) * VOXEL_SIZE_M**2
    spread = float(radial.max())
    volume = occupied * VOXEL_SIZE_M**3
    return np.array([volume, height, footprint, spread, volume / max(footprint * height, 1e-9)])


def _read_points(path: Path) -> np.ndarray:
    values = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip() == "end_header":
                break
        for line in stream:
            fields = line.split()
            if len(fields) >= 3:
                values.append([float(fields[0]), float(fields[1]), float(fields[2])])
    points = np.asarray(values, dtype=np.float64).reshape(-1, 3)
    return points[np.isfinite(points).all(axis=1)]


def _write_filtered(source: Path, target: Path, pot_height_m: float) -> int:
    """Write the metric cloud with pot/stand points removed."""
    points = _read_points(source)
    points = points[points[:, 2] > pot_height_m] if points.size else points
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    lines.extend(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in points)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return int(len(points))


def _metrics(predicted: np.ndarray, target: np.ndarray) -> dict:
    error = predicted - target
    return {
        "n": int(target.size),
        "rmse_kg": float(np.sqrt(np.mean(error**2))),
        "mae_kg": float(np.mean(np.abs(error))),
        "mare": float(np.mean(np.abs(error) / np.maximum(np.abs(target), 1e-9))),
        "bias_kg": float(np.mean(error)),
        "r2": float(1.0 - np.sum(error**2) / max(np.sum((target - target.mean()) ** 2), 1e-12)),
    }


def _loocv_ridge(x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    predictions = np.zeros_like(y)
    for holdout in range(len(y)):
        train = np.arange(len(y)) != holdout
        mean = x[train].mean(axis=0)
        scale = x[train].std(axis=0)
        scale[scale < 1e-9] = 1.0
        z = (x[train] - mean) / scale
        design = np.column_stack([np.ones(z.shape[0]), z])
        weights = np.linalg.solve(
            design.T @ design + alpha * np.diag([0.0] + [1.0] * z.shape[1]),
            design.T @ y[train],
        )
        sample = np.r_[1.0, (x[holdout] - mean) / scale]
        predictions[holdout] = sample @ weights
    return predictions


def run(
    *,
    pointcloud_root: Path,
    methods: tuple[str, ...] = ("dust3r", "mast3r", "fast3r"),
    cache_dir: Path = WORK_DIR / "cache",
    out: Path = WORK_DIR / "reports" / "pointcloud_biomass.json",
    filtered_root: Path | None = None,
) -> dict:
    """Evaluate every method's above-ground cloud with plant-level LOOCV."""
    targets = {}
    for plant_id in usable_plant_ids(cache_dir):
        cached = load_cached(plant_id, cache_dir)
        if np.isfinite(cached.target_kg):
            targets[plant_id] = (float(cached.target_kg), float(cached.pot_height_m))

    report = {"note": "Point clouds are aligned to the rig frame and filtered at each estimated pot rim.", "methods": {}}
    for method in methods:
        rows = []
        for plant_id, (target, rim) in sorted(targets.items()):
            path = pointcloud_root / method / f"{plant_id}.ply"
            if path.exists():
                filtered_path = None
                if filtered_root is not None:
                    filtered_path = filtered_root / method / f"{plant_id}.ply"
                    _write_filtered(path, filtered_path, rim)
                rows.append((plant_id, target, _features(path, rim)))
        if len(rows) < 4:
            report["methods"][method] = {"n": len(rows), "status": "insufficient point clouds"}
            continue
        ids = [row[0] for row in rows]
        y = np.array([row[1] for row in rows])
        x = np.stack([row[2] for row in rows])
        volume = np.maximum(x[:, 0], 1e-9)
        log_pred = np.zeros_like(y)
        for holdout in range(len(y)):
            train = np.arange(len(y)) != holdout
            slope, intercept = np.polyfit(np.log(volume[train]), np.log(np.maximum(y[train], 1e-9)), 1)
            log_pred[holdout] = np.exp(intercept + slope * np.log(volume[holdout]))
        ridge_pred = _loocv_ridge(x, y)
        report["methods"][method] = {
            "n": len(rows),
            "plant_ids": ids,
            "allometric_volume": _metrics(log_pred, y),
            "ridge_geometry": _metrics(ridge_pred, y),
            "features": {plant_id: values.tolist() for plant_id, _, values in rows},
        }
        if filtered_root is not None:
            report["methods"][method]["filtered_pointcloud_dir"] = str(
                filtered_root / method
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


__all__ = ["run"]
