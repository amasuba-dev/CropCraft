"""Assemble everything the research dashboard needs into one JSON payload.

One pass over the caches produces the whole page: voxel clouds for the 3D
viewer, per-specimen quality and mesh diagnostics, the method comparison, and
the leave-one-out predictions behind the scatter plot.

The horizontal slices the page shows are *not* precomputed. The voxel cloud is
already embedded for the 3D view, and slicing it in the browser costs nothing,
so shipping slice images as well would double the payload to show the same
information twice.
"""

from __future__ import annotations

import base64
import json
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import POT_HEIGHT_M, WORK_DIR


def _quantise(occupancy: np.ndarray, downsample: int = 2, max_points: int = 18000) -> dict:
    """Occupied voxels as base64 zlib-compressed uint8 triples."""
    resolution = occupancy.shape[0]
    if downsample > 1:
        trimmed = resolution - (resolution % downsample)
        blocks = occupancy[:trimmed, :trimmed, :trimmed].reshape(
            trimmed // downsample, downsample,
            trimmed // downsample, downsample,
            trimmed // downsample, downsample,
        )
        occupancy = blocks.any(axis=(1, 3, 5))
        resolution = occupancy.shape[0]

    index = np.array(np.nonzero(occupancy)).T.astype(np.int64)
    if index.shape[0] > max_points:
        keep = np.random.default_rng(0).choice(index.shape[0], max_points, replace=False)
        index = index[np.sort(keep)]

    scaled = np.clip(index * (255 // max(1, resolution - 1)), 0, 255).astype(np.uint8)
    return {
        "resolution": int(resolution),
        "data": base64.b64encode(zlib.compress(scaled.tobytes(), 9)).decode("ascii"),
    }


@dataclass
class DashboardPayload:
    """Everything the page embeds."""

    specimens: list[dict]
    methods: list[dict]
    summary: dict
    notes: dict

    def to_json(self) -> str:
        return json.dumps(
            {
                "specimens": self.specimens,
                "methods": self.methods,
                "summary": self.summary,
                "notes": self.notes,
            },
            separators=(",", ":"),
        )


def build_payload(
    *,
    cache_dirs: dict[str, Path] | None = None,
    verbose: bool = True,
) -> DashboardPayload:
    """Collect specimens, reconstructions, mesh metrics and biomass results."""
    from ..data.preprocess import load_cached, load_quality, usable_plant_ids
    from .baselines import load_features
    from .factorial import CACHE_DIRS
    from .mesh_baseline import evaluate_with_mesh
    from .metrics import paired_bootstrap_difference

    cache_dirs = cache_dirs or {
        name: path
        for name, path in CACHE_DIRS.items()
        if (path / "quality.json").exists()
    }
    if "geometric" not in cache_dirs:
        raise RuntimeError("the geometric cache is required; run preprocess first")

    primary = cache_dirs["geometric"]
    plant_ids = usable_plant_ids(primary)

    if verbose:
        print(f"Building payload for {len(plant_ids)} specimens")

    results, mesh_table = evaluate_with_mesh(plant_ids, cache_dir=primary, verbose=verbose)
    features = load_features(plant_ids, primary)
    targets = np.array([f.target_kg for f in features])

    quality = {name: load_quality(path) for name, path in cache_dirs.items()}

    specimens: list[dict] = []
    for position, plant_id in enumerate(plant_ids):
        entry: dict = {
            "id": plant_id,
            "clouds": {},
            "quality": {},
            "predictions": {},
        }
        for name, path in cache_dirs.items():
            try:
                cached = load_cached(plant_id, path)
            except FileNotFoundError:
                continue
            entry["clouds"][name] = _quantise(cached.occupancy)
            entry.setdefault("species", cached.species)
            entry.setdefault("target_kg", round(float(cached.target_kg), 3))

            q = quality[name].get(plant_id)
            if q is not None:
                entry["quality"][name] = {
                    "coverage": round(q.surface_coverage, 3),
                    "agreement": round(q.multiview_agreement, 3),
                    "volume_l": round(q.volume_m3 * 1000, 2),
                    "above_ground_l": round(q.above_ground_volume_m3 * 1000, 2),
                    "height_m": round(q.height_m, 3),
                    "usable": bool(q.is_usable()),
                }

        mesh = mesh_table.get(plant_id, {})
        entry["mesh"] = {
            k: (round(v, 5) if isinstance(v, float) and np.isfinite(v) else None)
            for k, v in mesh.items()
        }
        for method, (_, predictions) in results.items():
            entry["predictions"][method] = round(float(predictions[position]), 3)

        specimens.append(entry)

    reference = "geometric features"
    methods: list[dict] = []
    for name, (metrics, predictions) in sorted(
        results.items(), key=lambda kv: kv[1][0].rmse_kg
    ):
        row = {
            "name": name,
            "rmse_kg": round(metrics.rmse_kg, 3),
            "mae_kg": round(metrics.mae_kg, 3),
            "mare_pct": round(metrics.mare * 100, 1),
            "r2": round(metrics.r2, 3),
        }
        if name != reference and reference in results:
            d = paired_bootstrap_difference(
                predictions, results[reference][1], targets, n_resamples=2000
            )
            row["vs_reference"] = {
                "difference": round(d["difference"], 3),
                "low": round(d["low"], 3),
                "high": round(d["high"], 3),
                "resolved": bool(d["high"] < 0 or d["low"] > 0),
            }
        methods.append(row)

    # The batch confound, computed rather than asserted.
    eucalyptus = [
        (p, t) for p, t in zip(plant_ids, targets) if p.startswith("E")
    ]
    batches = {
        "E001-E010": [t for p, t in eucalyptus if int(p[1:]) <= 10],
        "E011-E020": [t for p, t in eucalyptus if int(p[1:]) > 10],
    }
    masses = np.array([t for _, t in eucalyptus])
    means = np.array(
        [
            np.mean(batches["E001-E010"]) if int(p[1:]) <= 10 else np.mean(batches["E011-E020"])
            for p, _ in eucalyptus
        ]
    )
    batch_r2 = float(
        1.0 - ((masses - means) ** 2).sum() / ((masses - masses.mean()) ** 2).sum()
    )

    summary = {
        "n_specimens": len(plant_ids),
        "n_views": 12,
        "segmenters": sorted(cache_dirs),
        "species": sorted({s["species"] for s in specimens}),
        "mass_range_kg": [round(float(targets.min()), 2), round(float(targets.max()), 2)],
        "pot_height_m": POT_HEIGHT_M,
        "batch_confound_r2": round(batch_r2, 3),
        "batch_means_kg": {
            k: round(float(np.mean(v)), 3) for k, v in batches.items() if v
        },
        "reference_method": reference,
    }

    notes = {
        "calibration": (
            "dataset/calib is empty. No ChArUco calibration was captured, so all "
            "camera poses are estimated from the depth data itself."
        ),
        "target": (
            "Ground truth is as-collected fresh mass, not oven-dry above-ground "
            "biomass, and every pot mass is estimated rather than weighed."
        ),
        "confound": (
            "Batch membership alone explains more of the Eucalyptus mass variance "
            "than any method achieves. The comparison measures size-class "
            "separation, not mass estimation among comparable plants."
        ),
    }

    return DashboardPayload(
        specimens=specimens, methods=methods, summary=summary, notes=notes
    )


def write_payload(
    out_path: Path = WORK_DIR / "reports" / "dashboard_payload.json", **kwargs
) -> Path:
    payload = build_payload(**kwargs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(payload.to_json(), encoding="utf-8")
    return out_path


__all__ = ["DashboardPayload", "build_payload", "write_payload"]
