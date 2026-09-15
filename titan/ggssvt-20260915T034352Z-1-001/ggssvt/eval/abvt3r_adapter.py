"""Run ABVT3R's classical stages on a CropCraft specimen.

The external checkout is imported read-only; CropCraft capture files are never
copied, renamed, or edited.  This adapter deliberately does not import the
broken ``ABVT3R/classes/predict_rf.py`` entrypoint.  Its model artifact is a
pickle containing ``__main__.DecisionTreeRegressor`` and cannot be loaded
portably, so biomass is reported as geometry features plus an explicit blocker.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ABVT3R = Path("/home/titan/aaron/ABVT3R")
if str(ABVT3R) not in sys.path:
    sys.path.insert(0, str(ABVT3R))

from procedure_alpha.preprocessing import PreProcessing  # noqa: E402
from procedure_alpha.registration import Registration  # noqa: E402
from procedure_alpha.reconstruction import ThreeDReconstruction  # noqa: E402

from ..config import KINECT_V2  # noqa: E402
from ..data.dataset import load_specimen  # noqa: E402
from .abvt3r_rf import evaluate_artifact  # noqa: E402


def _write_ply(path: Path, points: np.ndarray) -> None:
    path.write_text(
        "ply\nformat ascii 1.0\n"
        f"element vertex {len(points)}\nproperty float x\nproperty float y\n"
        "property float z\nend_header\n"
        + "".join(f"{x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in points),
        encoding="utf-8",
    )


def _write_mask(path: Path, pixels: np.ndarray, shape: tuple[int, int]) -> str:
    """Write a binary (0/255) mask, returning the relative format used."""
    mask = np.zeros(shape, dtype=np.uint8)
    if len(pixels):
        mask[pixels[:, 1].astype(int), pixels[:, 0].astype(int)] = 255
    try:
        from PIL import Image
    except ImportError:
        np.save(path.with_suffix(".npy"), mask)
        return "npy (Pillow unavailable; rows/columns match source depth PNG)"
    Image.fromarray(mask, mode="L").save(path)
    return "PNG grayscale (0 background, 255 retained filtered pixel)"


def _write_gallery(path: Path, plant_id: str, views: list[dict], outputs: list[str]) -> None:
    rows = [
        "<!doctype html><meta charset='utf-8'><title>ABVT3R pilot</title>",
        f"<h1>{plant_id} ABVT3R classical pilot</h1>",
        "<p>Filtered point clouds are camera-frame PLYs; masks are binary retained-pixel masks.</p>",
        "<h2>Global outputs</h2><ul>",
    ]
    for output in outputs:
        rows.append(f"<li><a href='{output}'>{output}</a></li>")
    rows.append("</ul><h2>Per-view segmentation</h2>")
    for view in views:
        rows.append(
            f"<h3>{view['view']} ({view['azimuth_deg']}°)</h3>"
            f"<p>{view['points']} filtered points; <a href='{view['ply']}'>PLY</a> · "
            f"<a href='{view['mask']}'>mask</a></p>"
            f"<img src='{view['mask']}' alt='{view['view']} mask' "
            "style='image-rendering:pixelated;max-width:512px'>"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _safe_sequential_icp(reg: Registration, arranged: list[np.ndarray], angles: list[int]) -> tuple[list[np.ndarray], list[dict]]:
    """Keep a plant running when ABVT3R ICP finds no correspondences."""
    fine = [arranged[0].copy()]
    stats: list[dict] = []
    for i in range(1, len(arranged)):
        source, target = arranged[i], np.vstack(fine)
        initial = float(np.linalg.norm(reg.calculate_centroid(source) - reg.calculate_centroid(target)))
        try:
            result, registered, _ = reg.icp_registration(
                source, target, max_iterations=50, tolerance=1e-5, max_corr_dist=0.05
            )
            fine.append(registered)
            stats.append({"view": f"{angles[i]}°", "initial_error_mm": initial * 1000,
                          "final_rmse_mm": float(result["rmse"]) * 1000,
                          "iterations": int(result["iterations"]), "converged": bool(result["converged"])})
        except (UnboundLocalError, ValueError, RuntimeError) as exc:
            fine.append(source.copy())
            stats.append({"view": f"{angles[i]}°", "initial_error_mm": initial * 1000,
                          "final_rmse_mm": None, "iterations": 0, "converged": False,
                          "status": "fallback_unregistered", "error": f"{type(exc).__name__}: {exc}"})
    return fine, stats


def write_batch_index(out_dir: Path) -> Path:
    reports = sorted(out_dir.rglob("*_report.json"))
    rows = ["<!doctype html><meta charset='utf-8'><title>ABVT3R batch</title>",
            "<h1>ABVT3R CropCraft batch</h1><table border='1'><tr><th>Plant</th><th>Report</th><th>Gallery</th><th>Status</th></tr>"]
    for report_path in reports:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        plant = html.escape(str(payload.get("plant_id", report_path.stem)))
        status = html.escape(str(payload.get("biomass", {}).get("status", "ok")))
        gallery = report_path.with_name(report_path.name.replace("_report.json", "_gallery.html"))
        report_link = html.escape(str(report_path.relative_to(out_dir)))
        gallery_link = html.escape(str(gallery.relative_to(out_dir)))
        link = f"<a href='{gallery_link}'>gallery</a>" if gallery.exists() else "—"
        rows.append(f"<tr><td>{plant}</td><td><a href='{report_link}'>report</a></td><td>{link}</td><td>{status}</td></tr>")
    index = out_dir / "index.html"
    index.write_text("\n".join(rows) + "</table>\n", encoding="utf-8")
    return index


def run(plant_id: str = "M001", *, max_points: int = 2500, out_dir: Path | None = None) -> dict:
    out_dir = out_dir or ROOT / "work_dirs" / "ggssvt" / "abvt3r"
    out_dir.mkdir(parents=True, exist_ok=True)
    specimen = load_specimen(plant_id)
    pre = PreProcessing()
    clouds: list[np.ndarray] = []
    angles: list[int] = []
    counts: list[dict] = []
    view_outputs: list[dict] = []

    for view in specimen.views:
        depth = view.load_depth() / 0.001  # ABVT3R expects uint16-like mm values.
        points, pixels = pre.point_cloud_generation(
            depth, KINECT_V2.fx, KINECT_V2.fy, KINECT_V2.cx, KINECT_V2.cy
        )
        # ABVT3R's pass-through stage is the classical segmentation/ROI stage.
        points, pixels = pre.pass_through_filter(
            points, pixels, -0.9, 0.9, -0.9, 0.9, 0.1, 2.5
        )
        if len(points) > max_points:
            rng = np.random.default_rng(0)
            keep = rng.choice(len(points), max_points, replace=False)
            points, pixels = points[keep], pixels[keep]
        if len(points) >= 20:
            points, pixels = pre.statistical_outlier_removal(
                points, pixels, m=min(20, len(points) - 1), k=2.0
            )
        ply_path = out_dir / f"{plant_id}_{view.position_id}_filtered.ply"
        mask_path = out_dir / f"{plant_id}_{view.position_id}_mask.png"
        _write_ply(ply_path, points)
        mask_format = _write_mask(mask_path, pixels, depth.shape)
        clouds.append(points.astype(np.float64))
        angles.append(view.azimuth_deg)
        entry = {
            "view": view.position_id,
            "points": int(len(points)),
            "azimuth_deg": view.azimuth_deg,
            "ply": ply_path.name,
            "mask": mask_path.name if mask_path.exists() else mask_path.with_suffix(".npy").name,
            "mask_format": mask_format,
        }
        counts.append(entry)
        view_outputs.append(entry)

    reg = Registration()
    arranged, _ = reg.arrange_views_in_circle(clouds, np.radians(angles))
    fine, reg_stats = _safe_sequential_icp(reg, arranged, angles)
    recon = ThreeDReconstruction(verbose=False)
    result = recon.complete_reconstruction_pipeline(
        fine, method="grid_based", voxel_size=0.02, smooth_iterations=0
    )
    merged = np.asarray(result["merged_cloud"])
    final = np.asarray(result["final_vertices"])
    triangles = np.asarray(result["final_triangles"])
    _write_ply(out_dir / f"{plant_id}_merged.ply", merged)
    _write_ply(out_dir / f"{plant_id}_mesh_vertices.ply", final)
    np.save(out_dir / f"{plant_id}_triangles.npy", triangles)
    np.save(out_dir / f"{plant_id}_surface_normals.npy", result["surface_normals"])
    global_outputs = [
        f"{plant_id}_merged.ply",
        f"{plant_id}_mesh_vertices.ply",
        f"{plant_id}_triangles.npy",
        f"{plant_id}_surface_normals.npy",
    ]
    _write_gallery(out_dir / f"{plant_id}_gallery.html", plant_id, view_outputs, global_outputs)

    z = final[:, 2] if final.size else np.zeros(0)
    above = final[z > 0.28] if final.size else final
    extent = (above.max(axis=0) - above.min(axis=0)) if len(above) else np.zeros(3)
    occupied = np.unique(np.floor(above / 0.007), axis=0).shape[0] if len(above) else 0
    features = {
        "occupied_voxels_7mm": int(occupied),
        "volume_m3_7mm": float(occupied * 0.007**3),
        "height_m_above_280mm": float(extent[2]) if len(above) else 0.0,
        "width_m": float(extent[0]) if len(above) else 0.0,
        "depth_m": float(extent[1]) if len(above) else 0.0,
    }
    report = {
        "plant_id": plant_id,
        "source": str(specimen.root),
        "stages": ["preprocessing", "registration", "reconstruction", "segmentation", "biomass_features"],
        "view_counts": counts,
        "registration": reg_stats,
        "reconstruction": result["reconstruction_stats"],
        "biomass_features": features,
        "biomass": {
            **evaluate_artifact(ABVT3R / "RF_model" / "biomass_rf_model.npy"),
            "target_net_weight_g": specimen.ground_truth.net_weight_g if specimen.ground_truth else None,
        },
        "outputs": [str(p) for p in sorted(out_dir.glob(f"{plant_id}_*"))],
    }
    (out_dir / f"{plant_id}_report.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    write_batch_index(out_dir)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plant", default="M001")
    parser.add_argument("--max-points", type=int, default=2500)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--batch", nargs="+", help="run several plant ids")
    args = parser.parse_args()
    for plant in args.batch or [args.plant]:
        print(json.dumps(run(plant, max_points=args.max_points, out_dir=args.out_dir),
                         indent=2, default=float))
    write_batch_index(args.out_dir or ROOT / "work_dirs" / "ggssvt" / "abvt3r")
