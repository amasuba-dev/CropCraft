"""Convert DeepVoxels metric depth renders into pot-excluded biomass features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .pointcloud_biomass import _features
from ..config import WORK_DIR
from ..data.preprocess import load_cached


def depth_to_ply(
    depth: np.ndarray,
    pose_path: Path,
    intrinsics_path: Path,
    output: Path,
    pot_height_m: float,
) -> None:
    if depth.ndim == 4:
        depth = depth[:, :, :, 0]
    pose = np.loadtxt(pose_path)
    focal, cx, cy = np.loadtxt(intrinsics_path, max_rows=1)
    height, width = depth.shape
    scale = height / 424.0
    focal *= scale
    cx *= scale
    cy *= scale
    yy, xx = np.indices(depth.shape, dtype=np.float64)
    z = depth.astype(np.float64)
    valid = np.isfinite(z) & (z > 0)
    camera = np.stack(((xx - cx) * z / focal, (yy - cy) * z / focal, z), axis=-1)
    world = camera @ pose[:3, :3].T + pose[:3, 3]
    points = world[valid]
    points = points[points[:, 2] > pot_height_m]
    output.parent.mkdir(parents=True, exist_ok=True)
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
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    *,
    plant_id: str,
    render_dir: Path,
    plant_dir: Path,
    cache_dir: Path = WORK_DIR / "cache",
    output_dir: Path = WORK_DIR / "deepvoxels" / "ply_above_ground",
) -> dict:
    cached = load_cached(plant_id, cache_dir)
    depth = render_dir / "depth_metric.npy"
    metadata = json.loads((plant_dir / "cropcraft_metadata.json").read_text())
    pose_paths = sorted((plant_dir / "pose").glob("*.txt"))
    depth_stack = np.load(depth)
    clouds = []
    for index, pose_path in enumerate(pose_paths[: len(depth_stack)]):
        cloud = output_dir / plant_id / f"{index:06d}.ply"
        depth_to_ply(
            depth_stack[index],
            pose_path,
            plant_dir / "intrinsics.txt",
            cloud,
            cached.pot_height_m,
        )
        clouds.append(cloud)
    features = np.stack([_features(path, cached.pot_height_m) for path in clouds])
    feature = np.nanmean(features, axis=0)
    result = {
        "plant_id": plant_id,
        "views": len(clouds),
        "pot_height_m": float(cached.pot_height_m),
        "features_above_rim": feature.tolist(),
        "grid_barycenter_m": metadata["grid_barycenter_m"],
        "cloud_dir": str(output_dir / plant_id),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plant-id", required=True)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--plant-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=WORK_DIR / "cache")
    parser.add_argument("--output-dir", type=Path, default=WORK_DIR / "deepvoxels" / "ply_above_ground")
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
