"""Prepare a CropCraft specimen for the legacy DeepVoxels runner."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def _opencv_poses(positions: dict) -> list[tuple[str, np.ndarray]]:
    """Convert CropCraft's OpenGL camera-to-world matrices for DeepVoxels."""
    poses = []
    for name in sorted(positions):
        matrix = np.asarray(positions[name]["transform_matrix"], dtype=np.float64)
        matrix[:3, 1:3] *= -1.0
        poses.append((name, matrix))
    return poses


def _look_at_center(poses: list[tuple[str, np.ndarray]]) -> np.ndarray:
    """Estimate a stable canonical center from the camera optical rays."""
    system = []
    target = []
    for _, pose in poses:
        direction = pose[:3, 2]
        projector = np.eye(3) - np.outer(direction, direction)
        system.append(projector)
        target.append(projector @ pose[:3, 3])
    center, *_ = np.linalg.lstsq(np.concatenate(system), np.concatenate(target), rcond=None)
    return center


def prepare(
    plant_dir: Path,
    out_dir: Path,
    *,
    scale: float = 1.0,
    focal_scale: float = 0.5,
    view_indices: list[int] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir = out_dir / "rgb"
    pose_dir = out_dir / "pose"
    rgb_dir.mkdir(exist_ok=True)
    pose_dir.mkdir(exist_ok=True)

    positions = json.loads((plant_dir / "rig_positions.json").read_text())
    poses = _opencv_poses(positions)
    if view_indices is not None:
        poses = [pose for index, pose in enumerate(poses) if index in view_indices]
    for index, (name, matrix) in enumerate(poses):
        shutil.copy2(plant_dir / "images" / f"{name}.png", rgb_dir / f"{index:06d}.png")
        (pose_dir / f"{index:06d}.txt").write_text(
            "\n".join(" ".join(f"{value:.9f}" for value in row) for row in matrix) + "\n"
        )

    intrinsics = json.loads((plant_dir / "camA_intrinsics.json").read_text())
    width = int(intrinsics["w"])
    height = int(intrinsics["h"])
    crop = min(width, height)
    cx = float(intrinsics["cx"]) - (width - crop) / 2.0
    cy = float(intrinsics["cy"])
    focal = float(intrinsics["fl_x"]) * focal_scale
    center = _look_at_center(poses)
    (out_dir / "intrinsics.txt").write_text(
        f"{focal:.9f} {cx:.9f} {cy:.9f}\n"
        f"{center[0]:.9f} {center[1]:.9f} {center[2]:.9f}\n"
        "0.01\n"
        f"{scale}\n"
        f"{height} {width}\n"
    )
    (out_dir / "cropcraft_metadata.json").write_text(
        json.dumps(
            {
                "coordinate_conversion": "OpenGL camera-to-world to OpenCV camera-to-world",
                "grid_barycenter_m": center.tolist(),
                "grid_scale_m": scale,
                "focal_scale": focal_scale,
                "square_crop": True,
                "pot_exclusion": "downstream biomass evaluation uses specimen pot_height_m",
            },
            indent=2,
        )
        + "\n"
    )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plant-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--focal-scale", type=float, default=0.5)
    parser.add_argument(
        "--view-indices",
        type=int,
        nargs="+",
        help="Optional validated source-view indices; invalid frustum views are excluded.",
    )
    args = parser.parse_args()
    print(
        prepare(
            args.plant_dir,
            args.out_dir,
            scale=args.scale,
            focal_scale=args.focal_scale,
            view_indices=args.view_indices,
        )
    )


if __name__ == "__main__":
    main()
