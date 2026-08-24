"""Shape descriptors read from a TSDF fusion rather than from a carve.

The carved features summarise a visual hull, which for these species is a canopy
envelope: 25 of 36 specimens imply a bulk density one to two orders of magnitude
below plant tissue. These summarise the fused field instead, and the point of
comparison is whether a reconstruction that is allowed to have holes in it says
anything more useful about mass.

Two of the descriptors have no counterpart on the carve side and are worth
naming. ``tsdf_coverage`` is the fraction of the working volume any camera
measured, so it records how much of the answer is observation rather than
absence. ``tsdf_surface_m2`` counts the zero-crossing band, which is a surface
area that follows the leaves rather than wrapping them, and is the quantity the
canopy-area hypothesis wanted and the hull could not supply.

The interior here is a band behind the observed surfaces, not a filled solid, so
its volume is not the plant's volume in any strict sense. Whether it is
nonetheless a better predictor is an empirical question, and the answer is what
this module exists to supply.
"""

from __future__ import annotations

import numpy as np

from ..config import voxel_grid_centres
from ..geometry.fusion import FUSION_RESOLUTION, FUSION_VOXEL_M, fuse_cached

FUSION_KEYS = (
    "tsdf_above_rim_m3",
    "tsdf_surface_m2",
    "tsdf_height_m",
    "tsdf_mean_spread_m",
    "tsdf_max_spread_m",
    "tsdf_coverage",
    "tsdf_observed_m3",
)


def fusion_features(
    cached,
    *,
    resolution: int = FUSION_RESOLUTION,
    voxel_size_m: float = FUSION_VOXEL_M,
) -> dict:
    """Fuse one specimen and summarise the field.

    Returns a plain dict so it can be cached to JSON; fusion costs about fifteen
    seconds a specimen and should not be repeated for every experiment.
    """
    result = fuse_cached(cached, resolution=resolution, voxel_size_m=voxel_size_m)
    centres = voxel_grid_centres(resolution, voxel_size_m)
    heights = centres[..., 2]

    voxel_volume = voxel_size_m ** 3
    interior = result.interior
    above = interior & (heights > cached.pot_height_m)

    # The zero crossing, as the band of observed voxels whose distance to the
    # surface is under half a voxel. Counting them and multiplying by the face
    # area is a coarse surface estimate, but it needs no meshing and it does not
    # depend on the field being closed, which it is not.
    surface = result.observed & (np.abs(result.tsdf) < (0.5 / 3.0))
    surface_area = float(surface.sum()) * voxel_size_m ** 2

    points = centres[above]
    if points.size:
        radial = np.linalg.norm(points[:, :2], axis=1)
        height = float(points[:, 2].max())
        mean_spread = float(radial.mean())
        max_spread = float(radial.max())
    else:
        height = mean_spread = max_spread = 0.0

    return {
        "tsdf_above_rim_m3": float(above.sum()) * voxel_volume,
        "tsdf_surface_m2": surface_area,
        "tsdf_height_m": height,
        "tsdf_mean_spread_m": mean_spread,
        "tsdf_max_spread_m": max_spread,
        "tsdf_coverage": result.coverage(),
        "tsdf_observed_m3": float(result.observed.sum()) * voxel_volume,
    }


def fusion_vector(features: dict) -> np.ndarray:
    """The descriptors as a feature vector, in a fixed order."""
    return np.array([features[k] for k in FUSION_KEYS], dtype=np.float64)


__all__ = ["FUSION_KEYS", "fusion_features", "fusion_vector"]
