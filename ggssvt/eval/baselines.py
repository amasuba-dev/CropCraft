"""Baselines GG-SSVT has to beat.

Three of them, chosen to isolate what the method claims:

``VolumeAllometric``
    Mass regressed on carved above-ground volume alone, in log-log space --
    the classical allometric form. If GG-SSVT cannot beat this, the learned
    parts are contributing nothing over the geometry that produced them.

``GeometricFeatures``
    Ridge regression on hand-crafted shape descriptors from the same carved
    volume: volume, height, spread, compactness and so on. This is the
    reconstruct-then-regress pipeline used in the predecessor work on this rig,
    and it is the honest comparator for "does a learned volumetric
    representation beat hand-designed features on the same reconstruction?".

``Direct2D``
    Mass from image statistics only -- silhouette area, depth spread, apparent
    height -- with no 3D reconstruction at all. This is the comparator research
    question 3 needs: does reconstructing geometry actually improve biomass
    estimation over predicting straight from pixels?

All three are fitted with the same leave-one-out protocol as the model, so the
numbers are directly comparable. Every model here is closed-form, so LOOCV over
thirty specimens costs milliseconds and there is no excuse for reporting a
baseline fitted on the full set.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import POT_HEIGHT_M, WORK_DIR, voxel_grid_centres
from ..data.preprocess import CachedSpecimen, load_cached
from .metrics import RegressionMetrics, regression_metrics


def _ridge_fit(
    features: np.ndarray, targets: np.ndarray, alpha: float
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Fit ridge regression on standardised features.

    Returns:
        ``(weights, intercept, mean, scale)`` so the caller can apply the same
        standardisation at prediction time.
    """
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-9] = 1.0
    standardised = (features - mean) / scale

    n_features = standardised.shape[1]
    gram = standardised.T @ standardised + alpha * np.eye(n_features)
    centred = targets - targets.mean()
    weights = np.linalg.solve(gram, standardised.T @ centred)

    return weights, float(targets.mean()), mean, scale


@dataclass
class SpecimenFeatures:
    """Everything the baselines need from one cached specimen."""

    plant_id: str
    target_kg: float
    above_ground_volume_m3: float
    total_volume_m3: float
    height_m: float
    mean_spread_m: float
    max_spread_m: float
    compactness: float
    silhouette_area_m2: float
    mean_subject_pixels: float
    mean_subject_depth_m: float
    subject_pixel_height: float

    def geometric_vector(self) -> np.ndarray:
        """Descriptors derived from the 3D reconstruction."""
        return np.array(
            [
                self.above_ground_volume_m3,
                self.above_ground_volume_m3 ** (2.0 / 3.0),
                self.height_m,
                self.mean_spread_m,
                self.max_spread_m,
                self.compactness,
                self.silhouette_area_m2,
            ],
            dtype=np.float64,
        )

    def image_vector(self) -> np.ndarray:
        """Descriptors available without any 3D reconstruction."""
        return np.array(
            [
                self.mean_subject_pixels,
                self.mean_subject_pixels ** 1.5,
                self.mean_subject_depth_m,
                self.subject_pixel_height,
                self.mean_subject_pixels * self.mean_subject_depth_m ** 2,
            ],
            dtype=np.float64,
        )


def extract_features(cached: CachedSpecimen) -> SpecimenFeatures:
    """Compute baseline features for one preprocessed specimen."""
    centres = voxel_grid_centres()
    occupancy = cached.occupancy
    voxel_volume = cached.voxel_size_m ** 3

    above = occupancy & (centres[..., 2] > POT_HEIGHT_M)
    above_points = centres[above]

    if above_points.size:
        height = float(above_points[:, 2].max())
        radial = np.linalg.norm(above_points[:, :2], axis=1)
        mean_spread = float(radial.mean())
        max_spread = float(radial.max())
        above_volume = float(above.sum()) * voxel_volume
        # Occupied fraction of the cylinder the canopy sweeps out.
        envelope = np.pi * max(max_spread, 1e-3) ** 2 * max(height - POT_HEIGHT_M, 1e-3)
        compactness = above_volume / envelope
        # Projected area of the canopy onto the floor.
        footprint = above.any(axis=2).sum() * cached.voxel_size_m ** 2
    else:
        height = mean_spread = max_spread = above_volume = compactness = 0.0
        footprint = 0.0

    subject_pixels = cached.mask.sum(axis=(1, 2)).astype(np.float64)
    depths = [
        cached.depth_m[i][cached.mask[i]] for i in range(cached.n_views)
    ]
    valid_depths = np.concatenate([d[d > 0] for d in depths]) if depths else np.zeros(1)

    rows = np.arange(cached.mask.shape[1])
    pixel_heights = []
    for index in range(cached.n_views):
        occupied_rows = rows[cached.mask[index].any(axis=1)]
        pixel_heights.append(
            float(occupied_rows.max() - occupied_rows.min()) if occupied_rows.size else 0.0
        )

    return SpecimenFeatures(
        plant_id=cached.plant_id,
        target_kg=cached.target_kg,
        above_ground_volume_m3=above_volume,
        total_volume_m3=float(occupancy.sum()) * voxel_volume,
        height_m=height,
        mean_spread_m=mean_spread,
        max_spread_m=max_spread,
        compactness=compactness,
        silhouette_area_m2=float(footprint),
        mean_subject_pixels=float(subject_pixels.mean()),
        mean_subject_depth_m=float(valid_depths.mean()) if valid_depths.size else 0.0,
        subject_pixel_height=float(np.mean(pixel_heights)) if pixel_heights else 0.0,
    )


def load_features(
    plant_ids: list[str], cache_dir: Path = WORK_DIR / "cache"
) -> list[SpecimenFeatures]:
    """Extract baseline features for many specimens."""
    return [extract_features(load_cached(pid, cache_dir)) for pid in plant_ids]


class Baseline(ABC):
    """A closed-form biomass regressor."""

    name: str = "baseline"

    @abstractmethod
    def fit(self, features: list[SpecimenFeatures]) -> "Baseline":
        ...

    @abstractmethod
    def predict(self, features: list[SpecimenFeatures]) -> np.ndarray:
        ...


class VolumeAllometric(Baseline):
    """Classical allometry: ``log m = a log V + b``.

    The one-parameter form every biomass paper starts from. Fitted in log space
    because plant mass and volume are related by a power law, not a line.
    """

    name = "volume allometric"

    def __init__(self, eps: float = 1e-9):
        self.eps = eps
        self.slope = 1.0
        self.intercept = 0.0

    def fit(self, features: list[SpecimenFeatures]) -> "VolumeAllometric":
        volume = np.array([max(f.above_ground_volume_m3, self.eps) for f in features])
        mass = np.array([max(f.target_kg, self.eps) for f in features])
        design = np.stack([np.log(volume), np.ones_like(volume)], axis=1)
        (self.slope, self.intercept), *_ = np.linalg.lstsq(design, np.log(mass), rcond=None)
        return self

    def predict(self, features: list[SpecimenFeatures]) -> np.ndarray:
        volume = np.array([max(f.above_ground_volume_m3, self.eps) for f in features])
        return np.exp(self.slope * np.log(volume) + self.intercept)


class _RidgeBaseline(Baseline):
    """Shared ridge machinery for the feature-vector baselines."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.weights: np.ndarray | None = None
        self.intercept = 0.0
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    @staticmethod
    def _vector(feature: SpecimenFeatures) -> np.ndarray:
        raise NotImplementedError

    def fit(self, features: list[SpecimenFeatures]) -> "_RidgeBaseline":
        design = np.stack([self._vector(f) for f in features])
        targets = np.array([f.target_kg for f in features])
        self.weights, self.intercept, self.mean, self.scale = _ridge_fit(
            design, targets, self.alpha
        )
        return self

    def predict(self, features: list[SpecimenFeatures]) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError(f"{self.name} was not fitted")
        design = np.stack([self._vector(f) for f in features])
        standardised = (design - self.mean) / self.scale
        return standardised @ self.weights + self.intercept


class GeometricFeatures(_RidgeBaseline):
    """Ridge on hand-crafted descriptors of the carved reconstruction."""

    name = "geometric features"

    @staticmethod
    def _vector(feature: SpecimenFeatures) -> np.ndarray:
        return feature.geometric_vector()


class Direct2D(_RidgeBaseline):
    """Ridge on image statistics only, with no 3D reconstruction."""

    name = "direct 2D"

    @staticmethod
    def _vector(feature: SpecimenFeatures) -> np.ndarray:
        return feature.image_vector()


class MeanPredictor(Baseline):
    """Predicts the training mean. The floor any real method must clear."""

    name = "mean"

    def __init__(self):
        self.value = 0.0

    def fit(self, features: list[SpecimenFeatures]) -> "MeanPredictor":
        self.value = float(np.mean([f.target_kg for f in features]))
        return self

    def predict(self, features: list[SpecimenFeatures]) -> np.ndarray:
        return np.full(len(features), self.value)


def loocv_baseline(
    baseline_factory, features: list[SpecimenFeatures]
) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-out predictions for a baseline.

    Args:
        baseline_factory: a zero-argument callable returning a fresh baseline.

    Returns:
        ``(predicted_kg, target_kg)``, aligned with ``features``.
    """
    predictions = np.zeros(len(features))
    targets = np.array([f.target_kg for f in features])

    for index in range(len(features)):
        train = [f for position, f in enumerate(features) if position != index]
        model = baseline_factory().fit(train)
        predictions[index] = float(model.predict([features[index]])[0])

    return predictions, targets


def evaluate_baselines(
    features: list[SpecimenFeatures],
) -> dict[str, tuple[RegressionMetrics, np.ndarray]]:
    """Run every baseline under LOOCV.

    Returns:
        Per baseline name: its metrics and its leave-one-out predictions.
    """
    factories = {
        MeanPredictor.name: MeanPredictor,
        VolumeAllometric.name: VolumeAllometric,
        GeometricFeatures.name: GeometricFeatures,
        Direct2D.name: Direct2D,
    }

    results: dict[str, tuple[RegressionMetrics, np.ndarray]] = {}
    for name, factory in factories.items():
        predicted, target = loocv_baseline(factory, features)
        results[name] = (regression_metrics(predicted, target), predicted)
    return results


__all__ = [
    "Baseline",
    "Direct2D",
    "GeometricFeatures",
    "MeanPredictor",
    "SpecimenFeatures",
    "VolumeAllometric",
    "evaluate_baselines",
    "extract_features",
    "load_features",
    "loocv_baseline",
]
