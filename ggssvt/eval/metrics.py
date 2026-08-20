"""Evaluation metrics for reconstruction and for biomass estimation.

Reconstruction is reported with both a volumetric metric (voxel IoU) and a
surface metric (Chamfer distance and F-score), deliberately. They disagree on
branching plants, and which one you report changes the ranking of methods: a
surface F-score rewards getting the visible envelope right, while volumetric IoU
punishes a hull that is hollow or inflated where the cameras could not see. On a
thin sapling a method can score well on F-score while recovering a fraction of
the true volume, so reporting only F-score overstates volumetric accuracy.

Both are computed here so the gap between them can be measured rather than
assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegressionMetrics:
    """Standard biomass-estimation errors."""

    rmse_kg: float
    mae_kg: float
    mare: float          # mean absolute relative error, unitless
    r2: float
    bias_kg: float
    n: int

    def as_dict(self) -> dict[str, float]:
        return {
            "rmse_kg": self.rmse_kg,
            "mae_kg": self.mae_kg,
            "mare": self.mare,
            "r2": self.r2,
            "bias_kg": self.bias_kg,
            "n": float(self.n),
        }

    def __str__(self) -> str:  # pragma: no cover - reporting convenience
        return (
            f"RMSE {self.rmse_kg:.3f} kg  MAE {self.mae_kg:.3f} kg  "
            f"MARE {self.mare * 100:.1f}%  R2 {self.r2:.3f}  "
            f"bias {self.bias_kg:+.3f} kg  (n={self.n})"
        )


def regression_metrics(
    predicted_kg: np.ndarray, target_kg: np.ndarray
) -> RegressionMetrics:
    """RMSE, MAE, MARE and R-squared for a set of mass predictions.

    R-squared is the coefficient of determination against the target mean, so a
    model that always predicts the mean scores 0 and a worse-than-mean model
    scores negative. With thirty specimens that distinction matters: a model can
    look plausible on RMSE while carrying no information at all.
    """
    predicted = np.asarray(predicted_kg, dtype=np.float64).ravel()
    target = np.asarray(target_kg, dtype=np.float64).ravel()

    if predicted.shape != target.shape:
        raise ValueError(
            f"predicted {predicted.shape} and target {target.shape} must match"
        )
    if predicted.size == 0:
        raise ValueError("no predictions to score")

    error = predicted - target
    ss_residual = float((error ** 2).sum())
    ss_total = float(((target - target.mean()) ** 2).sum())

    return RegressionMetrics(
        rmse_kg=float(np.sqrt((error ** 2).mean())),
        mae_kg=float(np.abs(error).mean()),
        mare=float((np.abs(error) / np.maximum(target, 1e-6)).mean()),
        r2=float(1.0 - ss_residual / ss_total) if ss_total > 0 else float("nan"),
        bias_kg=float(error.mean()),
        n=int(predicted.size),
    )


def voxel_iou(predicted: np.ndarray, truth: np.ndarray) -> float:
    """Intersection over union of two boolean occupancy grids."""
    predicted = np.asarray(predicted, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    union = np.logical_or(predicted, truth).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(predicted, truth).sum() / union)


def _nearest_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Distance from every source point to its nearest target point.

    Chunked brute force. The point sets here are thousands of voxel centres, not
    millions, so a KD-tree would add a SciPy dependency for no practical gain.
    """
    if source.size == 0 or target.size == 0:
        return np.full(max(source.shape[0], 1), np.inf)

    out = np.empty(source.shape[0], dtype=np.float64)
    chunk = max(1, int(4e7 // max(1, target.shape[0])))
    for start in range(0, source.shape[0], chunk):
        block = source[start : start + chunk]
        distances = np.linalg.norm(block[:, None, :] - target[None, :, :], axis=-1)
        out[start : start + chunk] = distances.min(axis=1)
    return out


@dataclass(frozen=True)
class ReconstructionMetrics:
    """Surface and volumetric reconstruction quality."""

    chamfer_m: float
    precision: float
    recall: float
    f_score: float
    voxel_iou: float
    threshold_m: float

    def as_dict(self) -> dict[str, float]:
        return {
            "chamfer_m": self.chamfer_m,
            "precision": self.precision,
            "recall": self.recall,
            "f_score": self.f_score,
            "voxel_iou": self.voxel_iou,
            "threshold_m": self.threshold_m,
        }


def reconstruction_metrics(
    predicted_points: np.ndarray,
    truth_points: np.ndarray,
    *,
    predicted_grid: np.ndarray | None = None,
    truth_grid: np.ndarray | None = None,
    threshold_m: float = 0.02,
) -> ReconstructionMetrics:
    """Chamfer distance, F-score at a distance threshold, and voxel IoU.

    Args:
        predicted_points: ``(N, 3)`` reconstructed points.
        truth_points: ``(M, 3)`` reference points.
        predicted_grid, truth_grid: boolean grids for the IoU term. Omit to skip
            it (reported as NaN).
        threshold_m: F-score distance threshold, following the Tanks-and-Temples
            convention of a fixed metric tolerance.
    """
    predicted_points = np.asarray(predicted_points, dtype=np.float64).reshape(-1, 3)
    truth_points = np.asarray(truth_points, dtype=np.float64).reshape(-1, 3)

    forward = _nearest_distances(predicted_points, truth_points)
    backward = _nearest_distances(truth_points, predicted_points)

    chamfer = float(forward.mean() + backward.mean()) / 2.0
    precision = float((forward < threshold_m).mean())
    recall = float((backward < threshold_m).mean())
    f_score = (
        0.0
        if precision + recall == 0
        else 2.0 * precision * recall / (precision + recall)
    )

    iou = (
        voxel_iou(predicted_grid, truth_grid)
        if predicted_grid is not None and truth_grid is not None
        else float("nan")
    )

    return ReconstructionMetrics(
        chamfer_m=chamfer,
        precision=precision,
        recall=recall,
        f_score=f_score,
        voxel_iou=iou,
        threshold_m=threshold_m,
    )


def average_precision(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Threshold-free ranking quality of an occupancy field.

    Voxel IoU at a fixed 0.5 threshold is a brittle way to report reconstruction
    on this data: over 99% of the working volume is empty, so an under-confident
    field scores exactly 0.0 even when it ranks occupied voxels well above empty
    ones. That reads as total failure when the real problem is a miscalibrated
    operating point, and it hides genuine progress during training.

    Average precision has no threshold, so it separates "the model cannot tell
    occupied from empty" from "the model can, but its probabilities are shifted".
    Report it alongside IoU rather than instead of it -- IoU is what a downstream
    volume integral actually depends on.
    """
    scores = np.asarray(probabilities, dtype=np.float64).ravel()
    truth = np.asarray(labels, dtype=np.float64).ravel() > 0.5

    n_positive = int(truth.sum())
    if n_positive == 0:
        return float("nan")
    if n_positive == truth.size:
        return 1.0

    order = np.argsort(scores)[::-1]
    ranked = truth[order]

    true_positives = np.cumsum(ranked)
    precision = true_positives / np.arange(1, ranked.size + 1)

    # Sum precision at each rank where a positive is retrieved.
    return float(precision[ranked].sum() / n_positive)


def best_threshold_iou(
    probabilities: np.ndarray, labels: np.ndarray, n_steps: int = 50
) -> tuple[float, float]:
    """The highest voxel IoU achievable over any threshold, and that threshold.

    Read together with the IoU at 0.5, the gap between the two says whether a
    weak reconstruction score is a calibration problem or a real one.
    """
    scores = np.asarray(probabilities, dtype=np.float64).ravel()
    truth = np.asarray(labels, dtype=np.float64).ravel() > 0.5

    if truth.sum() == 0:
        return float("nan"), float("nan")

    best_iou, best_at = 0.0, 0.5
    for threshold in np.linspace(scores.min(), scores.max(), n_steps)[1:-1]:
        predicted = scores > threshold
        union = np.logical_or(predicted, truth).sum()
        if union == 0:
            continue
        score = float(np.logical_and(predicted, truth).sum() / union)
        if score > best_iou:
            best_iou, best_at = score, float(threshold)

    return best_iou, best_at


def bootstrap_interval(
    predicted_kg: np.ndarray,
    target_kg: np.ndarray,
    metric: str = "rmse_kg",
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for one regression metric.

    With thirty specimens a point estimate on its own is close to meaningless --
    a single badly reconstructed plant moves RMSE substantially. Report the
    interval alongside it.
    """
    predicted = np.asarray(predicted_kg, dtype=np.float64).ravel()
    target = np.asarray(target_kg, dtype=np.float64).ravel()
    rng = np.random.default_rng(seed)

    samples = []
    for _ in range(n_resamples):
        index = rng.integers(0, predicted.size, predicted.size)
        try:
            samples.append(
                regression_metrics(predicted[index], target[index]).as_dict()[metric]
            )
        except (ValueError, ZeroDivisionError):
            continue

    if not samples:
        return float("nan"), float("nan")

    tail = (1.0 - confidence) / 2.0
    values = np.array(samples)
    return float(np.quantile(values, tail)), float(np.quantile(values, 1.0 - tail))


__all__ = [
    "ReconstructionMetrics",
    "RegressionMetrics",
    "average_precision",
    "best_threshold_iou",
    "bootstrap_interval",
    "reconstruction_metrics",
    "regression_metrics",
    "voxel_iou",
]
