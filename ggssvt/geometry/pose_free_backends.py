"""Adapters for the DUSt3R, MASt3R and Fast3R backends.

**These have not been executed against the real weights.** None of the three
installs without a GPU and the machine this was written on has none, so they are
written against each project's documented interface and verified only for shape
and convention handling against stubs. The maths they feed --
:func:`~ggssvt.geometry.pose_free.compare_poses` and
:func:`~ggssvt.geometry.pose_free.recover_scale_from_depth` -- *is* tested, so a
first run should be watched for import and API drift rather than for silently
wrong numbers.

Two conventions decide whether the output is meaningful, and both are handled
here rather than left to the caller:

**Camera convention.** All three return camera-to-world poses in the OpenCV
convention (+x right, +y down, +z forward), the same as
:mod:`ggssvt.geometry.rig`, so no flip is applied. This is asserted rather than
assumed -- :func:`sanity_check_result` verifies the cameras end up looking
roughly at the scene centroid, which catches a convention change in an upstream
release.

**Scale.** DUSt3R and Fast3R return geometry up to an unknown global scale;
MASt3R's ``_metric`` checkpoint does not. Feeding a scale-free reconstruction
into a volume calculation would measure the rescaling rather than the plant, so
the non-metric backends must go through
:func:`~ggssvt.geometry.pose_free.recover_scale_from_depth` first, and
``PoseFreeResult.is_metric`` records which happened.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .pose_free import (
    DUST3R_REPO,
    FAST3R_REPO,
    INSTALL_HELP,
    MAST3R_METRIC_REPO,
    PoseFreeError,
    PoseFreeResult,
    backend_is_available,
)

DEFAULT_IMAGE_SIZE = 512


class PoseFreeBackend(ABC):
    """Common interface for a pose-free reconstruction method."""

    name: str = "posefree"
    repo: str = ""
    returns_metric_scale: bool = False

    def __init__(self, device: str = "cuda", image_size: int = DEFAULT_IMAGE_SIZE):
        self.device = device
        self.image_size = image_size
        self._model = None

    def ensure_available(self) -> None:
        available, reason = backend_is_available(self.name)
        if not available:
            raise PoseFreeError(reason)

    @abstractmethod
    def load(self) -> "PoseFreeBackend":
        """Load the weights. Idempotent."""

    @abstractmethod
    def reconstruct(self, specimen) -> PoseFreeResult:
        """Run the method over one specimen's views."""

    def _image_paths(self, specimen) -> tuple[list[str], list[str]]:
        views = sorted(specimen.views, key=lambda v: v.azimuth_deg)
        return [str(v.rgb_path) for v in views], [v.position_id for v in views]


class Dust3rBackend(PoseFreeBackend):
    """DUSt3R: pairwise point maps fused by global alignment.

    Pairwise means the cost grows with the square of the view count. Twelve views
    is 66 pairs at ``complete`` pairing, which is affordable; the swin or oneref
    strategies exist if it is not.
    """

    name = "dust3r"
    repo = DUST3R_REPO
    returns_metric_scale = False

    def load(self) -> "Dust3rBackend":
        if self._model is not None:
            return self
        self.ensure_available()
        try:
            from dust3r.model import AsymmetricCroCo3DStereo
        except ImportError as exc:
            raise PoseFreeError(INSTALL_HELP["dust3r"]) from exc

        self._model = AsymmetricCroCo3DStereo.from_pretrained(self.repo).to(self.device)
        self._model.eval()
        return self

    def reconstruct(self, specimen) -> PoseFreeResult:
        self.load()
        try:
            from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
            from dust3r.image_pairs import make_pairs
            from dust3r.inference import inference
            from dust3r.utils.image import load_images
        except ImportError as exc:
            raise PoseFreeError(INSTALL_HELP["dust3r"]) from exc

        paths, position_ids = self._image_paths(specimen)
        images = load_images(paths, size=self.image_size)
        pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)
        output = inference(pairs, self._model, self.device, batch_size=1)

        scene = global_aligner(
            output, device=self.device, mode=GlobalAlignerMode.PointCloudOptimizer
        )
        scene.compute_global_alignment(init="mst", niter=300, schedule="cosine", lr=0.01)

        return _from_scene(scene, self.name, position_ids, is_metric=False)


class Mast3rBackend(PoseFreeBackend):
    """MASt3R with the metric checkpoint.

    The metric variant is the one worth running here: the others return arbitrary
    scale, and a volume computed from arbitrary scale measures the rescaling.
    """

    name = "mast3r"
    repo = MAST3R_METRIC_REPO
    returns_metric_scale = True

    def load(self) -> "Mast3rBackend":
        if self._model is not None:
            return self
        self.ensure_available()
        try:
            from mast3r.model import AsymmetricMASt3R
        except ImportError as exc:
            raise PoseFreeError(INSTALL_HELP["mast3r"]) from exc

        self._model = AsymmetricMASt3R.from_pretrained(self.repo).to(self.device)
        self._model.eval()

        if "metric" not in self.repo:
            raise PoseFreeError(
                f"{self.repo} is not the metric checkpoint. Use "
                f"{MAST3R_METRIC_REPO}, or the reconstruction will carry an "
                "arbitrary scale and its volume will be meaningless."
            )
        return self

    def reconstruct(self, specimen) -> PoseFreeResult:
        self.load()
        try:
            from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
            from dust3r.image_pairs import make_pairs
            from dust3r.inference import inference
            from dust3r.utils.image import load_images
        except ImportError as exc:
            raise PoseFreeError(INSTALL_HELP["mast3r"]) from exc

        paths, position_ids = self._image_paths(specimen)
        images = load_images(paths, size=self.image_size)
        pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)
        output = inference(pairs, self._model, self.device, batch_size=1)

        scene = global_aligner(
            output, device=self.device, mode=GlobalAlignerMode.PointCloudOptimizer
        )
        scene.compute_global_alignment(init="mst", niter=300, schedule="cosine", lr=0.01)

        return _from_scene(scene, self.name, position_ids, is_metric=True)


class Fast3rBackend(PoseFreeBackend):
    """Fast3R: every view in a single forward pass.

    No pairwise blow-up and no global alignment step, which is why it is the one
    to try first on twelve views.
    """

    name = "fast3r"
    repo = FAST3R_REPO
    returns_metric_scale = False

    def load(self) -> "Fast3rBackend":
        if self._model is not None:
            return self
        self.ensure_available()
        try:
            from fast3r.models.fast3r import Fast3R
        except ImportError as exc:
            raise PoseFreeError(INSTALL_HELP["fast3r"]) from exc

        self._model = Fast3R.from_pretrained(self.repo).to(self.device)
        self._model.eval()
        return self

    def reconstruct(self, specimen) -> PoseFreeResult:
        self.load()
        try:
            from fast3r.dust3r.inference_multiview import inference_multiview
            from fast3r.dust3r.utils.image import load_images
        except ImportError as exc:
            raise PoseFreeError(INSTALL_HELP["fast3r"]) from exc

        paths, position_ids = self._image_paths(specimen)
        images = load_images(paths, size=self.image_size)
        output = inference_multiview(self._model, images, device=self.device)

        return _from_multiview(output, self.name, position_ids)


# --------------------------------------------------------------------------
# Output normalisation
# --------------------------------------------------------------------------


def _from_scene(scene, method: str, position_ids: list[str], *, is_metric: bool):
    """Normalise a DUSt3R/MASt3R global-alignment scene."""
    poses = scene.get_im_poses().detach().cpu().numpy()      # (V, 4, 4) cam-to-world
    points = np.concatenate(
        [p.detach().cpu().numpy().reshape(-1, 3) for p in scene.get_pts3d()], axis=0
    )
    confidence = None
    if hasattr(scene, "get_conf"):
        confidence = np.concatenate(
            [c.detach().cpu().numpy().reshape(-1) for c in scene.get_conf()], axis=0
        )

    return PoseFreeResult(
        method=method,
        position_ids=list(position_ids[: poses.shape[0]]),
        rotations=poses[:, :3, :3],
        centres=poses[:, :3, 3],
        points=points,
        confidence=confidence,
        is_metric=is_metric,
        notes=[] if is_metric else ["scale is arbitrary; align to depth before use"],
    )


def _from_multiview(output, method: str, position_ids: list[str]):
    """Normalise a Fast3R multi-view output."""
    preds = output["preds"] if isinstance(output, dict) else output
    rotations, centres, clouds = [], [], []

    for view in preds:
        pose = view["camera_pose"] if "camera_pose" in view else view.get("pose")
        if pose is None:
            raise PoseFreeError(
                "Fast3R output carries no camera pose; the upstream API changed"
            )
        pose = np.asarray(pose.detach().cpu() if hasattr(pose, "detach") else pose)
        pose = pose.reshape(4, 4)
        rotations.append(pose[:3, :3])
        centres.append(pose[:3, 3])

        pts = view.get("pts3d_in_other_view", view.get("pts3d"))
        if pts is not None:
            arr = pts.detach().cpu().numpy() if hasattr(pts, "detach") else np.asarray(pts)
            clouds.append(arr.reshape(-1, 3))

    return PoseFreeResult(
        method=method,
        position_ids=list(position_ids[: len(rotations)]),
        rotations=np.stack(rotations),
        centres=np.stack(centres),
        points=np.concatenate(clouds, axis=0) if clouds else np.zeros((0, 3)),
        is_metric=False,
        notes=["scale is arbitrary; align to depth before use"],
    )


def sanity_check_result(result: PoseFreeResult) -> list[str]:
    """Catch a convention change before it becomes a wrong number.

    All three backends document OpenCV camera-to-world poses, so the optical axis
    (+z in camera coordinates) should point roughly at the scene. If an upstream
    release switches to OpenGL, or to world-to-camera, this is where it shows --
    as a warning rather than as a plausible-looking reconstruction that is
    inside out.
    """
    warnings: list[str] = []

    if result.n_views != result.rotations.shape[0]:
        warnings.append(
            f"{result.n_views} position ids but {result.rotations.shape[0]} poses"
        )

    for index in range(result.rotations.shape[0]):
        determinant = float(np.linalg.det(result.rotations[index]))
        if abs(determinant - 1.0) > 1e-3:
            warnings.append(
                f"view {index}: rotation determinant {determinant:+.3f}, not a proper rotation"
            )

    if result.points.size:
        centroid = result.points.mean(axis=0)
        facing = 0
        for index in range(result.rotations.shape[0]):
            forward = result.rotations[index] @ np.array([0.0, 0.0, 1.0])
            to_centre = centroid - result.centres[index]
            norm = np.linalg.norm(to_centre)
            if norm > 1e-9 and forward @ (to_centre / norm) > 0:
                facing += 1
        if facing < result.rotations.shape[0] / 2:
            warnings.append(
                f"only {facing}/{result.rotations.shape[0]} cameras face the scene "
                "centroid; the pose convention may have changed upstream "
                "(OpenGL vs OpenCV, or world-from-camera vs camera-from-world)"
            )

    return warnings


BACKENDS = {
    "dust3r": Dust3rBackend,
    "mast3r": Mast3rBackend,
    "fast3r": Fast3rBackend,
}


def build_backend(method: str, **kwargs) -> PoseFreeBackend:
    """Construct a pose-free backend by name."""
    if method not in BACKENDS:
        raise PoseFreeError(
            f"unknown method {method!r}; expected one of {sorted(BACKENDS)}"
        )
    return BACKENDS[method](**kwargs)


__all__ = [
    "BACKENDS",
    "Dust3rBackend",
    "Fast3rBackend",
    "Mast3rBackend",
    "PoseFreeBackend",
    "build_backend",
    "sanity_check_result",
]
