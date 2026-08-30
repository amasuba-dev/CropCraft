"""One configurable entry point for looking at anything in the pipeline.

Until now the visualisation surface was scattered: `gallery` drew contact sheets
and PLY files, `visualise` drew rig and mask overlays, `mesh --export` wrote OBJ,
and each had its own flags. Wanting to see one specimen's segmentation next to
its reconstruction at a size suitable for a figure meant three commands and a
mental note about which one took `--columns`.

This is the pyplot-shaped alternative: a config object holding every knob, and
one call that renders whichever layers were asked for. The config is the API, so
adding a knob does not mean adding a flag to four commands.

    from ggssvt.eval.viz import VizConfig, render
    render("M001", VizConfig(layers=("rgb", "segmentation", "occupancy"),
                             size=400, view_index=3))

**On colormaps.** Only perceptually uniform ramps are offered, plus greys. Jet
and rainbow are not, deliberately: they invent structure that is not in the data
by putting sharp luminance edges at arbitrary values, and a depth-cued plant
projection is exactly the kind of image where a reader would mistake that for a
boundary. Greyscale is included because a printed thesis is often read in it, and
a figure that only works in colour is a figure that fails at the viva.

**On backends.** PIL is the default and needs nothing beyond what the project
already installs. Matplotlib is offered for interactive 3D inspection, where
rotating a point cloud is worth more than any static projection, and is imported
only when asked for so it stays optional.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import WORK_DIR

# What can be drawn. Each is a function of one cached specimen.
LAYERS = ("rgb", "depth", "segmentation", "occupancy", "points", "mesh")

# Where a reconstruction can come from. The names match the cache directories.
SOURCES = {"carve": "cache", "fused": "cache_tsdf", "sam3d": "cache_sam3d"}

# Perceptually uniform ramps only, plus greys. See the module docstring.
COLORMAPS = ("viridis", "greys", "greys_r")


@dataclass
class VizConfig:
    """Every knob, in one place. The pyplot rcParams of this project.

    Attributes:
        layers: which panels to draw, in order, from :data:`LAYERS`.
        views: axes for the ``occupancy`` and ``points`` layers.
        view_index: which captured view the per-view layers use.
        size: pixels per panel. 200 for a contact sheet, 400+ for a figure.
        point_radius: half-width of the square drawn per voxel, output pixels.
        supersample: render at this multiple and average down. 1 disables it.
        cmap: from :data:`COLORMAPS`.
        source: which reconstruction, from :data:`SOURCES`.
        max_points: cap for the ``points`` layer and the interactive backend.
        label: draw a caption strip with the specimen's id, species and mass.
        backend: ``pil`` writes an image; ``matplotlib`` opens an interactive
            window, which is worth it for point clouds and nothing else.
    """

    layers: tuple[str, ...] = ("occupancy",)
    views: tuple[str, ...] = ("front", "side", "top")
    view_index: int = 0
    size: int = 300
    point_radius: int = 1
    supersample: int = 3
    cmap: str = "viridis"
    background: tuple[int, int, int] = (255, 255, 255)
    source: str = "carve"
    max_points: int = 20000
    label: bool = True
    backend: str = "pil"
    cache_root: Path = field(default=WORK_DIR)

    def __post_init__(self) -> None:
        unknown = [layer for layer in self.layers if layer not in LAYERS]
        if unknown:
            raise ValueError(
                f"unknown layer(s) {unknown}; expected any of {list(LAYERS)}"
            )
        if self.source not in SOURCES:
            raise ValueError(
                f"unknown source {self.source!r}; expected one of {list(SOURCES)}"
            )
        if self.cmap not in COLORMAPS:
            raise ValueError(
                f"unknown colormap {self.cmap!r}; expected one of {list(COLORMAPS)}. "
                "Jet and rainbow are deliberately absent: they put luminance "
                "edges at arbitrary values and a reader reads those as structure."
            )

    @property
    def cache_dir(self) -> Path:
        return self.cache_root / SOURCES[self.source]


def ramp(values: np.ndarray, cmap: str) -> np.ndarray:
    """Map ``values`` in [0, 1] onto a colormap. Returns ``(n, 3)`` uint8."""
    from .render import viridis

    values = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    if cmap == "viridis":
        return viridis(values)

    grey = values if cmap == "greys_r" else 1.0 - values
    channel = np.round(grey * 255).astype(np.uint8)
    return np.stack([channel] * 3, axis=-1)


def _caption(cached, config: VizConfig) -> str:
    litres = cached.occupancy.sum() * cached.voxel_size_m ** 3 * 1000
    return (f"{cached.plant_id}  {cached.species[:12]}  "
            f"{float(cached.target_kg):.2f} kg  {litres:.1f} L  [{config.source}]")


def _panel_rgb(cached, config: VizConfig) -> np.ndarray:
    return cached.rgb[config.view_index]


def _panel_depth(cached, config: VizConfig) -> np.ndarray:
    """Depth as a colormapped image, scaled over the subject's own range.

    Scaled to the subject rather than the sensor range, because the plant
    occupies a small part of 0.5 to 4.5 m and a global scale would render it as
    one flat colour.
    """
    depth = cached.depth_m[config.view_index]
    subject = cached.mask[config.view_index].astype(bool) & (depth > 0)
    if not subject.any():
        return np.full((*depth.shape, 3), config.background, dtype=np.uint8)

    low, high = float(depth[subject].min()), float(depth[subject].max())
    span = max(high - low, 1e-6)
    normalised = np.clip((depth - low) / span, 0.0, 1.0)
    coloured = ramp(normalised.ravel(), config.cmap).reshape(*depth.shape, 3)
    return np.where(subject[..., None], coloured,
                    np.asarray(config.background, dtype=np.uint8))


def _panel_segmentation(cached, config: VizConfig) -> np.ndarray:
    """The subject mask over the colour image, so the cut is visible.

    Tinted rather than replaced: a binary mask beside an RGB frame makes you
    compare two pictures, and what matters is where the boundary falls on the
    plant.
    """
    rgb = cached.rgb[config.view_index].astype(np.float64)
    mask = cached.mask[config.view_index].astype(bool)
    tint = np.asarray(ramp(np.array([0.75]), config.cmap)[0], dtype=np.float64)

    out = rgb * 0.45                                  # dim what was excluded
    out[mask] = rgb[mask] * 0.6 + tint * 0.4
    return np.clip(out, 0, 255).astype(np.uint8)


def _panel_occupancy(cached, config: VizConfig) -> np.ndarray:
    from .render import render_volume

    panels = [
        render_volume(cached.occupancy, view=view, size=config.size,
                      background=config.background,
                      point_radius=config.point_radius,
                      supersample=config.supersample)
        for view in config.views
    ]
    return np.concatenate(panels, axis=1)


def _panel_points(cached, config: VizConfig) -> np.ndarray:
    """A thinned point cloud, drawn the same way but sparser.

    The difference from ``occupancy`` is honesty about density: a subsampled
    cloud shows where the reconstruction is thin, which a filled projection
    hides.
    """
    from .render import render_volume

    occupancy = cached.occupancy
    total = int(occupancy.sum())
    if total > config.max_points:
        index = np.array(np.nonzero(occupancy)).T
        keep = np.random.default_rng(0).choice(
            total, config.max_points, replace=False
        )
        thinned = np.zeros_like(occupancy)
        chosen = index[keep]
        thinned[chosen[:, 0], chosen[:, 1], chosen[:, 2]] = True
        occupancy = thinned

    panels = [
        render_volume(occupancy, view=view, size=config.size,
                      background=config.background,
                      point_radius=max(1, config.point_radius),
                      supersample=config.supersample)
        for view in config.views
    ]
    return np.concatenate(panels, axis=1)


def _panel_mesh(cached, config: VizConfig) -> np.ndarray:
    """Marching-cubes surface, projected. Needs scikit-image."""
    from ..geometry.mesh import mesh_from_occupancy
    from .render import render_volume

    try:
        mesh = mesh_from_occupancy(cached.occupancy,
                                   voxel_size_m=cached.voxel_size_m)
    except Exception as exc:
        raise RuntimeError(
            f"mesh layer needs scikit-image: {exc}. `pip install scikit-image`, "
            "or drop 'mesh' from layers."
        ) from exc

    # Vertices back onto the grid, so the projection code is shared and the
    # mesh lines up with the occupancy panel beside it.
    resolution = cached.occupancy.shape[0]
    index = np.floor(
        (mesh.vertices / cached.voxel_size_m)
        + np.array([resolution / 2, resolution / 2, 0.0])
    ).astype(int)
    inside = np.all((index >= 0) & (index < resolution), axis=1)
    surface = np.zeros_like(cached.occupancy)
    kept = index[inside]
    surface[kept[:, 0], kept[:, 1], kept[:, 2]] = True

    panels = [
        render_volume(surface, view=view, size=config.size,
                      background=config.background,
                      point_radius=config.point_radius,
                      supersample=config.supersample)
        for view in config.views
    ]
    return np.concatenate(panels, axis=1)


_PANELS = {
    "rgb": _panel_rgb,
    "depth": _panel_depth,
    "segmentation": _panel_segmentation,
    "occupancy": _panel_occupancy,
    "points": _panel_points,
    "mesh": _panel_mesh,
}


def _stack(panels: list[np.ndarray], background) -> np.ndarray:
    """Pad to a common width and stack vertically."""
    width = max(p.shape[1] for p in panels)
    padded = []
    for panel in panels:
        canvas = np.full((panel.shape[0], width, 3), background, dtype=np.uint8)
        canvas[:, : panel.shape[1]] = panel
        padded.append(canvas)
    return np.concatenate(padded, axis=0)


def render(plant_id: str, config: VizConfig | None = None) -> np.ndarray:
    """Draw every requested layer for one specimen, stacked."""
    from ..data.preprocess import load_cached
    from .render import _label_strip

    config = config or VizConfig()
    cached = load_cached(plant_id, config.cache_dir)

    panels = [_PANELS[layer](cached, config) for layer in config.layers]
    image = _stack(panels, config.background)

    if config.label:
        strip = _label_strip(_caption(cached, config), image.shape[1])
        image = np.concatenate([strip, image], axis=0)
    return image


def show(plant_id: str, config: VizConfig | None = None):
    """Render, and either save or open an interactive window.

    The matplotlib backend draws the occupied voxels as a 3D scatter, which is
    the one thing a static projection cannot do: rotate it and the difference
    between a hull and a fused shell is obvious in a way no fixed viewpoint
    shows.
    """
    config = config or VizConfig()
    if config.backend != "matplotlib":
        return render(plant_id, config)

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "the matplotlib backend needs matplotlib, which this project does "
            "not otherwise require. `pip install matplotlib`, or use "
            "backend='pil'."
        ) from exc

    from ..data.preprocess import load_cached
    from .render import occupied_points

    cached = load_cached(plant_id, config.cache_dir)
    points = occupied_points(cached.occupancy, cached.voxel_size_m)
    if points.shape[0] > config.max_points:
        keep = np.random.default_rng(0).choice(
            points.shape[0], config.max_points, replace=False
        )
        points = points[keep]

    figure = plt.figure(figsize=(7, 7))
    axes = figure.add_subplot(111, projection="3d")
    height = points[:, 2]
    axes.scatter(points[:, 0], points[:, 1], height,
                 c=height, cmap="viridis" if config.cmap == "viridis" else "Greys",
                 s=config.point_radius * 4, marker=".", linewidths=0)
    axes.set_title(_caption(cached, config))
    axes.set_xlabel("x (m)")
    axes.set_ylabel("y (m)")
    axes.set_zlabel("z (m)")
    # Equal aspect, or a 1.5 m tall plant looks like a pancake.
    extent = np.ptp(points, axis=0).max() / 2.0
    middle = points.mean(axis=0)
    for setter, centre in zip(
        (axes.set_xlim, axes.set_ylim, axes.set_zlim), middle, strict=False
    ):
        setter(centre - extent, centre + extent)
    plt.tight_layout()
    plt.show()
    return figure


def save(plant_id: str, out: Path, config: VizConfig | None = None) -> Path:
    """Render and write a PNG."""
    from .render import _require_pil

    Image = _require_pil()
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(render(plant_id, config)).save(out)
    return out


__all__ = ["COLORMAPS", "LAYERS", "SOURCES", "VizConfig", "ramp", "render",
           "save", "show"]
