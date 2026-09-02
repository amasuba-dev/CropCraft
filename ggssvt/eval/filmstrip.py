"""The whole pipeline for one capture, as pictures.

The page could show a finished reconstruction and its number, and that is what it
did. What it could not show is the evidence in between: the frames the cameras
returned, which pixels the segmenter kept, what the depth sensor actually
measured, and what each operator then made of it. Those are the steps where this
project's findings live. E001's problem is not visible in its volume. It is
visible the moment you look at the raw frame and see a plant standing on an
upturned pot.

So each specimen gets a strip:

``capture``
    Six of the twelve colour frames, evenly spaced in azimuth, uncropped. The
    staging is part of the evidence, so the whole scene stays in frame.
``segmentation``
    The same six views with the subject mask outlined over a dimmed frame, and
    the measured depth beside it. Together they answer "did the camera see the
    plant" separately from "did the operator keep it".
``volume``
    What each operator built, in the project's own depth-cued projection.
``shaded``
    One presentation render of the mesh, lit rather than depth-cued.

**The shaded render is deliberately last and deliberately labelled.**
``render.py`` avoids shaded renders on purpose, and the reason is sound: a lit
surface reads as solid everywhere, which is exactly the property a carved hull
does not have. It hides the hollows and the places no camera ever saw. It is
here because a shaded view communicates shape to a reader faster than anything
else on the page, and it is kept out of every panel that carries a verdict. Read
it as an illustration and read the projection beside it as the evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import WORK_DIR

# Six of twelve. Enough to show that the rig goes all the way round and that the
# staging is the same at every azimuth, without a row that wraps on a laptop.
N_VIEWS = 6

TILE_H = 150          # thumbnail height in pixels; width follows the frame
SHADED_PX = 460       # the one large render
SPLAT_RADIUS = 1      # half-width of a vertex splat, in output pixels

# Light in view space: up, slightly left, slightly toward the camera. Chosen so
# a leaf facing the viewer is bright and the underside of the canopy is not
# black, because a black region reads as absence rather than as shadow.
LIGHT = np.array([-0.35, 0.72, 0.60])
AMBIENT = 0.30


def _view_indices(n_total: int, n_wanted: int = N_VIEWS) -> list[int]:
    """Evenly spaced views, so the row spans the full turntable."""
    if n_total <= n_wanted:
        return list(range(n_total))
    return sorted({int(round(k * n_total / n_wanted)) % n_total
                   for k in range(n_wanted)})


def _downscale(image: np.ndarray, height: int = TILE_H) -> np.ndarray:
    """Box-average down to the target height."""
    factor = max(1, image.shape[0] // height)
    h = (image.shape[0] // factor) * factor
    w = (image.shape[1] // factor) * factor
    cropped = image[:h, :w].astype(np.float32)
    shape = (h // factor, factor, w // factor, factor) + image.shape[2:]
    return cropped.reshape(shape).mean(axis=(1, 3)).astype(np.uint8)


def _outline(mask: np.ndarray) -> np.ndarray:
    """One-pixel boundary, from the mask's disagreement with its own shifts."""
    edge = np.zeros_like(mask)
    edge[:-1, :] |= mask[:-1, :] != mask[1:, :]
    edge[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    return edge


def _rgb_tile(cached, view: int) -> np.ndarray:
    """The frame as the camera returned it."""
    return _downscale(np.asarray(cached.rgb)[view])


def _mask_tile(cached, view: int) -> np.ndarray:
    """Subject mask outlined over a dimmed frame.

    Dimmed rather than blacked out, because the question a reader asks here is
    what the segmenter *excluded*, and that cannot be answered against a void.
    """
    frame = np.asarray(cached.rgb)[view].astype(np.float32)
    mask = np.asarray(cached.mask)[view]

    out = frame * 0.32
    out[mask] = frame[mask]
    out[_outline(mask)] = (255, 214, 87)
    return _downscale(np.clip(out, 0, 255).astype(np.uint8))


def _depth_tile(cached, view: int) -> np.ndarray:
    """Measured depth, viridis over the valid range, with the mask outlined.

    Where the sensor returned nothing the pixel stays grey rather than taking the
    low end of the ramp, so a hole in the depth is not read as a near surface.
    """
    from .render import viridis

    depth = np.asarray(cached.depth_m)[view]
    mask = np.asarray(cached.mask)[view]
    valid = depth > 0

    out = np.full(depth.shape + (3,), 232, dtype=np.uint8)
    if valid.any():
        inside = depth[valid & mask] if (valid & mask).any() else depth[valid]
        lo, hi = float(inside.min()), float(inside.max())
        span = max(hi - lo, 1e-6)
        # Near is bright, matching the projections elsewhere on the page.
        t = np.clip(1.0 - (depth[valid] - lo) / span, 0.0, 1.0)
        out[valid] = viridis(t)

    out[_outline(mask)] = (40, 40, 44)
    return _downscale(out)


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals, oriented outward.

    The orientation is not decoration. Marching cubes here winds its faces so
    that the accumulated normals point *into* the volume: measured across a Mango
    mesh, only 29 percent of them agree with the outward radial direction. Lit
    with those, every surface facing the viewer faces away from the light and the
    whole render comes back at the ambient term, a near-black silhouette that
    looks like a shading bug rather than a plant.

    A global flip is the right correction rather than a per-vertex one. The sign
    is consistent across the mesh, so the majority vote recovers it, while
    flipping vertices individually would erase the concavities that are the only
    reason to light the surface at all.
    """
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    face_normal = np.cross(b - a, c - a)
    normals = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(normals, faces[:, i], face_normal)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(length, 1e-12)

    radial = vertices - vertices.mean(axis=0)
    if float((normals * radial).sum()) < 0.0:
        normals = -normals
    return normals


def _rotate(points: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    """The viewer's own camera convention, so the render matches the canvas."""
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    rx = x * cy + y * sy
    rz = -x * sy + y * cy
    return np.column_stack([rx, z * cp - rz * sp, z * sp + rz * cp])


def shaded_render(mesh, *, size: int = SHADED_PX, yaw: float = 0.62,
                  pitch: float = 0.18, supersample: int = 2) -> np.ndarray:
    """Lambert-shaded splat render of a mesh.

    Vertices rather than triangles: rasterising every face means a Python loop
    over as many as two hundred thousand of them, and splatting each vertex with
    its own normal closes the same surface in one vectorised pass. Depth ordering
    is exact, by sorting every splat sample back to front and letting the nearer
    write land last.

    **The splat radius has to follow the vertex count or the surface does not
    close.** A fixed radius of one pixel left E001 as a stipple: 5,945 vertices
    covering five pixels each reach barely half of the silhouette they should
    fill, while a Mango's two hundred thousand at the same radius would be solid
    several times over. The radius is solved for instead, from the area each
    vertex has to cover, and the whole thing is rendered at twice the output size
    and averaged down so the edges are not staircases.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        return np.full((size, size, 3), 255, dtype=np.uint8)

    normals = _rotate(_vertex_normals(vertices, faces), yaw, pitch)
    view = _rotate(vertices - vertices.mean(axis=0), yaw, pitch)

    inner = size * max(1, supersample)
    half = max(float(np.abs(view[:, :2]).max()), 1e-6) * 1.06
    px = np.round(inner / 2 + view[:, 0] / half * (inner / 2)).astype(np.int64)
    py = np.round(inner / 2 - view[:, 1] / half * (inner / 2)).astype(np.int64)

    light = LIGHT / np.linalg.norm(LIGHT)
    lambert = np.clip(normals @ light, 0.0, 1.0)
    shade = AMBIENT + (1.0 - AMBIENT) * lambert

    # A green that reads as foliage without pretending to be a measured colour,
    # plus a narrow specular term so the surface has a direction to it.
    base = np.array([104.0, 168.0, 96.0])
    colour = np.clip(shade[:, None] * base + 26.0 * shade[:, None] ** 8, 0, 255)

    # The radius comes from the mesh's own vertex spacing, not from an area
    # budget. Marching cubes puts vertices on the voxel lattice, so they project
    # to a regular grid, and an area estimate is defeated by it twice over: the
    # silhouette is covered by both the front and the back of a closed surface,
    # and a lattice leaves holes an average cannot see. Discs of 0.71 times the
    # spacing are what close a square lattice, so that is what is asked for.
    a, b = vertices[faces[:, 0]], vertices[faces[:, 1]]
    # The 75th percentile rather than the median. Marching cubes returns edges
    # from near zero, where two vertices share a grid point, up to the voxel
    # pitch, and it is the long ones that decide whether the surface closes: a
    # radius sized for the median leaves the widely spaced regions stippled
    # while the crowded ones were already covered several times over.
    edge_m = float(np.percentile(np.linalg.norm(b - a, axis=1), 75))
    spacing_px = edge_m * (inner / 2.0) / half
    r = int(np.clip(np.ceil(0.71 * spacing_px), SPLAT_RADIUS, 14))

    offsets = [(dx, dy) for dy in range(-r, r + 1) for dx in range(-r, r + 1)
               if dx * dx + dy * dy <= r * r]
    xs = np.concatenate([px + dx for dx, _ in offsets])
    ys = np.concatenate([py + dy for _, dy in offsets])
    zs = np.tile(view[:, 2], len(offsets))
    cs = np.tile(colour, (len(offsets), 1))

    inside = (xs >= 0) & (xs < inner) & (ys >= 0) & (ys < inner)
    xs, ys, zs, cs = xs[inside], ys[inside], zs[inside], cs[inside]

    order = np.argsort(zs, kind="stable")      # far first, so near overwrites
    image = np.full((inner * inner, 3), 255.0)
    image[ys[order] * inner + xs[order]] = cs[order]
    image = image.reshape(inner, inner, 3).astype(np.uint8)
    return _downscale(image, height=size) if inner != size else image


def _save(image: np.ndarray, path: Path, *, quality: int = 82) -> int:
    """Write a tile, JPEG for photographs and PNG for anything with hard edges."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    picture = Image.fromarray(np.ascontiguousarray(image))
    if path.suffix == ".jpg":
        picture.save(path, quality=quality, optimize=True)
    else:
        picture.save(path, optimize=True)
    return path.stat().st_size


def strip_for(
    plant_id: str,
    *,
    cache_dir: Path,
    fused_dir: Path | None,
    out_dir: Path,
    n_views: int = N_VIEWS,
) -> dict:
    """Render every tile for one specimen and return its manifest."""
    from ..data.preprocess import load_cached
    from ..geometry.mesh import mesh_from_occupancy
    from .render import render_volume

    cached = load_cached(plant_id, cache_dir)
    here = out_dir / plant_id
    rel = lambda name: f"{plant_id}/{name}"

    views = _view_indices(np.asarray(cached.mask).shape[0], n_views)
    frames = []
    for view in views:
        names = {}
        for kind, tile, suffix in (
            ("rgb", _rgb_tile(cached, view), ".jpg"),
            ("mask", _mask_tile(cached, view), ".jpg"),
            ("depth", _depth_tile(cached, view), ".png"),
        ):
            name = f"{kind}_{view:02d}{suffix}"
            _save(tile, here / name)
            names[kind] = rel(name)
        names["azimuth_deg"] = int(round(view * 360.0 / len(cached.position_ids)))
        frames.append(names)

    # What each operator built, in the projection the rest of the page uses.
    volumes = []
    grids = [("carve", cached.occupancy, "silhouette carving")]
    if fused_dir is not None and (fused_dir / "quality.json").exists():
        try:
            volumes_source = load_cached(plant_id, fused_dir)
        except (FileNotFoundError, KeyError):
            volumes_source = None
        if volumes_source is not None:
            grids.append(("fusion", volumes_source.occupancy, "depth fusion"))

    for key, grid, title in grids:
        name = f"volume_{key}.png"
        _save(render_volume(grid, view="front", size=300, point_radius=1), here / name)
        volumes.append({"key": key, "title": title, "image": rel(name)})

    shaded = None
    try:
        mesh = mesh_from_occupancy(cached.occupancy,
                                   voxel_size_m=float(cached.voxel_size_m))
        if mesh.n_faces:
            _save(shaded_render(mesh), here / "shaded.png")
            shaded = {
                "image": rel("shaded.png"),
                "n_vertices": mesh.n_vertices,
                "n_faces": mesh.n_faces,
                "area_m2": round(mesh.surface_area_m2(), 4),
            }
    except Exception:
        # A missing scikit-image costs the illustration and nothing else.
        shaded = None

    return {
        "plant_id": plant_id,
        "species": cached.species,
        "mass_kg": round(float(cached.target_kg), 3),
        "n_views_total": int(np.asarray(cached.mask).shape[0]),
        "frames": frames,
        "volumes": volumes,
        "shaded": shaded,
    }


def run(
    *,
    cache_dir: Path | None = None,
    fused_dir: Path | None = None,
    out_dir: Path | None = None,
    out: Path = WORK_DIR / "reports" / "filmstrip.json",
    limit: int | None = None,
    n_views: int = N_VIEWS,
    verbose: bool = True,
) -> dict:
    """Render the strip for every usable specimen and index it."""
    from ..data.preprocess import usable_plant_ids

    cache_dir = cache_dir or WORK_DIR / "cache"
    fused_dir = fused_dir or WORK_DIR / "cache_tsdf"
    out_dir = out_dir or WORK_DIR / "reports" / "filmstrip"
    out_dir.mkdir(parents=True, exist_ok=True)

    plant_ids = sorted(usable_plant_ids(cache_dir))
    if limit:
        plant_ids = plant_ids[:limit]

    specimens = []
    for plant_id in plant_ids:
        entry = strip_for(plant_id, cache_dir=cache_dir, fused_dir=fused_dir,
                          out_dir=out_dir, n_views=n_views)
        specimens.append(entry)
        if verbose:
            print(f"  {plant_id:6s} {len(entry['frames'])} views, "
                  f"{len(entry['volumes'])} volumes"
                  + (", shaded" if entry["shaded"] else ""), flush=True)

    total = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
    report = {
        "note": "raw frames, masks, depth and volumes per specimen, written as "
                "files rather than embedded so the page stays small and the "
                "tiles load on demand",
        "caveat": "the shaded render is an illustration, not evidence: a lit "
                  "surface reads as solid everywhere, which is the property a "
                  "carved hull does not have",
        "n_views": n_views,
        "n_specimens": len(specimens),
        "bytes_on_disk": total,
        "specimens": specimens,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, separators=(",", ":")), encoding="utf-8")
    if verbose:
        print(f"\n  wrote {out} and {total // 1024} KB of tiles under {out_dir}")
    return report


__all__ = ["N_VIEWS", "SHADED_PX", "run", "shaded_render", "strip_for"]
