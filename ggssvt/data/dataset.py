"""Specimen index over ``dataset/plants`` joined to ``dataset/ground_truth.csv``."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import (
    EXCLUDED_PLANTS,
    GROUND_TRUTH_CSV,
    PLANTS_DIR,
    SPECIES_COLUMN,
    TARGET_COLUMN,
)
from .io import load_depth, load_rgb
from .naming import Position, resolve_positions


@dataclass(frozen=True)
class GroundTruth:
    """One row of ``ground_truth.csv``."""

    plant_id: str
    date: str
    species: str
    total_fresh_weight_with_pot_g: float
    pot_weight_g: float
    net_weight_g: float
    pot_weight_source: str
    notes: str

    @property
    def net_weight_kg(self) -> float:
        return self.net_weight_g / 1000.0

    @property
    def pot_weight_is_estimated(self) -> bool:
        return self.pot_weight_source.strip().lower() != "weighed"


@dataclass(frozen=True)
class View:
    """One RGB-D frame taken from one rig position."""

    position: Position
    rgb_path: Path
    depth_path: Path

    @property
    def position_id(self) -> str:
        return self.position.position_id

    @property
    def azimuth_deg(self) -> int:
        return self.position.azimuth_deg

    @property
    def azimuth_rad(self) -> float:
        return math.radians(self.position.azimuth_deg)

    def load_rgb(self) -> np.ndarray:
        return load_rgb(self.rgb_path)

    def load_depth(self) -> np.ndarray:
        return load_depth(self.depth_path)


@dataclass
class Specimen:
    """One plant: its views, its ground truth, and any data-integrity notes."""

    plant_id: str
    root: Path
    views: list[View]
    ground_truth: GroundTruth | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def n_views(self) -> int:
        return len(self.views)

    @property
    def species(self) -> str | None:
        return self.ground_truth.species if self.ground_truth else None

    @property
    def target_kg(self) -> float | None:
        return self.ground_truth.net_weight_kg if self.ground_truth else None

    @property
    def azimuths_deg(self) -> list[int]:
        return [v.azimuth_deg for v in self.views]

    def view(self, position_id: str) -> View:
        for v in self.views:
            if v.position_id == position_id:
                return v
        raise KeyError(f"{self.plant_id} has no view {position_id!r}")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        target = "n/a" if self.target_kg is None else f"{self.target_kg:.3f} kg"
        return (
            f"Specimen({self.plant_id}, {self.n_views} views, "
            f"{self.species or 'unknown'}, {target})"
        )


def _normalise_species(raw: str) -> str:
    """Strip the stray punctuation in rows such as ``Eucalyptus'``."""
    return raw.strip().strip("'\"").strip()


def load_ground_truth(path: Path = GROUND_TRUTH_CSV) -> dict[str, GroundTruth]:
    """Read ``ground_truth.csv`` keyed by plant id."""
    if not path.exists():
        raise FileNotFoundError(f"ground truth CSV not found at {path}")

    rows: dict[str, GroundTruth] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            plant_id = row["plant_id"].strip()
            if not plant_id:
                continue
            rows[plant_id] = GroundTruth(
                plant_id=plant_id,
                date=row.get("date", "").strip(),
                species=_normalise_species(row.get(SPECIES_COLUMN, "")),
                total_fresh_weight_with_pot_g=float(
                    row["total_fresh_weight_with_pot_g"]
                ),
                pot_weight_g=float(row["pot_weight_g"]),
                net_weight_g=float(row[TARGET_COLUMN]),
                pot_weight_source=row.get("pot_weight_source", "").strip(),
                notes=row.get("notes", "").strip(),
            )
    return rows


def _discover_views(plant_dir: Path) -> tuple[list[View], list[str]]:
    """Build the view list for one plant directory.

    The per-plant ``frames_manifest.json`` is authoritative when present; the
    ``images``/``depth`` directories are the fallback. Either way, positions are
    resolved through :func:`ggssvt.data.naming.resolve_positions` so the two
    camB naming conventions collapse onto the same physical azimuths.
    """
    warnings: list[str] = []
    manifest_path = plant_dir / "frames_manifest.json"

    entries: dict[str, tuple[Path, Path]] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for position_id, paths in manifest.items():
            entries[position_id] = (
                plant_dir / paths["rgb"],
                plant_dir / paths["depth"],
            )
    else:
        warnings.append("no frames_manifest.json; falling back to directory listing")
        for rgb_path in sorted((plant_dir / "images").glob("*.png")):
            depth_path = plant_dir / "depth" / rgb_path.name
            if depth_path.exists():
                entries[rgb_path.stem] = (rgb_path, depth_path)

    positions, dropped = resolve_positions(list(entries))
    for pid in dropped:
        warnings.append(
            f"dropped {pid}: its physical azimuth duplicates an earlier view"
        )

    views: list[View] = []
    for position in positions:
        rgb_path, depth_path = entries[position.position_id]
        if not rgb_path.exists():
            warnings.append(f"{position.position_id}: missing RGB at {rgb_path}")
            continue
        if not depth_path.exists():
            warnings.append(f"{position.position_id}: missing depth at {depth_path}")
            continue
        views.append(View(position=position, rgb_path=rgb_path, depth_path=depth_path))

    return views, warnings


def load_specimen(
    plant_id: str,
    *,
    plants_dir: Path = PLANTS_DIR,
    ground_truth: dict[str, GroundTruth] | None = None,
) -> Specimen:
    """Load one specimen by id."""
    plant_dir = plants_dir / plant_id
    if not plant_dir.is_dir():
        raise FileNotFoundError(f"no capture directory for {plant_id!r} at {plant_dir}")

    if ground_truth is None:
        ground_truth = load_ground_truth()

    views, warnings = _discover_views(plant_dir)

    gt = ground_truth.get(plant_id)
    if gt is None:
        warnings.append("no ground-truth row; specimen is reconstruction-only")

    if len(views) != 12:
        warnings.append(f"{len(views)} views resolved, expected 12")

    return Specimen(
        plant_id=plant_id,
        root=plant_dir,
        views=views,
        ground_truth=gt,
        warnings=warnings,
    )


def load_dataset(
    *,
    plants_dir: Path = PLANTS_DIR,
    require_ground_truth: bool = True,
    min_views: int = 12,
    exclude: tuple[str, ...] = EXCLUDED_PLANTS,
    species: str | None = None,
) -> list[Specimen]:
    """Load every usable specimen.

    Args:
        require_ground_truth: drop specimens with no row in the CSV.
        min_views: drop specimens with fewer resolved views than this.
        exclude: plant ids to skip outright.
        species: if given, keep only this species (case-insensitive).

    Returns:
        Specimens sorted by plant id.
    """
    gt = load_ground_truth()
    specimens: list[Specimen] = []

    for plant_dir in sorted(p for p in plants_dir.iterdir() if p.is_dir()):
        plant_id = plant_dir.name
        if plant_id in exclude:
            continue

        specimen = load_specimen(plant_id, plants_dir=plants_dir, ground_truth=gt)

        if require_ground_truth and specimen.ground_truth is None:
            continue
        if specimen.n_views < min_views:
            continue
        if species is not None and (specimen.species or "").lower() != species.lower():
            continue

        specimens.append(specimen)

    return specimens


def dataset_summary(specimens: list[Specimen]) -> dict:
    """Aggregate counts and target statistics, for the CLI and the report."""
    targets = np.array(
        [s.target_kg for s in specimens if s.target_kg is not None], dtype=np.float64
    )
    species_counts: dict[str, int] = {}
    for s in specimens:
        key = s.species or "unknown"
        species_counts[key] = species_counts.get(key, 0) + 1

    summary = {
        "n_specimens": len(specimens),
        "n_views_total": sum(s.n_views for s in specimens),
        "species_counts": species_counts,
        "n_with_warnings": sum(1 for s in specimens if s.warnings),
    }
    if targets.size:
        summary["target_kg"] = {
            "min": float(targets.min()),
            "max": float(targets.max()),
            "mean": float(targets.mean()),
            "std": float(targets.std(ddof=1)) if targets.size > 1 else 0.0,
        }
    return summary


__all__ = [
    "GroundTruth",
    "Specimen",
    "View",
    "dataset_summary",
    "load_dataset",
    "load_ground_truth",
    "load_specimen",
]
