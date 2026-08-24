"""What has been run, what has not, and when.

The project page reports whatever numbers exist on disk, which makes it easy to
lose track of which experiments produced them and which are still outstanding.
Across a long project that turns into a real problem: a table of six methods
looks complete whether or not four more were meant to be in it.

So this walks the work directory and reports the state of every experiment in
the programme, including the ones that have never run. Each entry names the
artefact that proves it ran, the command that produces it, and where it sits in
the argument. Nothing here computes a result; it only reports whether one exists,
which means it stays cheap enough to run on every page build.

The `stale` flag compares an artefact against the cache it was derived from. A
result older than the specimens it was fitted to is worse than a missing one,
because it looks finished. That has already happened once in this project, when
the ground truth changed and every trained number silently predated it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..config import WORK_DIR


@dataclass(frozen=True)
class Experiment:
    """One step in the programme, and how to tell whether it has run."""

    key: str
    label: str
    stage: str            # data | reconstruction | classical | learned | check
    command: str
    artefact: Path
    why: str
    needs_gpu: bool = False


@dataclass
class ExperimentStatus:
    """Whether an experiment has run, and what it produced."""

    key: str
    label: str
    stage: str
    command: str
    why: str
    needs_gpu: bool
    done: bool
    stale: bool = False
    ran_at: str | None = None
    headline: str | None = None
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "stage": self.stage,
            "command": self.command,
            "why": self.why,
            "needs_gpu": self.needs_gpu,
            "done": self.done,
            "stale": self.stale,
            "ran_at": self.ran_at,
            "headline": self.headline,
        }


def _programme(work_dir: Path) -> tuple[Experiment, ...]:
    reports = work_dir / "reports"
    return (
        Experiment(
            "preprocess", "Preprocess, geometric carve", "data",
            "python -m ggssvt.cli preprocess",
            work_dir / "cache" / "quality.json",
            "Registers, segments and carves every specimen. Everything else reads "
            "this cache, so nothing runs before it.",
        ),
        Experiment(
            "preprocess_sam3d", "Preprocess, SAM3D masks", "data",
            "python -m ggssvt.cli preprocess --segmenter sam3d "
            "--cache-dir work_dirs/ggssvt/cache_sam3d --sam-device cuda",
            work_dir / "cache_sam3d" / "quality.json",
            "The segmentation arm of the factorial. Slower, and it drops three "
            "specimens the geometric gate keeps.",
            needs_gpu=True,
        ),
        Experiment(
            "views", "View-count caches, 3, 4 and 6", "data",
            "for v in 3 4 6; do python -m ggssvt.cli preprocess --views $v "
            "--cache-dir work_dirs/ggssvt/cache_v$v; done",
            work_dir / "cache_v4" / "quality.json",
            "Answers whether four images would have done. They would not.",
        ),
        Experiment(
            "fuse", "TSDF depth fusion", "reconstruction",
            "python -m ggssvt.cli fuse --write-cache",
            reports / "fusion.json",
            "The reconstruction that escapes the visual hull. Also writes the "
            "fused cache the campaign and the page both read.",
        ),
        Experiment(
            "baselines", "Classical biomass baselines", "classical",
            "python -m ggssvt.cli baselines",
            reports / "metrics.json",
            "Leave-one-out over every closed-form method. The comparison the "
            "trained model has to beat.",
        ),
        Experiment(
            "mesh", "Mesh arm", "classical",
            "python -m ggssvt.cli mesh",
            reports / "mesh.json",
            "Marching cubes, then surface area and solidity as biomass features.",
        ),
        Experiment(
            "view_ablation", "View-count ablation", "classical",
            "python -m ggssvt.cli views",
            reports / "view_ablation.json",
            "Scores the reduced-view caches on quality and on physical "
            "plausibility.",
        ),
        Experiment(
            "dino_probe", "DINO frozen-feature probe", "classical",
            "python -m ggssvt.cli dino-probe",
            reports / "dino_probe.json",
            "Does a self-supervised ViT backbone carry more than the CNN stem, "
            "before any training.",
            needs_gpu=True,
        ),
        Experiment(
            "factorial", "SAM3D by DINO factorial", "classical",
            "python -m ggssvt.cli factorial",
            reports / "factorial.json",
            "Two by three over segmenter and backbone, through the frozen probe.",
            needs_gpu=True,
        ),
        Experiment(
            "gallery", "Reconstruction gallery", "check",
            "python -m ggssvt.cli gallery",
            work_dir / "reports" / "gallery" / "reconstructions.html",
            "Every specimen rendered. Catches what no metric does.",
        ),
        Experiment(
            "campaign", "Training campaign", "learned",
            "python -m ggssvt.campaign --plan core --device cuda "
            "--workers 8 --batch-size 2",
            work_dir / "campaign" / "summary.txt",
            "The seven runs that close H1, H2 and H3, plus the fused-cache "
            "comparison. Eight to ten hours.",
            needs_gpu=True,
        ),
        Experiment(
            "posefree", "Pose-free reconstruction", "check",
            "python -m ggssvt.cli posefree --methods fast3r dust3r mast3r "
            "--device cuda",
            reports / "posefree.json",
            "DUSt3R, MASt3R and Fast3R estimate cameras from images alone, so "
            "they are the independent check on a registration that has never "
            "been validated against a calibration target.",
            needs_gpu=True,
        ),
    )


def _headline(key: str, path: Path) -> tuple[str | None, dict]:
    """One number worth showing beside the entry, read from its artefact."""
    try:
        if path.suffix != ".json":
            return None, {}
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, {}

    if key == "baselines" and isinstance(data, dict):
        best = min(
            (v for v in data.values() if isinstance(v, dict) and "rmse_kg" in v),
            key=lambda v: v["rmse_kg"],
            default=None,
        )
        if best:
            return f"best RMSE {best['rmse_kg']:.3f} kg", {}
    if key == "fuse" and isinstance(data, dict):
        return f"{len(data)} specimens fused", {}
    if key == "view_ablation" and isinstance(data, list) and data:
        worst = min(data, key=lambda r: r.get("n_views", 99))
        return f"down to {worst.get('n_views')} views", {}
    if key == "factorial" and isinstance(data, dict):
        cells = data.get("cells", {})
        ran = sum(1 for v in cells.values() if isinstance(v, dict) and "rmse_kg" in v)
        return f"{ran} of {len(cells)} cells", {}
    return None, {}


def survey(work_dir: Path = WORK_DIR) -> list[ExperimentStatus]:
    """State of every experiment in the programme."""
    cache_stamp = (work_dir / "cache" / "quality.json")
    cache_mtime = cache_stamp.stat().st_mtime if cache_stamp.exists() else 0.0

    out: list[ExperimentStatus] = []
    for experiment in _programme(work_dir):
        exists = experiment.artefact.exists()
        ran_at = None
        stale = False
        headline = None

        if exists:
            mtime = experiment.artefact.stat().st_mtime
            ran_at = datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%d")
            # Derived results must not predate the cache they were fitted to.
            stale = experiment.stage != "data" and mtime < cache_mtime
            headline, _ = _headline(experiment.key, experiment.artefact)

        out.append(
            ExperimentStatus(
                key=experiment.key,
                label=experiment.label,
                stage=experiment.stage,
                command=experiment.command,
                why=experiment.why,
                needs_gpu=experiment.needs_gpu,
                done=exists,
                stale=stale,
                ran_at=ran_at,
                headline=headline,
            )
        )
    return out


def summarise(statuses: list[ExperimentStatus]) -> dict:
    """Counts for the page header."""
    return {
        "total": len(statuses),
        "done": sum(1 for s in statuses if s.done),
        "stale": sum(1 for s in statuses if s.stale),
        "pending_gpu": sum(1 for s in statuses if not s.done and s.needs_gpu),
    }


__all__ = ["Experiment", "ExperimentStatus", "summarise", "survey"]
