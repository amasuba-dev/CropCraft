"""Run the whole training programme with one command.

Designed to be started and left. Every run writes its own JSON result and a
completion marker, so a campaign that dies at 3am can be restarted and will skip
what already finished rather than starting over. Failures are recorded and the
campaign continues -- one OOM on the largest condition should not cost the other
eleven runs.

    python -m ggssvt.campaign --plan full --device cuda

The plans, smallest first:

``smoke``
    Two tiny runs. Confirms the loop, the checkpointing and the result writing
    work end to end before committing a night to it. **Always run this first.**
``core``
    The runs that answer a pending hypothesis: geometry-grounding ablation (H2),
    the CNN/DINO backbone comparison (H1), and the Fourier band sweep (H3).
``full``
    ``core`` plus the SAM3D arm and the 2x3 factorial.

Ordering is deliberate: within a plan, runs are ordered so that the ones
answering a hypothesis come first. If the campaign is cut short, what completes
is what the write-up needs.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from .config import MODEL, TRAIN, WORK_DIR

CAMPAIGN_DIR = WORK_DIR / "campaign"


@dataclass
class Run:
    """One training run: a pretrain, a leave-one-out sweep, and its results."""

    name: str
    question: str                     # which hypothesis or question it serves
    cache: str = "geometric"          # geometric | sam3d
    backbone: str = "cnn"
    variant: str = "base"
    geometry_grounded: bool = True
    fourier_bands: int = MODEL.fourier_bands
    fourier_max_freq: float = MODEL.fourier_max_freq
    pretrain_epochs: int = 120
    finetune_epochs: int = 60
    tokens_per_view: int = 64
    max_specimens: int | None = None   # smoke runs only; None uses every usable specimen

    def cache_dir(self) -> Path:
        return WORK_DIR / ("cache_sam3d" if self.cache == "sam3d" else "cache")

    def model_config(self):
        return dataclasses.replace(
            MODEL,
            backbone=self.backbone,
            backbone_variant=self.variant,
            fourier_bands=self.fourier_bands,
            fourier_max_freq=self.fourier_max_freq,
        )

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def _core_runs() -> list[Run]:
    """Runs that each close out a pending hypothesis."""
    return [
        Run(
            name="baseline_cnn",
            question="reference condition for every comparison below",
        ),
        Run(
            name="h2_no_geometry",
            question="H2: does geometry grounding help? Ablation vs baseline_cnn",
            geometry_grounded=False,
        ),
        Run(
            name="h1_dinov2",
            question="H1: does a self-supervised ViT backbone beat the CNN stem?",
            backbone="dinov2",
        ),
        # H3: the encoding reaches 83 cycles/m; the 12 mm grid tops out at 42.
        # If the top octaves carry nothing, cutting them should cost nothing --
        # which is the parameter-efficiency claim stated as an experiment.
        Run(
            name="h3_bands_6_freq6",
            question="H3: encoding trimmed to roughly the grid Nyquist",
            fourier_bands=6,
            fourier_max_freq=6.0,
        ),
        Run(
            name="h3_bands_4_freq4",
            question="H3: encoding well below the grid Nyquist; expected to hurt",
            fourier_bands=4,
            fourier_max_freq=4.0,
        ),
        Run(
            name="h3_bands_16_freq10",
            question="H3: encoding well above the grid Nyquist; expected to add nothing",
            fourier_bands=16,
            fourier_max_freq=10.0,
        ),
    ]


def _full_runs() -> list[Run]:
    runs = _core_runs()
    runs += [
        Run(
            name="sam3d_cnn",
            question="factorial: SAM3D segmentation, no DINO",
            cache="sam3d",
        ),
        Run(
            name="sam3d_dinov2",
            question="factorial: SAM3D and DINO together; tests the probe's interaction",
            cache="sam3d",
            backbone="dinov2",
        ),
        Run(
            name="h1_dinov3",
            question="H1: DINOv3, if access has been granted",
            backbone="dinov3",
        ),
    ]
    return runs


def _smoke_runs() -> list[Run]:
    return [
        Run(
            name="smoke_cnn",
            question="plumbing check only; results are meaningless",
            pretrain_epochs=2,
            finetune_epochs=2,
            tokens_per_view=16,
            max_specimens=4,
        ),
        Run(
            name="smoke_dinov2",
            question="plumbing check for the DINO path",
            backbone="dinov2",
            variant="small",
            pretrain_epochs=2,
            finetune_epochs=2,
            tokens_per_view=16,
            max_specimens=4,
        ),
    ]


PLANS = {"smoke": _smoke_runs, "core": _core_runs, "full": _full_runs}


@dataclass
class RunResult:
    """Outcome of one run."""

    name: str
    question: str
    status: str                       # done | failed | skipped
    seconds: float = 0.0
    metrics: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    error: str = ""


def _result_path(name: str, out_dir: Path) -> Path:
    return out_dir / f"{name}.json"


def execute(
    run: Run,
    *,
    device,
    out_dir: Path,
    train_config,
    verbose: bool = True,
) -> RunResult:
    """Pretrain, cross-validate and score one run."""
    import numpy as np
    import torch

    from .data.preprocess import usable_plant_ids
    from .eval.metrics import bootstrap_interval, regression_metrics
    from .models.backbones import backbone_is_available
    from .models.ggssvt import GGSSVT
    from .training.dataset import SpecimenDataset
    from .training.trainer import loocv, train_stage

    started = time.time()
    result = RunResult(
        name=run.name, question=run.question, status="failed", config=run.as_dict()
    )

    cache_dir = run.cache_dir()
    if not (cache_dir / "quality.json").exists():
        result.status = "skipped"
        result.error = (
            f"no cache at {cache_dir}. Build it: python -m ggssvt.cli preprocess"
            + (
                f" --segmenter sam3d --cache-dir {cache_dir}"
                if run.cache == "sam3d"
                else ""
            )
        )
        return result

    if run.backbone != "cnn":
        available, reason = backbone_is_available(run.backbone, run.variant)
        if not available:
            result.status = "skipped"
            result.error = reason.splitlines()[0]
            return result

    try:
        plant_ids = usable_plant_ids(cache_dir)
        if run.max_specimens is not None:
            # A smoke run proves the plumbing, so it trims the specimen list rather
            # than the epoch count alone: leave-one-out over all 28 costs hours even
            # at two epochs, and a check nobody waits for is a check nobody runs.
            plant_ids = plant_ids[: run.max_specimens]
        config = run.model_config()

        model = GGSSVT(
            config=config,
            tokens_per_view=run.tokens_per_view,
            geometry_grounded=run.geometry_grounded,
        )
        if verbose:
            print(
                f"  {model.n_parameters(False) / 1e6:.1f}M params "
                f"({model.n_parameters(True) / 1e6:.1f}M trainable), "
                f"{len(plant_ids)} specimens, cache={run.cache}"
            )

        train_stage(
            model,
            SpecimenDataset(plant_ids, cache_dir=cache_dir, mode="occupancy"),
            stage="pretrain",
            epochs=run.pretrain_epochs,
            config=train_config,
            device=device,
            log_every=max(1, run.pretrain_epochs // 4),
            verbose=verbose,
        )

        checkpoint = out_dir / f"{run.name}.pt"
        torch.save({"state_dict": model.state_dict(), "run": run.as_dict()}, checkpoint)

        folds = loocv(
            plant_ids,
            cache_dir=cache_dir,
            model_config=config,
            train_config=train_config,
            tokens_per_view=run.tokens_per_view,
            geometry_grounded=run.geometry_grounded,
            pretrained_state=model.state_dict(),
            device=device,
            verbose=False,
        )

        predicted = np.array([f.predicted_kg for f in folds])
        target = np.array([f.target_kg for f in folds])
        metrics = regression_metrics(predicted, target)
        low, high = bootstrap_interval(predicted, target)

        result.metrics = {
            **metrics.as_dict(),
            "rmse_ci": [low, high],
            "occupancy_ap": float(np.mean([f.occupancy_ap for f in folds])),
            "occupancy_best_iou": float(np.mean([f.occupancy_best_iou for f in folds])),
            "n_parameters": model.n_parameters(False),
            "n_trainable": model.n_parameters(True),
            "checkpoint": str(checkpoint),
            "predictions": predicted.tolist(),
            "plant_ids": list(plant_ids),
        }
        result.status = "done"

        if verbose:
            print(f"  {metrics}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    except Exception as exc:  # keep the campaign alive
        result.error = f"{type(exc).__name__}: {exc}"
        result.metrics["traceback"] = traceback.format_exc()[-1500:]
        if verbose:
            print(f"  FAILED: {result.error}")

    result.seconds = time.time() - started
    return result


def run_campaign(
    plan: str = "core",
    *,
    device=None,
    out_dir: Path = CAMPAIGN_DIR,
    workers: int = 8,
    batch_size: int = 2,
    only: list[str] | None = None,
    force: bool = False,
    verbose: bool = True,
) -> list[RunResult]:
    """Execute a plan, skipping runs that already have results."""
    from .training.trainer import resolve_device

    if plan not in PLANS:
        raise ValueError(f"unknown plan {plan!r}; expected one of {sorted(PLANS)}")

    device = device or resolve_device(TRAIN.device)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_config = dataclasses.replace(
        TRAIN, num_workers=workers, batch_size=batch_size
    )

    runs = PLANS[plan]()
    if only:
        runs = [r for r in runs if r.name in only]

    results: list[RunResult] = []
    campaign_started = time.time()

    for index, run in enumerate(runs, start=1):
        path = _result_path(run.name, out_dir)

        if path.exists() and not force:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("status") == "done":
                if verbose:
                    print(f"[{index}/{len(runs)}] {run.name}: already done, skipping")
                results.append(RunResult(**existing))
                continue

        if verbose:
            print(f"\n[{index}/{len(runs)}] {run.name}")
            print(f"  {run.question}")

        config = dataclasses.replace(
            train_config, finetune_epochs=run.finetune_epochs
        )
        result = execute(
            run, device=device, out_dir=out_dir, train_config=config, verbose=verbose
        )
        results.append(result)

        path.write_text(json.dumps(dataclasses.asdict(result), indent=2), encoding="utf-8")

        elapsed = time.time() - campaign_started
        if verbose:
            print(
                f"  {result.status} in {result.seconds / 60:.0f} min "
                f"(campaign {elapsed / 3600:.1f} h)"
            )

    _write_summary(results, out_dir)
    return results


def _write_summary(results: list[RunResult], out_dir: Path) -> None:
    """A single table of everything, for pasting into the write-up."""
    lines = [
        f"{'run':22s} {'status':8s} {'RMSE':>7s} {'R2':>7s} {'occ AP':>7s} {'min':>6s}",
        "-" * 62,
    ]
    for result in results:
        m = result.metrics
        if result.status == "done":
            lines.append(
                f"{result.name:22s} {result.status:8s} "
                f"{m.get('rmse_kg', float('nan')):7.3f} "
                f"{m.get('r2', float('nan')):7.3f} "
                f"{m.get('occupancy_ap', float('nan')):7.3f} "
                f"{result.seconds / 60:6.0f}"
            )
        else:
            lines.append(
                f"{result.name:22s} {result.status:8s} {result.error[:38]}"
            )

    text = "\n".join(lines)
    (out_dir / "summary.txt").write_text(text + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps([dataclasses.asdict(r) for r in results], indent=2), encoding="utf-8"
    )
    print("\n" + text)
    print(f"\nWrote {out_dir / 'summary.txt'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ggssvt.campaign",
        description="Run the full training programme unattended",
    )
    parser.add_argument("--plan", default="core", choices=sorted(PLANS))
    parser.add_argument("--device", default=TRAIN.device)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=CAMPAIGN_DIR)
    parser.add_argument("--only", nargs="*", help="run only these named runs")
    parser.add_argument(
        "--force", action="store_true", help="re-run even if results exist"
    )
    parser.add_argument(
        "--list", action="store_true", help="print the plan and exit"
    )
    args = parser.parse_args(argv)

    if args.list:
        for run in PLANS[args.plan]():
            print(f"{run.name:22s} {run.question}")
            print(
                f"{'':22s} cache={run.cache} backbone={run.backbone} "
                f"geometry={run.geometry_grounded} "
                f"bands={run.fourier_bands}@2^{run.fourier_max_freq:.0f} "
                f"epochs={run.pretrain_epochs}/{run.finetune_epochs}"
            )
        return 0

    from .training.trainer import resolve_device

    device = resolve_device(args.device)
    if device.type != "cuda":
        print(
            "WARNING: no CUDA device. This plan is many hours on a GPU and "
            "impractical on a CPU.",
        )

    run_campaign(
        args.plan,
        device=device,
        out_dir=args.out_dir,
        workers=args.workers,
        batch_size=args.batch_size,
        only=args.only,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
