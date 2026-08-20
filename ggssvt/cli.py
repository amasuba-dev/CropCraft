"""Command-line interface.

    python -m ggssvt.cli inspect        # dataset audit, no computation
    python -m ggssvt.cli preprocess     # register, segment, carve, cache
    python -m ggssvt.cli baselines      # LOOCV baselines from the cache
    python -m ggssvt.cli pretrain       # stage 1, self-supervised
    python -m ggssvt.cli loocv          # stage 2 + leave-one-out evaluation
    python -m ggssvt.cli report         # write the result tables and figures
    python -m ggssvt.cli visualise      # rig overlays and mask overlays
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .config import MODEL, TRAIN, WORK_DIR


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=WORK_DIR / "cache",
        help="where preprocessed specimens live",
    )


def cmd_inspect(args: argparse.Namespace) -> int:
    """Audit the raw dataset without running any geometry."""
    from .data import dataset_summary, load_dataset

    specimens = load_dataset(require_ground_truth=False, min_views=0, exclude=())
    print(json.dumps(dataset_summary(specimens), indent=2))

    print("\nPer-specimen:")
    problems = 0
    for specimen in specimens:
        flags = []
        if specimen.n_views != 12:
            flags.append(f"{specimen.n_views} views")
        if specimen.ground_truth is None:
            flags.append("no ground truth")
        if specimen.warnings:
            flags.extend(specimen.warnings)
        if flags:
            problems += 1
            print(f"  {specimen.plant_id}: " + "; ".join(flags))

    if problems == 0:
        print("  (no issues)")

    print("\nCalibration:")
    from .config import CALIB_DIR

    intrinsics = sorted(CALIB_DIR.glob("intrinsics/*_intrinsics.json"))
    extrinsics = sorted(CALIB_DIR.glob("extrinsics/*/rig_positions.json"))
    print(f"  intrinsics files: {len(intrinsics)}")
    print(f"  per-day rig_positions files: {len(extrinsics)}")
    if not intrinsics and not extrinsics:
        print(
            "  none found. Extrinsics are being estimated from the depth data\n"
            "  instead (ggssvt.geometry.rig). Capturing the ChArUco sequence\n"
            "  described in dataset/README.md would replace that estimate with a\n"
            "  measurement."
        )
    return 0


def cmd_preprocess(args: argparse.Namespace) -> int:
    from .data.preprocess import preprocess_dataset

    report = preprocess_dataset(
        cache_dir=args.cache_dir, plant_ids=args.plants, seed=args.seed
    )
    usable = sum(1 for q in report if q.is_usable())
    print(f"\n{usable}/{len(report)} specimens passed the quality gate.")
    print(f"Quality report: {args.cache_dir / 'quality.json'}")
    return 0


def cmd_baselines(args: argparse.Namespace) -> int:
    from .data.preprocess import usable_plant_ids
    from .eval.baselines import evaluate_baselines, load_features
    from .eval.metrics import bootstrap_interval

    plant_ids = args.plants or usable_plant_ids(args.cache_dir)
    features = load_features(plant_ids, args.cache_dir)
    targets = np.array([f.target_kg for f in features])

    print(f"Leave-one-out over {len(features)} specimens\n")
    for name, (metrics, predictions) in evaluate_baselines(features).items():
        low, high = bootstrap_interval(predictions, targets)
        print(f"  {name:20s} {metrics}")
        print(f"  {'':20s} RMSE 95% CI [{low:.3f}, {high:.3f}]")
    return 0


def cmd_pretrain(args: argparse.Namespace) -> int:
    from .data.preprocess import usable_plant_ids
    from .models.ggssvt import GGSSVT
    from .training.dataset import SpecimenDataset
    from .training.trainer import resolve_device, train_stage

    plant_ids = args.plants or usable_plant_ids(args.cache_dir)
    dataset = SpecimenDataset(plant_ids, cache_dir=args.cache_dir, mode="occupancy")

    model = GGSSVT(
        tokens_per_view=args.tokens_per_view,
        geometry_grounded=not args.no_geometry,
    )
    device = resolve_device(args.device)
    print(
        f"Pretraining on {len(dataset)} specimens, {model.n_parameters() / 1e6:.2f}M "
        f"parameters, device {device}"
    )

    run = train_stage(
        model,
        dataset,
        stage="pretrain",
        epochs=args.epochs,
        device=device,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    from . import __version__
    import torch

    torch.save(
        {
            "version": __version__,
            "state_dict": model.state_dict(),
            "model_config": MODEL.__dict__,
            "tokens_per_view": args.tokens_per_view,
            "geometry_grounded": not args.no_geometry,
            "plant_ids": plant_ids,
        },
        args.out,
    )
    (args.out.with_suffix(".log.json")).write_text(run.to_json(), encoding="utf-8")
    print(f"Saved checkpoint to {args.out}")
    return 0


def cmd_loocv(args: argparse.Namespace) -> int:
    import torch

    from .data.preprocess import usable_plant_ids
    from .eval.metrics import bootstrap_interval, regression_metrics
    from .training.trainer import loocv, resolve_device

    plant_ids = args.plants or usable_plant_ids(args.cache_dir)

    state = None
    if args.checkpoint and args.checkpoint.exists():
        state = torch.load(args.checkpoint, map_location="cpu")["state_dict"]
        print(f"Starting each fold from {args.checkpoint}")
    elif not args.strict:
        print(
            "No pretrained checkpoint given and --strict not set: every fold will "
            "start from random initialisation, which is not the intended protocol.",
            file=sys.stderr,
        )

    results = loocv(
        plant_ids,
        cache_dir=args.cache_dir,
        tokens_per_view=args.tokens_per_view,
        geometry_grounded=not args.no_geometry,
        strict=args.strict,
        pretrained_state=state,
        device=resolve_device(args.device),
    )

    predicted = np.array([r.predicted_kg for r in results])
    target = np.array([r.target_kg for r in results])
    metrics = regression_metrics(predicted, target)
    low, high = bootstrap_interval(predicted, target)

    print(f"\nGG-SSVT leave-one-out: {metrics}")
    print(f"  RMSE 95% CI [{low:.3f}, {high:.3f}]")
    print(
        f"  occupancy: IoU@0.5 {np.mean([r.occupancy_iou for r in results]):.3f}  "
        f"best-threshold IoU {np.mean([r.occupancy_best_iou for r in results]):.3f}  "
        f"AP {np.mean([r.occupancy_ap for r in results]):.3f}"
    )
    print(
        f"  protocol: {'strict (inductive)' if args.strict else 'transductive pretraining'}"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "strict": args.strict,
                "geometry_grounded": not args.no_geometry,
                "metrics": metrics.as_dict(),
                "folds": [r.__dict__ for r in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved fold results to {args.out}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .eval.report import write_report

    model_predictions = None
    if args.folds and args.folds.exists():
        payload = json.loads(args.folds.read_text(encoding="utf-8"))
        model_predictions = {
            fold["held_out"]: fold["predicted_kg"] for fold in payload["folds"]
        }
        print(f"Including GG-SSVT predictions from {args.folds}")

    path = write_report(args.plants, model_predictions=model_predictions)
    print(f"Wrote {path}")
    print(f"  and {path.parent / 'comparison.csv'}, {path.parent / 'scatter.svg'}")
    return 0


def cmd_visualise(args: argparse.Namespace) -> int:
    from .eval.visualise import write_overlays

    for plant_id in args.plants or ["E002", "M001"]:
        paths = write_overlays(plant_id, out_dir=args.out_dir)
        for path in paths:
            print(f"Wrote {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ggssvt",
        description="Geometry-grounded self-supervised volumetric transformer",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="audit the raw dataset")
    inspect.set_defaults(func=cmd_inspect)

    preprocess = sub.add_parser(
        "preprocess", help="register, segment and carve every specimen"
    )
    _add_common(preprocess)
    preprocess.add_argument("--plants", nargs="*", help="specific plant ids")
    preprocess.add_argument("--seed", type=int, default=0)
    preprocess.set_defaults(func=cmd_preprocess)

    baselines = sub.add_parser("baselines", help="LOOCV baselines from the cache")
    _add_common(baselines)
    baselines.add_argument("--plants", nargs="*")
    baselines.set_defaults(func=cmd_baselines)

    pretrain = sub.add_parser("pretrain", help="stage 1, self-supervised")
    _add_common(pretrain)
    pretrain.add_argument("--plants", nargs="*")
    pretrain.add_argument("--epochs", type=int, default=TRAIN.pretrain_epochs)
    pretrain.add_argument("--device", default=TRAIN.device)
    pretrain.add_argument("--tokens-per-view", type=int, default=64)
    pretrain.add_argument(
        "--no-geometry",
        action="store_true",
        help="ablation: disable Fourier grounding and the distance bias",
    )
    pretrain.add_argument(
        "--out", type=Path, default=WORK_DIR / "checkpoints" / "pretrain.pt"
    )
    pretrain.set_defaults(func=cmd_pretrain)

    evaluate = sub.add_parser("loocv", help="stage 2 and leave-one-out evaluation")
    _add_common(evaluate)
    evaluate.add_argument("--plants", nargs="*")
    evaluate.add_argument("--device", default=TRAIN.device)
    evaluate.add_argument("--tokens-per-view", type=int, default=64)
    evaluate.add_argument("--no-geometry", action="store_true")
    evaluate.add_argument(
        "--checkpoint", type=Path, default=WORK_DIR / "checkpoints" / "pretrain.pt"
    )
    evaluate.add_argument(
        "--strict",
        action="store_true",
        help="re-run pretraining inside each fold for an inductive result",
    )
    evaluate.add_argument("--out", type=Path, default=WORK_DIR / "reports" / "folds.json")
    evaluate.set_defaults(func=cmd_loocv)

    report = sub.add_parser("report", help="write result tables and figures")
    report.add_argument("--plants", nargs="*")
    report.add_argument(
        "--folds",
        type=Path,
        default=WORK_DIR / "reports" / "folds.json",
        help="GG-SSVT fold results to include in the comparison",
    )
    report.set_defaults(func=cmd_report)

    visualise = sub.add_parser("visualise", help="rig and segmentation overlays")
    visualise.add_argument("--plants", nargs="*")
    visualise.add_argument(
        "--out-dir", type=Path, default=WORK_DIR / "reports" / "overlays"
    )
    visualise.set_defaults(func=cmd_visualise)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
