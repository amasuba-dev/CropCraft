"""Command-line interface.

    python -m ggssvt.cli inspect        # dataset audit, no computation
    python -m ggssvt.cli access         # HuggingFace account and model access
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


def _add_training(parser: argparse.ArgumentParser) -> None:
    """Flags that decide whether a run finishes today or next week."""
    parser.add_argument(
        "--workers",
        type=int,
        default=TRAIN.num_workers,
        help="dataloader workers; 4 keeps a GPU fed, 0 starves it",
    )
    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=TRAIN.finetune_epochs,
        help=(
            "epochs per leave-one-out fold. This multiplies by the number of "
            "folds, so the default is a multi-day run on one GPU"
        ),
    )
    parser.add_argument(
        "--fourier-bands",
        type=int,
        default=MODEL.fourier_bands,
        help=(
            "frequency bands in the positional encoding. The H3 parameter-"
            "efficiency knob: the encoding currently reaches 83 cycles/m while "
            "the 12 mm grid tops out at 42, so it is over-provisioned"
        ),
    )
    parser.add_argument(
        "--fourier-max-freq",
        type=float,
        default=MODEL.fourier_max_freq,
        help="top exponent of the frequency ladder (2^k). Pairs with --fourier-bands",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=TRAIN.batch_size,
        help=(
            "specimens per step, each carrying 12 views. 1 fits 8 GB; 2 or 4 "
            "roughly halves or quarters epoch time on a 16 GB card"
        ),
    )


def _model_config(args: argparse.Namespace):
    """Build a ModelConfig from the architecture flags."""
    import dataclasses

    return dataclasses.replace(
        MODEL,
        fourier_bands=getattr(args, "fourier_bands", MODEL.fourier_bands),
        fourier_max_freq=getattr(args, "fourier_max_freq", MODEL.fourier_max_freq),
    )


def _training_config(args: argparse.Namespace):
    """Build a TrainConfig from the common training flags."""
    import dataclasses

    return dataclasses.replace(
        TRAIN,
        num_workers=getattr(args, "workers", TRAIN.num_workers),
        finetune_epochs=getattr(args, "finetune_epochs", TRAIN.finetune_epochs),
        batch_size=getattr(args, "batch_size", TRAIN.batch_size),
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


def cmd_access(args: argparse.Namespace) -> int:
    """Report which account is authenticated and what it can actually download.

    Authentication is ambient: `hf auth login` writes a token that
    huggingface_hub and transformers pick up on their own, so nothing in this
    codebase takes a token argument and no script should contain one. What goes
    wrong is almost never the token itself -- it is being logged in as a
    different account than the one that was approved.
    """
    from .geometry.sam3d import SAM3D_OBJECTS_REPO, SAM3_REPO, SAM_REPOS
    from .models.backbones import DINOV2_REPOS, DINOV3_REPOS, repo_access

    try:
        from huggingface_hub import get_token, whoami
    except ImportError:
        print("huggingface_hub is not installed", file=sys.stderr)
        return 2

    token = get_token()
    if not token:
        print("No HuggingFace token found.")
        print("  Interactive:  hf auth login")
        print("  Headless:     export HF_TOKEN=hf_...")
        print("Gated models will be unavailable until one of those is set.")
    else:
        try:
            account = whoami().get("name", "?")
            print(f"Authenticated as: {account}")
        except Exception as exc:
            print(f"Token present but rejected: {str(exc)[:120]}", file=sys.stderr)
            return 1
        import os

        # HF_TOKEN wins over the cached login, so a stale environment variable
        # silently overrides `hf auth login` and you stay on the wrong account
        # no matter how many times you log in again.
        source = (
            "HF_TOKEN environment variable"
            if os.environ.get("HF_TOKEN")
            else "cached login (~/.cache/huggingface)"
        )
        print(f"Token source: {source}")

    checks: list[tuple[str, str]] = []
    for variant, repo in DINOV2_REPOS.items():
        checks.append((f"dinov2-{variant}", repo))
    for variant, repo in DINOV3_REPOS.items():
        checks.append((f"dinov3-{variant}", repo))
    for variant, repo in SAM_REPOS.items():
        checks.append((f"sam-{variant}", repo))
    checks.append(("sam3", SAM3_REPO))
    checks.append(("sam-3d-objects", SAM3D_OBJECTS_REPO))

    print()
    blocked = []
    for label, repo in checks:
        accessible, reason = repo_access(repo)
        print(f"  {'OK     ' if accessible else 'BLOCKED'}  {label:16s} {repo}")
        if not accessible:
            blocked.append((label, reason))

    if blocked:
        print(f"\n{len(blocked)} repositories are not accessible to this account.")
        print("If the HuggingFace settings page shows them ACCEPTED, the usual")
        print("cause is being logged in here as a different account than the one")
        print("that was approved. Check the name printed above.")
    else:
        print("\nEverything this project can use is accessible.")
    return 0


def cmd_preprocess(args: argparse.Namespace) -> int:
    from .data.preprocess import preprocess_dataset

    if args.segmenter == "sam3d" and args.cache_dir == WORK_DIR / "cache":
        print(
            "Refusing to overwrite the geometric cache with SAM3D output. "
            "Pass --cache-dir work_dirs/ggssvt/cache_sam3d so both conditions "
            "can be compared.",
            file=sys.stderr,
        )
        return 2

    report = preprocess_dataset(
        cache_dir=args.cache_dir,
        plant_ids=args.plants,
        seed=args.seed,
        segmenter=args.segmenter,
        sam_model=args.sam_model,
        sam_device=args.sam_device,
        n_views=args.views,
    )
    usable = sum(1 for q in report if q.is_usable())
    print(f"\n{usable}/{len(report)} specimens passed the quality gate.")
    print(f"Quality report: {args.cache_dir / 'quality.json'}")
    return 0


def cmd_views(args: argparse.Namespace) -> int:
    from .eval.view_ablation import format_table, run_ablation

    results = run_ablation(args.work_dir, verbose=True)
    if not results:
        print("\nNo view-count caches found. Build them first (see above).")
        return 1

    print()
    print(format_table(results))
    print(
        "\n`plausible` counts specimens whose reconstructed above-ground volume "
        "could\nphysically weigh the measured mass (200-1000 kg/m3). Agreement "
        "degrades gently\nas views are removed; plausibility does not, which is "
        "the point of reporting both."
    )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps([r.as_dict() for r in results], indent=2), encoding="utf-8"
        )
        print(f"\nSaved to {args.out}")
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
        config=_model_config(args),
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
        config=_training_config(args),
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
        model_config=_model_config(args),
        train_config=_training_config(args),
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


def _write_comparison(report, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.to_json(), encoding="utf-8")
    print(f"\nSaved comparison to {out}")


def cmd_dino_probe(args: argparse.Namespace) -> int:
    """Frozen-feature DINO vs no-DINO comparison. CPU, minutes."""
    from .data.preprocess import usable_plant_ids
    from .eval.experiment import probe_experiment

    plant_ids = args.plants or usable_plant_ids(args.cache_dir)
    print(f"Frozen-feature probe over {len(plant_ids)} specimens, leave-one-out\n")

    report = probe_experiment(
        plant_ids,
        backbones=tuple(args.backbones),
        variant=args.variant,
        cache_dir=args.cache_dir,
        n_components=args.components,
        alpha=args.alpha,
    )
    print()
    print(report.to_table())
    _write_comparison(report, args.out)
    return 0


def cmd_experiment(args: argparse.Namespace) -> int:
    """Full GG-SSVT backbone comparison. Needs a GPU."""
    from .data.preprocess import usable_plant_ids
    from .eval.experiment import backbone_experiment
    from .training.trainer import resolve_device

    plant_ids = args.plants or usable_plant_ids(args.cache_dir)
    device = resolve_device(args.device)

    if device.type != "cuda":
        print(
            "WARNING: no CUDA device. This trains one model per backbone and will "
            "take many hours on a CPU. Use `dino-probe` for a CPU-scale answer.",
            file=sys.stderr,
        )

    report = backbone_experiment(
        plant_ids,
        backbones=tuple(args.backbones),
        variant=args.variant,
        cache_dir=args.cache_dir,
        train_config=_training_config(args),
        epochs=args.epochs,
        tokens_per_view=args.tokens_per_view,
        out_dir=args.out_dir,
        device=device,
    )
    print()
    print(report.to_table())
    _write_comparison(report, args.out_dir / "backbone_comparison.json")
    return 0


def cmd_factorial(args: argparse.Namespace) -> int:
    """SAM3D on/off crossed with the appearance backbone."""
    from .eval.factorial import probe_factorial, train_factorial

    if args.train:
        from .training.trainer import resolve_device

        device = resolve_device(args.device)
        if device.type != "cuda":
            print(
                "WARNING: no CUDA device. --train runs one pretrain plus one "
                "leave-one-out sweep per cell and will take many hours on a CPU. "
                "Drop --train for the CPU-scale probe.",
                file=sys.stderr,
            )
        report = train_factorial(
            segmenters=tuple(args.segmenters),
            backbones=tuple(args.backbones),
            variant=args.variant,
            train_config=_training_config(args),
            epochs=args.epochs,
            tokens_per_view=args.tokens_per_view,
            device=device,
        )
    else:
        report = probe_factorial(
            segmenters=tuple(args.segmenters),
            backbones=tuple(args.backbones),
            variant=args.variant,
            n_components=args.components,
            alpha=args.alpha,
        )
    print()
    print(report.to_table())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report.to_json(), encoding="utf-8")
    print(f"\nSaved factorial to {args.out}")
    return 0


def cmd_gallery(args: argparse.Namespace) -> int:
    """Render every reconstruction: contact sheets, PLY clouds, interactive page."""
    import json

    from .data.preprocess import usable_plant_ids
    from .eval.factorial import CACHE_DIRS
    from .eval.gallery_html import build_html
    from .eval.render import build_gallery

    caches = {
        name: path
        for name, path in CACHE_DIRS.items()
        if (path / "quality.json").exists()
    }
    if not caches:
        print("No preprocessed cache found. Run `preprocess` first.", file=sys.stderr)
        return 2

    plant_ids = args.plants or sorted(
        {pid for path in caches.values() for pid in usable_plant_ids(path)}
    )
    print(f"Rendering {len(plant_ids)} specimens across {len(caches)} segmenter(s)")

    manifest = build_gallery(
        plant_ids,
        cache_dirs=caches,
        out_dir=args.out_dir,
        write_ply=not args.no_ply,
        write_sheets=not args.no_sheets,
        columns=args.columns,
    )
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    page = build_html(manifest, args.out_dir / "reconstructions.html")
    print(f"  wrote {page}")
    if not args.no_ply:
        print(f"  wrote {len(manifest['files'])} files under {args.out_dir}")
    return 0


def cmd_nerfstudio(args: argparse.Namespace) -> int:
    """Export estimated rig poses as Nerfstudio transforms.json."""
    from .eval.nerfstudio_export import export_dataset, training_commands

    results = export_dataset(
        args.plants,
        in_place=not args.out_dir,
        out_root=args.out_dir or (WORK_DIR / "nerfstudio"),
        include_depth=not args.no_depth,
        write_rig_positions=not args.no_rig_positions,
    )
    if not results:
        print("No specimens exported.", file=sys.stderr)
        return 2

    first = sorted(results)[0]
    print(f"\nExported {len(results)} specimens.")
    print(
        "\nPoses are estimated from depth, not measured by ChArUco calibration.\n"
        "Train with a camera optimiser so the radiance field can refine them:\n"
    )
    for line in training_commands(first):
        print("  " + line)
    return 0


def cmd_mesh(args: argparse.Namespace) -> int:
    """Extract meshes and score biomass from them against every other method."""
    from .data.preprocess import load_cached, usable_plant_ids
    from .eval.baselines import load_features
    from .eval.mesh_baseline import evaluate_with_mesh
    from .eval.metrics import paired_bootstrap_difference
    from .geometry.mesh import mesh_from_occupancy

    plant_ids = args.plants or usable_plant_ids(args.cache_dir)
    print(f"Meshing {len(plant_ids)} specimens from {args.cache_dir}")

    results, table = evaluate_with_mesh(
        plant_ids, cache_dir=args.cache_dir, alpha=args.alpha
    )
    targets = np.array([f.target_kg for f in load_features(plant_ids, args.cache_dir)])

    header = f"\n{'method':28s} {'RMSE':>7s} {'MAE':>7s} {'MARE%':>7s} {'R2':>7s}"
    print(header)
    print("-" * (len(header) - 1))
    for name, (metrics, _) in sorted(results.items(), key=lambda kv: kv[1][0].rmse_kg):
        print(
            f"{name:28s} {metrics.rmse_kg:7.3f} {metrics.mae_kg:7.3f} "
            f"{metrics.mare * 100:7.1f} {metrics.r2:7.3f}"
        )

    reference = "geometric features"
    if reference in results:
        print(f"\nPaired bootstrap against '{reference}':")
        for name in ("mesh geometry", "canopy area allometric"):
            if name not in results:
                continue
            d = paired_bootstrap_difference(
                results[name][1], results[reference][1], targets
            )
            verdict = (
                "significant" if d["high"] < 0 or d["low"] > 0 else "not resolved"
            )
            print(
                f"  {name:26s} dRMSE {d['difference']:+.3f} kg  "
                f"95% CI [{d['low']:+.3f}, {d['high']:+.3f}]  {verdict}"
            )

    if args.export:
        args.export.mkdir(parents=True, exist_ok=True)
        for plant_id in plant_ids:
            cached = load_cached(plant_id, args.cache_dir)
            mesh = mesh_from_occupancy(
                cached.occupancy,
                voxel_size_m=cached.voxel_size_m,
                smoothing=args.smoothing,
            )
            (args.export / f"{plant_id}.obj").write_text(mesh.to_obj(), encoding="utf-8")
        print(f"\nWrote {len(plant_ids)} OBJ meshes to {args.export}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "n_specimens": len(plant_ids),
                "mesh_metrics": table,
                "results": {
                    name: {**m.as_dict(), "predictions": p.tolist()}
                    for name, (m, p) in results.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved to {args.out}")
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    """Build the self-contained research walkthrough page."""
    from .eval.dashboard_data import build_payload
    from .eval.site_html import build_site

    payload = build_payload()
    teaser = WORK_DIR / "reports" / "gallery" / "contact_sheet_geometric.png"
    index = build_site(
        payload.to_json(), payload.summary, out_dir=args.out, teaser=teaser
    )
    size = index.stat().st_size / 1e6
    print(f"Wrote {index} ({size:.2f} MB, {len(payload.specimens)} specimens)")
    if not teaser.exists():
        print(f"  no teaser image at {teaser}; run `cli gallery` to produce it")
    print("A GitHub Pages layout: index.html plus static/. Open index.html directly,")
    print("or push the whole directory to a *.github.io repository.")
    return 0


def cmd_posefree(args: argparse.Namespace) -> int:
    """Compare DUSt3R / MASt3R / Fast3R against the estimated rig."""
    from .eval.pose_free_experiment import run_experiment
    from .geometry.pose_free import available_backends

    if args.check_only:
        print("Pose-free backend availability:\n")
        for name, (ok, reason) in available_backends().items():
            print(f"  {'OK     ' if ok else 'MISSING'}  {name}")
            if reason:
                for line in reason.splitlines():
                    print(f"      {line}")
        return 0

    report = run_experiment(
        args.plants,
        methods=tuple(args.methods),
        device=args.device,
        image_size=args.image_size,
        cache_dir=args.cache_dir,
        out_path=args.out,
    )
    print()
    print(report.to_table())
    if args.out:
        print(f"\nSaved to {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ggssvt",
        description="Geometry-grounded self-supervised volumetric transformer",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="audit the raw dataset")
    inspect.set_defaults(func=cmd_inspect)

    access = sub.add_parser(
        "access", help="which HuggingFace account is active, and what it can download"
    )
    access.set_defaults(func=cmd_access)

    preprocess = sub.add_parser(
        "preprocess", help="register, segment and carve every specimen"
    )
    _add_common(preprocess)
    preprocess.add_argument("--plants", nargs="*", help="specific plant ids")
    preprocess.add_argument("--seed", type=int, default=0)
    preprocess.add_argument(
        "--segmenter",
        default="geometric",
        choices=["geometric", "sam3d"],
        help="subject segmentation; sam3d needs its own --cache-dir",
    )
    preprocess.add_argument(
        "--sam-model", default="base", help="SAM checkpoint: base | large | huge"
    )
    preprocess.add_argument("--sam-device", default="cpu")
    preprocess.add_argument(
        "--views", type=int, default=None,
        help=(
            "use an evenly spaced subset of the 12 views (must divide 12: "
            "2, 3, 4, 6). Needs its own --cache-dir"
        ),
    )
    preprocess.set_defaults(func=cmd_preprocess)

    baselines = sub.add_parser("baselines", help="LOOCV baselines from the cache")
    _add_common(baselines)
    baselines.add_argument("--plants", nargs="*")
    baselines.set_defaults(func=cmd_baselines)

    views = sub.add_parser(
        "views", help="view-count ablation across the per-view caches"
    )
    views.add_argument("--work-dir", type=Path, default=WORK_DIR)
    views.add_argument(
        "--out", type=Path, default=WORK_DIR / "reports" / "view_ablation.json"
    )
    views.set_defaults(func=cmd_views)

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
    _add_training(pretrain)
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
    _add_training(evaluate)
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

    probe = sub.add_parser(
        "dino-probe",
        help="frozen-feature DINO vs no-DINO comparison (CPU, minutes)",
    )
    _add_common(probe)
    probe.add_argument("--plants", nargs="*")
    probe.add_argument(
        "--backbones",
        nargs="+",
        default=["dinov2", "dinov3"],
        choices=["dinov2", "dinov3"],
        help="DINO backbones to probe; the no-DINO control always runs",
    )
    probe.add_argument("--variant", default="base", choices=["small", "base", "large"])
    probe.add_argument("--components", type=int, default=8, help="PCA components")
    probe.add_argument("--alpha", type=float, default=1.0, help="ridge strength")
    probe.add_argument(
        "--out", type=Path, default=WORK_DIR / "reports" / "dino_probe.json"
    )
    probe.set_defaults(func=cmd_dino_probe)

    experiment = sub.add_parser(
        "experiment", help="full GG-SSVT backbone comparison (needs a GPU)"
    )
    _add_common(experiment)
    experiment.add_argument("--plants", nargs="*")
    experiment.add_argument(
        "--backbones",
        nargs="+",
        default=["cnn", "dinov2", "dinov3"],
        choices=["cnn", "dinov2", "dinov3"],
    )
    experiment.add_argument(
        "--variant", default="base", choices=["small", "base", "large"]
    )
    experiment.add_argument("--epochs", type=int, default=TRAIN.pretrain_epochs)
    experiment.add_argument("--device", default=TRAIN.device)
    experiment.add_argument("--tokens-per-view", type=int, default=64)
    experiment.add_argument("--out-dir", type=Path, default=WORK_DIR / "experiments")
    _add_training(experiment)
    experiment.set_defaults(func=cmd_experiment)

    factorial = sub.add_parser(
        "factorial",
        help="full SAM3D x DINO factorial via the frozen-feature probe (CPU)",
    )
    factorial.add_argument(
        "--segmenters", nargs="+", default=["geometric", "sam3d"],
        choices=["geometric", "sam3d"],
    )
    factorial.add_argument(
        "--backbones", nargs="+", default=["cnn", "dinov2", "dinov3"],
        choices=["cnn", "dinov2", "dinov3"],
    )
    factorial.add_argument(
        "--variant", default="base", choices=["small", "base", "large"]
    )
    factorial.add_argument("--components", type=int, default=8)
    factorial.add_argument("--alpha", type=float, default=1.0)
    factorial.add_argument(
        "--train",
        action="store_true",
        help="train GG-SSVT per cell instead of probing frozen features (GPU)",
    )
    factorial.add_argument("--epochs", type=int, default=TRAIN.pretrain_epochs)
    factorial.add_argument("--device", default=TRAIN.device)
    factorial.add_argument("--tokens-per-view", type=int, default=64)
    factorial.add_argument(
        "--out", type=Path, default=WORK_DIR / "reports" / "factorial.json"
    )
    _add_training(factorial)
    factorial.set_defaults(func=cmd_factorial)

    gallery = sub.add_parser(
        "gallery", help="render every reconstruction for viewing"
    )
    gallery.add_argument("--plants", nargs="*")
    gallery.add_argument(
        "--out-dir", type=Path, default=WORK_DIR / "reports" / "gallery"
    )
    gallery.add_argument("--columns", type=int, default=4)
    gallery.add_argument("--no-ply", action="store_true", help="skip PLY export")
    gallery.add_argument(
        "--no-sheets", action="store_true", help="skip the PNG contact sheets"
    )
    gallery.set_defaults(func=cmd_gallery)

    mesh = sub.add_parser(
        "mesh", help="extract meshes and score biomass from them"
    )
    _add_common(mesh)
    mesh.add_argument("--plants", nargs="*")
    mesh.add_argument("--alpha", type=float, default=1.0, help="ridge strength")
    mesh.add_argument(
        "--smoothing", type=int, default=0,
        help="Laplacian smoothing passes on exported meshes (changes area/volume)",
    )
    mesh.add_argument(
        "--export", type=Path, default=None, help="also write OBJ meshes here"
    )
    mesh.add_argument(
        "--out", type=Path, default=WORK_DIR / "reports" / "mesh.json"
    )
    mesh.set_defaults(func=cmd_mesh)

    dashboard = sub.add_parser(
        "dashboard", help="build the research walkthrough page"
    )
    dashboard.add_argument(
        "--out",
        type=Path,
        default=WORK_DIR / "site",
        help="site directory to write (index.html plus static/)",
    )
    dashboard.set_defaults(func=cmd_dashboard)

    posefree = sub.add_parser(
        "posefree",
        help="DUSt3R / MASt3R / Fast3R poses vs the estimated rig (needs a GPU)",
    )
    _add_common(posefree)
    posefree.add_argument("--plants", nargs="*")
    posefree.add_argument(
        "--methods", nargs="+", default=["fast3r", "dust3r", "mast3r"],
        choices=["dust3r", "mast3r", "fast3r"],
    )
    posefree.add_argument("--device", default="cuda")
    posefree.add_argument("--image-size", type=int, default=512)
    posefree.add_argument(
        "--check-only", action="store_true",
        help="report which backends are installed and stop",
    )
    posefree.add_argument(
        "--out", type=Path, default=WORK_DIR / "reports" / "posefree.json"
    )
    posefree.set_defaults(func=cmd_posefree)

    nerf = sub.add_parser(
        "nerfstudio", help="export estimated poses as Nerfstudio transforms.json"
    )
    nerf.add_argument("--plants", nargs="*")
    nerf.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="write here instead of into each specimen directory",
    )
    nerf.add_argument("--no-depth", action="store_true", help="omit depth_file_path")
    nerf.add_argument(
        "--no-rig-positions",
        action="store_true",
        help="skip the rig_positions.json / intrinsics files make_transforms.py uses",
    )
    nerf.set_defaults(func=cmd_nerfstudio)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
