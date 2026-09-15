"""Create presentation figures from existing reports and derived outputs.

These figures are evidence visualisations, not new benchmark runs.  They make
logged or machine-readable results usable in slides while retaining provenance.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "work_dirs" / "ggssvt" / "reports"
ABVT = ROOT / "work_dirs" / "ggssvt" / "abvt3r"
OUT = REPORTS / "presentation_evidence"


def _load_ply_vertices(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="utf-8").splitlines()
    end = lines.index("end_header")
    count = next(int(line.split()[-1]) for line in lines[:end] if line.startswith("element vertex"))
    return np.asarray([[float(v) for v in line.split()[:3]] for line in lines[end + 1:end + 1 + count]])


def _contact_sheet(plant_ids: list[str]) -> None:
    fig, axes = plt.subplots(len(plant_ids), 4, figsize=(12, 2.5 * len(plant_ids)), squeeze=False)
    for row, plant in enumerate(plant_ids):
        masks = sorted(ABVT.glob(f"{plant}_*_mask.png"))
        for col, path in enumerate(masks[::max(1, len(masks) // 4)][:4]):
            axes[row, col].imshow(Image.open(path), cmap="gray", vmin=0, vmax=255)
            axes[row, col].set_title(path.stem.replace(f"{plant}_", "").replace("_mask", ""))
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(plant, rotation=0, labelpad=30, va="center", fontweight="bold")
    fig.suptitle("ABVT3R geometric segmentation masks (0 = excluded, 255 = retained)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "abvt3r_segmentation_contact_sheet.png", dpi=180)
    plt.close(fig)


def _pot_exclusion(plant: str) -> None:
    vertices = _load_ply_vertices(ABVT / f"{plant}_mesh_vertices.ply")
    threshold = 0.28
    above = vertices[:, 2] > threshold
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(vertices[:, 0], vertices[:, 2], s=0.6, c=np.where(above, "#159a80", "#8a5a3b"), alpha=0.5)
    axes[0].axhline(threshold, color="#d85b35", linestyle="--", label="pot exclusion threshold: 0.28 m")
    axes[0].set_title(f"{plant}: side projection")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("z (m)")
    axes[0].legend(fontsize=8)
    axes[1].scatter(vertices[:, 0], vertices[:, 1], s=0.6, c=np.where(above, "#159a80", "#8a5a3b"), alpha=0.5)
    axes[1].set_title("Top projection: retained vs excluded mesh vertices")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("y (m)")
    fig.suptitle("ABVT3R pot/shoot geometric exclusion", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / f"{plant}_pot_exclusion.png", dpi=180)
    plt.close(fig)


def _biomass_chart() -> None:
    values = {
        "Fused geometry": (0.430, 0.416),
        "DINOv2": (0.400, 0.483),
        "Fast3R": (0.523, 0.139),
        "ABVT3R RF": (0.402, 0.451),
        "ABVT3R ANN": (0.509, 0.118),
    }
    labels = list(values)
    rmse = [values[x][0] for x in labels]
    r2 = [values[x][1] for x in labels]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, rmse, color=["#0e9384", "#7b3fa8", "#2563c9", "#c4622d", "#8e4b9d"])
    ax.set_ylabel("LOOCV RMSE (kg)")
    ax.set_title("Biomass comparison: validated and external-baseline probes")
    ax.grid(axis="y", alpha=0.2)
    for bar, score in zip(bars, rmse):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 0.01, f"{score:.3f}", ha="center", fontsize=9)
    ax2 = ax.twinx()
    ax2.plot(labels, r2, color="#222222", marker="o", linewidth=2, label="R²")
    ax2.set_ylabel("R²")
    ax2.axhline(0, color="#777777", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "biomass_comparison.png", dpi=180)
    plt.close(fig)


def _status_figure() -> None:
    rows = [
        ("ABVT3R", "reconstruction + masks", "complete", "#16805a"),
        ("DUSt3R/MASt3R/Fast3R", "aligned clouds", "complete", "#16805a"),
        ("GG-SSVT", "campaign + occupancy", "partial", "#c58524"),
        ("VGGT", "confidence-filtered clouds", "incomplete/OOM", "#b84b43"),
        ("Nerfstudio", "neural-field scoring", "not run", "#777777"),
        ("Pheno4D", "ground-truth geometry", "not scored", "#777777"),
    ]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, (name, output, status, colour) in enumerate(rows):
        ax.barh(i, 1, color=colour, alpha=0.85)
        ax.text(0.02, i, f"{name} — {output}", va="center", color="white", fontweight="bold", fontsize=10)
        ax.text(1.02, i, status, va="center", fontsize=10)
    ax.set_xlim(0, 1.55)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title("Evidence availability for presentation")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "evidence_status.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _contact_sheet(["E001", "E010", "M001", "V001"])
    _pot_exclusion("M001")
    _biomass_chart()
    _status_figure()
    manifest = {
        "note": "Derived presentation figures from existing reports/logs; not new benchmark runs.",
        "figures": sorted(str(p.relative_to(REPORTS)) for p in OUT.glob("*.png")),
        "sources": [
            "work_dirs/ggssvt/abvt3r/*_report.json",
            "work_dirs/ggssvt/reports/metrics.json",
            "work_dirs/ggssvt/reports/pointcloud_biomass_retrained_20260913.json",
            "work_dirs/ggssvt/reports/vggt_valid_all_20260913.log",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
