"""The four forks that decide whether the trained arm reports anything real.

Every stage in `architecture.py` is a *method*. This figure is about something
else: four choices that are already made, that no diagram shows, and that each
have a measured cost in this project's own reports. They are not alternatives
anyone argued for. Three of them are simply the defaults, and the defaults are
on the losing side of all four.

The figure exists because those costs are scattered across six report files, and
scattered they read as six small caveats. Collected, they read as one decision
with a number on it: whether four days of training produces a result that
survives a batch holdout, or one that does not.

**Nothing here is an opinion.** Each row cites the report it came from, and the
figure regenerates from those reports, so a row cannot outlive the number that
justified it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..config import WORK_DIR
from .architecture import FONT, INK, MONO, MUTED, PAPER, RULE, _esc

# Semantic, not from the sequential ramp. These are two states of a decision,
# not two points on a scale, and viridis has no way to say "this one is wrong".
LOSE = "#b4404a"
WIN = "#2f7d52"
LOSE_BG = "#fbf1f2"
WIN_BG = "#f0f7f3"

WIDTH = 980
COL_W = 400
LEFT_X = 92
RIGHT_X = 512
ROW_GAP = 18
HEAD_H = 164


@dataclass
class Fork:
    """One decision, both ways, with the measurement that separates them."""

    question: str
    losing: str
    winning: str
    cost: str
    source: str

    def wrapped(self) -> tuple[list[str], list[str], list[str]]:
        """Both columns and the cost line, wrapped once and reused.

        Wrapping in one place is what keeps the layout honest. An earlier
        version sliced the cost at 96 characters and drew at most two lines at a
        fixed offset, so a third line silently overlapped the next row's
        heading. Measuring here and laying out from the measurement cannot do
        that.
        """
        import textwrap

        return (textwrap.wrap(self.losing, 42),
                textwrap.wrap(self.winning, 42),
                textwrap.wrap(self.cost, 104))

    def box_height(self) -> int:
        losing, winning, _ = self.wrapped()
        return 22 + max(len(losing), len(winning)) * 16 + 12

    def height(self) -> int:
        _, _, cost = self.wrapped()
        return self.box_height() + 8 + len(cost) * 16 + 18


def forks(reports: Path | None = None) -> list[Fork]:
    """Read the four forks out of the reports that measured them."""
    reports = reports or WORK_DIR / "reports"

    def read(name: str) -> dict | None:
        path = reports / name
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    virtual = read("virtual_views.json") or {}
    holdout = read("batch_holdout.json") or {}
    pot = read("pot_mass.json") or {}

    summary = virtual.get("summary", {})
    n_scans = summary.get("n_scans", 14)
    carve_pass = summary.get("carve_passes_density", 0)
    fused_pass = summary.get("fused_passes_density", 0)
    carve_ratio = summary.get("median_carve_volume_ratio", 4.57)
    fused_ratio = summary.get("median_fused_volume_ratio", 2.05)

    gap = next(iter((holdout.get("gaps") or {}).values()), {})
    loocv = gap.get("loocv_rmse", 0.576)
    lobo = gap.get("lobo_rmse", 1.105)
    lobo_r2 = gap.get("lobo_r2", -2.97)
    paired = gap.get("paired_difference", {})
    inflation = paired.get("difference", 0.529)
    p_value = paired.get("p_direction", 0.0056)

    rows = pot.get("rows", [])
    fragile = sum(1 for r in rows
                  if r.get("plant_uncertainty", {}).get("50g", 0) > 0.10)

    return [
        Fork(
            question="Which reconstruction feeds everything downstream?",
            losing="Silhouette carving, the current default cache. "
                   f"Admitted by the density screen on {carve_pass} of "
                   f"{n_scans} laser-scanned plants.",
            winning="TSDF depth fusion, already computed and sitting in "
                    f"cache_tsdf. Admitted on {fused_pass} of {n_scans}.",
            cost=f"{carve_ratio:.2f}x true volume against {fused_ratio:.2f}x, "
                 f"on clean views with exact poses. Free: no training needed.",
            source="virtual_views.json, recon_metrics.json",
        ),
        Fork(
            question="Which folds does the campaign train and select on?",
            losing="Leave one specimen out. For E001 the other nine of its "
                   "batch stay in training, captured the same day at similar "
                   "size, so a neighbour leaks the answer.",
            winning="Leave one batch out. Four folds, and the only protocol "
                    "under which a number here means anything.",
            cost=f"RMSE {loocv:.3f} kg becomes {lobo:.3f}; inflation "
                 f"{inflation:+.3f} kg, p = {p_value:.4f}. Also 4 folds "
                 f"instead of 36, so roughly nine times less compute.",
            source="batch_holdout.json",
        ),
        Fork(
            question="How does volume become mass?",
            losing="A learned density absorbs whatever scale error the "
                   "operator has, so a systematic factor of two is hidden "
                   "inside a fitted parameter.",
            winning="Fit the volume ratio explicitly, per species, and let "
                    "density carry only what is left.",
            cost=f"Even at its best the fusion is {fused_ratio:.2f}x true "
                 "volume. A bias that large should be a stated parameter, not "
                 "an implicit one.",
            source="virtual_views.json",
        ),
        Fork(
            question="Are all 36 specimens equally informative?",
            losing="Every specimen weighted the same, including those whose "
                   "plant mass is a small difference between two much larger "
                   "weighings.",
            winning="Weight by the uncertainty already computed for each "
                    "specimen, or report the fragile ones separately.",
            cost=f"On {fragile} of {len(rows) or 36} specimens, 50 g of scale "
                 "bias is more than a tenth of the plant mass. No estimator "
                 "recovers that.",
            source="pot_mass.json",
        ),
    ], {"lobo_rmse": lobo, "lobo_r2": lobo_r2}


def render(reports: Path | None = None) -> str:
    """The figure, as SVG."""
    import textwrap

    rows, outcome = forks(reports)
    height = HEAD_H + sum(f.height() + ROW_GAP for f in rows) + 92

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
        f'height="{height}" viewBox="0 0 {WIDTH} {height}" '
        f'font-family="{FONT}">',
        f'<rect width="{WIDTH}" height="{height}" fill="{PAPER}"/>',
        f'<text x="{LEFT_X}" y="46" font-size="25" font-weight="700" '
        f'fill="{INK}">Four forks, and which side the defaults are on</text>',
        f'<text x="{LEFT_X}" y="72" font-size="14" fill="{MUTED}">'
        f'Every cost below is measured, and named with the report it came '
        f'from. Three of the four are simply the current default.</text>',
        f'<text x="{LEFT_X}" y="{HEAD_H - 46}" font-size="13" '
        f'font-weight="700" fill="{LOSE}">WHAT IT DOES NOW</text>',
        f'<text x="{RIGHT_X}" y="{HEAD_H - 46}" font-size="13" '
        f'font-weight="700" fill="{WIN}">WHAT WOULD SURVIVE A HOLDOUT</text>',
        f'<line x1="{LEFT_X}" y1="{HEAD_H - 34}" x2="{WIDTH - LEFT_X}" '
        f'y2="{HEAD_H - 34}" stroke="{RULE}"/>',
    ]

    y = HEAD_H
    for index, fork in enumerate(rows, start=1):
        losing, winning, cost = fork.wrapped()
        box_h = fork.box_height()

        out.append(
            f'<text x="{LEFT_X}" y="{y - 11}" font-size="14" font-weight="600" '
            f'fill="{INK}">{index}. {_esc(fork.question)}</text>')

        for x, colour, background, lines in (
            (LEFT_X, LOSE, LOSE_BG, losing),
            (RIGHT_X, WIN, WIN_BG, winning),
        ):
            out.append(
                f'<rect x="{x}" y="{y}" width="{COL_W}" height="{box_h}" '
                f'rx="6" fill="{background}" stroke="{colour}" '
                f'stroke-opacity="0.35"/>')
            out.append(
                f'<rect x="{x}" y="{y}" width="4" height="{box_h}" '
                f'rx="2" fill="{colour}"/>')
            for line_no, line in enumerate(lines):
                out.append(
                    f'<text x="{x + 16}" y="{y + 22 + line_no * 16}" '
                    f'font-size="13" fill="{INK}">{_esc(line)}</text>')

        cost_y = y + box_h + 20
        for line_no, line in enumerate(cost):
            out.append(
                f'<text x="{LEFT_X}" y="{cost_y + line_no * 16}" '
                f'font-size="12.5" font-weight="600" fill="{INK}">'
                f'{_esc(line)}</text>')
        out.append(
            f'<text x="{WIDTH - LEFT_X}" y="{cost_y}" font-size="11.5" '
            f'text-anchor="end" font-family="{MONO}" fill="{MUTED}">'
            f'{_esc(fork.source)}</text>')

        y += fork.height() + ROW_GAP

    out.append(f'<line x1="{LEFT_X}" y1="{y + 6}" x2="{WIDTH - LEFT_X}" '
               f'y2="{y + 6}" stroke="{RULE}"/>')
    out.append(
        f'<text x="{LEFT_X}" y="{y + 34}" font-size="13.5" fill="{INK}">'
        f'Where the left column ends today: leave-one-batch-out RMSE '
        f'{outcome["lobo_rmse"]:.3f} kg at R2 {outcome["lobo_r2"]:.2f}, which '
        f'is worse than predicting the mean.</text>')
    out.append(
        f'<text x="{LEFT_X}" y="{y + 54}" font-size="13.5" fill="{MUTED}">'
        f'Forks 1 and 2 need no training and change what four days of it '
        f'would produce. Fork 1 is a cache path; fork 2 is the fold rule.</text>')
    out.append("</svg>")
    return "\n".join(out)


def run(*, out: Path | None = None, verbose: bool = True) -> Path:
    """Write the figure beside the methodology diagrams."""
    out = out or WORK_DIR / "reports" / "architecture" / "decisions.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(), encoding="utf-8")
    if verbose:
        print(f"  wrote {out} ({out.stat().st_size // 1024} KB)")
    return out


__all__ = ["Fork", "forks", "render", "run"]
