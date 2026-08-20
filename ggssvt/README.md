# GG-SSVT

Geometry-Grounded Self-Supervised Vision Transformer for label-efficient
volumetric reconstruction and biomass estimation, implemented against the
dual-Kinect single-plant captures in [`../dataset`](../dataset).

The four components named in the dissertation scaffold:

| Component | Module | What it does |
|---|---|---|
| Fourier back-projected token embeddings | [`models/embedding.py`](models/embedding.py) | Positions tokens by their world coordinate, not their image index |
| Cross-view geometric attention | [`models/attention.py`](models/attention.py) | Biases attention logits by `-gamma * ||x_i - x_j||^2`, gamma learned per head |
| Implicit occupancy decoder | [`models/decoder.py`](models/decoder.py) | World coordinate + fused context to occupancy, evaluated in chunks |
| Space-carving self-supervision | [`geometry/carving.py`](geometry/carving.py) | Depth and silhouettes to the occupancy targets, no manual labels |
| Biomass head | [`models/head.py`](models/head.py) | Volume integration with a modulated density prior |

## Quick start

```bash
python -m ggssvt.cli inspect
```

```bash
python -m ggssvt.cli preprocess
```

```bash
python -m ggssvt.cli baselines
```

```bash
python -m ggssvt.cli pretrain --epochs 120 --out work_dirs/ggssvt/checkpoints/pretrain.pt
```

```bash
python -m ggssvt.cli loocv --checkpoint work_dirs/ggssvt/checkpoints/pretrain.pt
```

```bash
python -m ggssvt.cli report
```

`preprocess` takes roughly nine seconds per specimen and writes a ~4 MB archive
each; everything after it reads only the cache. Only `pretrain` and `loocv` need
a GPU.

## Pipeline

```
dataset/plants/<id>/          12 RGB-D frames, 512x424, registered
        |
        |  geometry/rig.py         floor plane -> tilt, roll, camera height
        |                          subject column -> world origin on the plant axis
        |                          geometry/refine.py -> residual azimuth correction
        v
   registered views
        |
        |  geometry/segment.py     cylinder about the world axis
        v
   subject masks
        |
        |  geometry/carving.py     silhouette + depth carving -> 128^3 occupancy
        v
   pseudo-labels ------------------> stage 1: self-supervised occupancy
        |                                        |
        |                                        v
        +--> ground_truth.csv ------> stage 2: biomass head, LOOCV
```

## Calibration: what is missing, and what is done instead

`dataset/calib` is empty. No ChArUco intrinsics were captured, no per-day
`rig_positions.json` exists, and the `positions/` directories contain no images.
There are therefore no measured extrinsics to load.

[`geometry/rig.py`](geometry/rig.py) estimates them from the depth data instead:

1. **Floor plane** (RANSAC per view) gives tilt, roll and camera height — four
   of six degrees of freedom, with no calibration target. Recovered camera
   heights are consistent to about 3 cm across a sweep, which is a useful
   independent check that the fit is working.
2. **Subject axis** puts the world origin on the plant. Several candidate
   columns are proposed per view and the winner is the one whose registration
   the *other views actually agree with* — scored by how many voxels several
   cameras land on together. This matters: an earlier version picked the
   strongest single-view candidate and silently locked onto background
   structure a metre behind the plant on several specimens.
3. **Azimuth** comes from the filename, then
   [`geometry/refine.py`](geometry/refine.py) corrects it. The correction is
   worth a lot — surface coverage on E011 goes from 0.29 to 0.48, and on M001
   from 0.16 to 0.43.

**This is an estimate, not a measurement.** The azimuth corrections saturate the
±8° search bound on most specimens, which means the true placement error is at
least that large and possibly larger. Capturing the ChArUco sequence that
`dataset/README.md` already specifies would replace all of step 3 with a
measurement and is the single highest-value thing that could be added to the
capture protocol.

## Two naming conventions in the capture set

`rig_calibration/collect_specimen.py` names each camB file with **camA's** step
angle (`camB_000` … `camB_150`) even though it prints the opposite angle to the
console and `dataset/README.md` documents the ids as `camB_180` … `camB_330`.
Both conventions are present in `dataset/plants`, and taking the filename
literally would place camB on top of camA instead of opposite it, collapsing the
12-view rig into a 6-view one.

[`data/naming.py`](data/naming.py) resolves both: camB physically occupies the
upper half-circle, so a camB filename angle below 180 is the step-angle
convention and is rotated by 180 degrees.

## Evaluation protocol

Leave-one-out cross-validation over the usable specimens, for every method in
the comparison table, baselines included.

Stage 1 pretraining runs **once over every specimen** by default. No mass label
is touched, but the held-out plant's images are seen, which makes the protocol
transductive rather than inductive. `--strict` re-runs pretraining inside each
fold for the inductive number; the gap between the two is worth reporting rather
than hiding.

## Ablations

```bash
python -m ggssvt.cli pretrain --no-geometry --out work_dirs/ggssvt/checkpoints/ablation.pt
```

`--no-geometry` zeroes and freezes both geometric pathways — the Fourier
back-projected positional code and the 3D-distance attention bias — leaving a
multi-view transformer of the same shape with no 3D prior. Everything else is
identical, so the difference is attributable to geometry grounding alone.

After training, `model.fusion.distance_scales()` returns the learned per-head
`gamma`. That is the direct evidence for whether the distance bias is used at
all, and it belongs in the ablation section.

## Known limitations

- **Registration is estimated, not measured.** See above.
- **Tall specimens are truncated.** At the ~1 m working radius the vertical
  field of view reaches about 1.15 m above the floor. The E011–E020 eucalyptus
  saplings extend past the top of frame, so their canopies are cut off and their
  carved volumes are underestimates. The rig should step back for tall plants,
  or add a raised second tier.
- **Ground truth is fresh mass, not oven-dry AGB.** `net_weight_g` is an
  as-collected weight; every dissertation and proposal draft specifies oven-dry
  above-ground biomass. These are not interchangeable and the distinction has to
  be stated wherever the numbers appear.
- **Pot masses are estimated, not weighed.** Every row of `ground_truth.csv`
  carries `pot_weight_source = estimated`, so the target itself has unquantified
  error. Weighing a sample of empty pots would bound it cheaply.
- **Two specimens fail the quality gate** (E012, E016) and `X001` has only two
  views. That leaves 28 usable specimens across two species — enough to fit a
  head, not enough for a strong generalisation claim.

## Layout

```
ggssvt/
  config.py            every camera constant, threshold and hyperparameter
  cli.py               command-line entry point
  data/
    naming.py          the camB convention fix
    io.py              PNG loading, back-projection, projection
    dataset.py         specimen index joined to ground_truth.csv
    preprocess.py      geometry cache and the quality report
  geometry/            NumPy only, no PyTorch
    plane.py           RANSAC plane fitting
    rig.py             calibration-free extrinsics
    refine.py          residual azimuth and offset correction
    segment.py         subject segmentation in the world frame
    carving.py         space carving to occupancy
  models/              PyTorch
    embedding.py       Fourier back-projected positional encoding
    attention.py       3D-distance-biased cross-view attention
    encoder.py         RGB-D patch tokens, pruned to the subject
    decoder.py         implicit occupancy decoder
    head.py            biomass head
    ggssvt.py          the assembled model
  training/
    dataset.py         torch dataset and query sampling
    losses.py          occupancy, biomass, volume consistency
    trainer.py         two-stage schedule and LOOCV
  eval/
    metrics.py         RMSE/MAE/MARE/R2, IoU, Chamfer, F-score
    baselines.py       allometric, geometric-feature, direct-2D, mean
    report.py          tables and figures
    visualise.py       rig and mask overlays
```

## Tests

```bash
python -m pytest tests/ -q
```

Tests needing the preprocessed cache skip when it is absent, so a fresh clone
passes without running preprocessing first.
