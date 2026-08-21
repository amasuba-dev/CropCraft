# One day on the RTX 4060

Everything in order, with what it costs. The RTX 4060 has **8 GB of VRAM**, which
is the binding constraint on batch size and query chunk, and one GPU, which is
the binding constraint on how many training runs fit in a day.

**Read this first:** the default `--finetune-epochs 200` in a leave-one-out sweep
is 28 folds × 200 epochs × 27 specimens. That is roughly **29 hours per
condition**. Every command below overrides it. If you copy a command from the
README instead of from here, you will start a multi-day run by accident.

---

## Two environments, not one

They cannot be merged. Nerfstudio pins torch 2.0.1+cu118, tiny-cuda-nn and
Python 3.8; GG-SSVT needs `transformers >= 4.56` for the DINOv3 and SAM model
classes, which will not install on Python 3.8.

```bash
conda env create -f ggssvt/environment.yml
```

```bash
conda activate ggssvt && pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

```bash
pip install -r ggssvt/requirements.txt
```

Keep the existing `cropcraft` environment for `ns-train` / `ns-viewer`.

### Verify before anything else

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Must print a CUDA build and `NVIDIA GeForce RTX 4060`. If `cuda.is_available()`
is False you installed a CPU wheel, and every command below will silently run on
the CPU at roughly a hundredth of the speed rather than failing.

```bash
python -m pytest tests/ -q
```

93 tests. Any failure here means something about the port is wrong; fix it before
spending GPU hours.

---

## Before you start: request the two gated models

Both are manual approvals by Meta and can take hours to days, so submit them
first and let them run in the background.

- DINOv3 — <https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m>
- SAM 3 / SAM 3D Objects — <https://huggingface.co/facebook/sam-3d-objects>

```bash
hf auth login
```

```bash
python -c "from ggssvt.models.backbones import backbone_is_available; print(backbone_is_available('dinov3','base'))"
```

Everything works without them; the DINOv3 cells simply report as skipped.

---

## Morning — cheap, and it unblocks the afternoon

### 1. Re-run SAM3D preprocessing on the GPU (~3 min, was ~40 min on CPU)

```bash
python -m ggssvt.cli preprocess --segmenter sam3d --cache-dir work_dirs/ggssvt/cache_sam3d --sam-device cuda
```

The geometric cache does not need rebuilding unless you changed the geometry
code.

### 2. Time one epoch before committing to anything (~2 min)

```bash
python -m ggssvt.cli pretrain --epochs 2 --workers 4 --device cuda --out /tmp/timing.pt
```

Note the seconds per epoch. **Every estimate below assumes ~35 s/epoch.** If
yours is 70, halve every epoch count here. If the GPU is idling, raise
`--workers`; the dataloader decompresses a 4 MB archive and back-projects 2.5 M
points per specimen, so it starves the GPU at `--workers 0`.

### 3. Full frozen-feature factorial (~10 min on GPU)

```bash
python -m ggssvt.cli factorial --variant base --backbones cnn dinov2 dinov3
```

This is the CPU-cheap answer to "does DINO or SAM3D help", and it already ran:
DINOv2-base was the best cell, but nothing was statistically resolved at n=26.
Re-running adds the DINOv3 row if your access came through. Descriptors are
cached, so the DINOv2 cells return instantly.

---

## Afternoon — the GPU work

Budget roughly **five hours** for this block. Do not try to run the full 2×3
factorial with training today; at these settings that is six pretrain runs plus
six LOOCV sweeps.

### 4. Pretrain the four factorial cells (~35 min each, ~2.5 h total)

Stage 1 is self-supervised and uses no mass labels, so one pretrain per cell
serves that cell's whole LOOCV sweep.

```bash
python -m ggssvt.cli pretrain --epochs 60 --workers 4 --device cuda --out work_dirs/ggssvt/checkpoints/geo_cnn.pt
```

```bash
python -m ggssvt.cli pretrain --epochs 60 --workers 4 --device cuda --cache-dir work_dirs/ggssvt/cache_sam3d --out work_dirs/ggssvt/checkpoints/sam_cnn.pt
```

60 epochs rather than the 120 default: the occupancy loss flattens well before
120 on 28 specimens, and you need the hours for the LOOCV sweeps. Record that you
did this — it is a deviation from the config default and belongs in the methods
section.

For the DINO cells, edit `backbone` in `ggssvt/config.py` to `"dinov2"`, or run
the combined command in step 6 which handles it.

### 5. One LOOCV sweep to calibrate the cost (~50 min)

```bash
python -m ggssvt.cli loocv --checkpoint work_dirs/ggssvt/checkpoints/geo_cnn.pt --finetune-epochs 25 --workers 4 --device cuda --out work_dirs/ggssvt/reports/folds_geo_cnn.json
```

25 fine-tune epochs is defensible here and not just a shortcut: the biomass head
is initialised as an exact physical model (density × volume, zero residual), so
it starts near a sensible solution rather than at random. Check the reported
fold errors are not still falling — if they are, raise it.

Time this. It sets whether step 6 fits before you go home.

### 6. The trained factorial (~3–4 h with these settings)

```bash
python -m ggssvt.cli factorial --train --backbones cnn dinov2 --variant base --epochs 60 --finetune-epochs 25 --workers 4 --device cuda
```

Four cells: geometric×cnn, geometric×dinov2, sam3d×cnn, sam3d×dinov2. It prints
the same paired-effect table as the frozen probe, so the two are directly
comparable — **and that comparison is itself a result.** If the trained factorial
reproduces the probe's sign flip (SAM3D alone hurts, SAM3D given DINO helps), the
finding survives moving from pooled descriptors to the full model. If it does
not, the probe was measuring an artefact of pooling.

Add `dinov3` to `--backbones` only if you have both the access and the hours.

---

## Evening — Nerfstudio, in the other environment

```bash
conda activate cropcraft
```

The vendored copy in `nerfstudio/` does not import: `.gitignore` line 1 is
`data/`, which matched `nerfstudio/nerfstudio/data/` and kept the dataparsers out
of the repository. Install upstream instead:

```bash
pip install nerfstudio
```

Then, on one good specimen and one bad one:

```bash
ns-train splatfacto --data dataset/plants/M003 --pipeline.model.camera-optimizer.mode SO3xR3 --experiment-name M003
```

```bash
ns-train depth-nerfacto --data dataset/plants/E001 --pipeline.model.camera-optimizer.mode SO3xR3 --experiment-name E001_depth
```

M003 reconstructs well and E001 is nearly all pot, so the pair tells you whether
a radiance field recovers canopy the carve missed, or fails the same way.

**Always pass the camera optimiser**, then diff the optimised poses against the
exported ones. The azimuth refinement saturates its ±8° bound on almost every
specimen, so a photometric estimate of the same poses is an independent
measurement of how wrong the registration is — the single most valuable number
available for the price of one training run.

---

## If VRAM runs out

8 GB is tight with a DINOv2-base backbone. In order of what to try:

| Symptom | Fix |
|---|---|
| OOM during fine-tuning | `--tokens-per-view 48` |
| OOM in the decoder | lower `query_chunk` in `config.py` to 8192 |
| OOM with DINOv2-base | use `--variant small` |
| Still OOM | set `use_checkpointing=True` in `config.py` (already the default) |

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` helps with fragmentation
across the 28 folds of a LOOCV sweep.

---

## What to record as you go

Because these are deviations from the config defaults, and the methods section
needs them:

- epochs actually used for stage 1 and stage 2, and why
- seconds per epoch, and total GPU hours per condition
- which cells were skipped for gated access
- the transductive-vs-`--strict` distinction, if you run the strict protocol

Everything writes JSON alongside its output, so the numbers survive the day.
