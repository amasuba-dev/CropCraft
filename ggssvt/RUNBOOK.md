# One day on the RTX 4080

Everything in order, with what it costs.

**Hardware note.** This was first written for an RTX 4060 (8 GB). The lab card is
an **RTX 4080: 16 GB and roughly three times the compute**. Same Ada
architecture, so the `cu121` torch install is unchanged — but the VRAM ceiling
that shaped the original settings is gone, and the epoch counts below are raised
accordingly.

**Read this first:** the default `--finetune-epochs 200` in a leave-one-out sweep
is 28 folds × 200 epochs × 27 specimens. Even on this card that is most of a day
per condition. Every command below overrides it. If you copy a command from the
README instead of from here, you will start a multi-day run by accident.

**The bottleneck moves to the CPU.** Each specimen means decompressing a 4 MB
archive and back-projecting 2.5 M points. On a 4060 that overlapped with GPU
work; on a 4080 it will not. Run with `--workers 8` throughout, and if
`nvidia-smi` shows the GPU below about 70% utilisation, raise it further — a
faster card buys nothing while it waits on numpy.

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

Must print a CUDA build and `NVIDIA GeForce RTX 4080`. If `cuda.is_available()`
is False you installed a CPU wheel, and every command below will silently run on
the CPU at roughly a hundredth of the speed rather than failing.

Expected: `2.5.1+cu121 True NVIDIA GeForce RTX 4080`.

```bash
python -m pytest tests/ -q
```

**Expect `89 passed, 7 skipped` at this point — that is correct.** Seven tests in
`test_pipeline.py` are integration tests gated on the preprocessed cache, and
`work_dirs/` is gitignored because it holds about 120 MB of derived data that
regenerates in four minutes. They skip until step 1 below has run, then all
**96** pass. Re-run the suite after preprocessing to collect them.

What matters is the failure count, not the pass count. Any *failure* means
something about the environment is wrong; fix it before spending GPU hours. A
much lower pass count than 89 usually means a missing module rather than a real
failure — check `git ls-files ggssvt/data/` returns five files.

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

### 1. Build both caches (~4 min geometric, ~3 min SAM3D on GPU)

`work_dirs/` is not in the repository, so on a fresh machine there is no cache at
all and everything downstream — baselines, probes, factorial, training — has
nothing to read. Build the geometric one first:

```bash
python -m ggssvt.cli preprocess
```

```bash
python -m ggssvt.cli preprocess --segmenter sam3d --cache-dir work_dirs/ggssvt/cache_sam3d --sam-device cuda
```

Expect 28/30 specimens to pass the quality gate on the geometric cache and 26/30
on SAM3D. Then re-run the tests and the seven skipped integration tests should
join in:

```bash
python -m pytest tests/ -q
```

### 2. Time one epoch before committing to anything (~2 min)

```bash
python -m ggssvt.cli pretrain --epochs 3 --workers 8 --batch-size 2 --device cuda --out /tmp/timing.pt
```

Note the seconds per epoch. **Every estimate below assumes ~12 s/epoch** on this
card. Watch `nvidia-smi` while it runs: if utilisation sits low you are
dataloader-bound and should raise `--workers` before anything else. Scale every
duration below by whatever ratio you actually measure — do not trust these
numbers over your own stopwatch.

### 3. Full frozen-feature factorial (~10 min on GPU)

```bash
python -m ggssvt.cli factorial --variant base --backbones cnn dinov2 dinov3
```

This is the cheap answer to "does DINO or SAM3D help", and it already ran:
DINOv2-base was the best cell, but nothing was statistically resolved at n=26.
Re-running adds the DINOv3 row if your access came through. Descriptors are
cached, so the DINOv2 cells return instantly.

### 3b. DINOv2-large — the best use of the extra VRAM (~15 min)

```bash
python -m ggssvt.cli dino-probe --variant large --backbones dinov2
```

Do this one. The probe already shows DINOv2 small to base improving
(RMSE 0.335 to 0.295), but a single pairwise delta at n=28 was not significant.
**A third scale turns two points into a trend**, and a monotone
small/base/large improvement is far more persuasive than any one comparison —
it is evidence the backbone is doing something rather than that one run got
lucky. ViT-L/14 is 300M frozen parameters; it did not fit comfortably in 8 GB and
fits easily in 16.

---

## Afternoon — the GPU work

Budget roughly **three hours** for this block on this card. Do not try to run the full 2×3
factorial with training today; at these settings that is six pretrain runs plus
six LOOCV sweeps.

### 4. Pretrain the four factorial cells (~25 min each, ~1.5 h total)

Stage 1 is self-supervised and uses no mass labels, so one pretrain per cell
serves that cell's whole LOOCV sweep.

```bash
python -m ggssvt.cli pretrain --epochs 120 --workers 8 --batch-size 2 --device cuda --out work_dirs/ggssvt/checkpoints/geo_cnn.pt
```

```bash
python -m ggssvt.cli pretrain --epochs 120 --workers 8 --batch-size 2 --device cuda --cache-dir work_dirs/ggssvt/cache_sam3d --out work_dirs/ggssvt/checkpoints/sam_cnn.pt
```

The full 120 epochs, not the 60 the 8 GB plan called for — on this card you can
afford the config default, so there is no deviation to justify in the methods
section. That is worth more than it sounds: one fewer "reduced for compute
reasons" caveat in the write-up.

For the DINO cells, edit `backbone` in `ggssvt/config.py` to `"dinov2"`, or run
the combined command in step 6 which handles it.

### 5. One LOOCV sweep to calibrate the cost (~40 min)

```bash
python -m ggssvt.cli loocv --checkpoint work_dirs/ggssvt/checkpoints/geo_cnn.pt --finetune-epochs 60 --workers 8 --device cuda --out work_dirs/ggssvt/reports/folds_geo_cnn.json
```

60 fine-tune epochs rather than 25. Still well short of the 200 default, and that
is defensible on more than time: the biomass head is initialised as an exact
physical model (density × volume, zero residual), so it starts near a sensible
solution rather than at random. Check the fold errors have actually flattened —
if they are still falling at 60, raise it, because you now have the headroom.

Time this. It sets whether step 6 fits before you go home.

### 6. The trained factorial (~2.5–3 h with these settings)

```bash
python -m ggssvt.cli factorial --train --backbones cnn dinov2 --variant base --epochs 120 --finetune-epochs 60 --workers 8 --batch-size 2 --device cuda
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

## Spending 16 GB

You should not hit an OOM at the settings above. If a run finishes early, the
headroom is best spent in this order:

| Option | Why |
|---|---|
| `--variant large` in the probe | Third scale point turns a delta into a trend |
| `--tokens-per-view 96` | 64 tokens covers ~4% of frame; more captures thin branches |
| `--batch-size 4` | Fewer, better-conditioned gradient steps |
| `dinov3` in the factorial | Only if access came through |

Raising `--tokens-per-view` is the one most likely to change a *result* rather
than a runtime: the encoder keeps the K best subject patches per view, and thin
eucalyptus stems are exactly what falls off the bottom of that ranking.

### If something does OOM anyway

| Symptom | Fix |
|---|---|
| OOM during fine-tuning | drop `--batch-size` to 1 |
| OOM in the decoder | lower `query_chunk` in `config.py` to 8192 |
| OOM with `--variant large` | fall back to `base` |

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` helps with fragmentation
across the 28 folds of a LOOCV sweep.

---

## What to record as you go

Because these are deviations from the config defaults, and the methods section
needs them:

- epochs actually used for stage 1 and stage 2, and why
- the GPU is an RTX 4080 (16 GB), not the 4060 the first draft assumed
- seconds per epoch, and total GPU hours per condition
- which cells were skipped for gated access
- the transductive-vs-`--strict` distinction, if you run the strict protocol

Everything writes JSON alongside its output, so the numbers survive the day.
