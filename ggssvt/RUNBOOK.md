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

**Expect `N passed, 7 skipped` — that is correct.** The seven are integration
tests in `test_pipeline.py` gated on the preprocessed cache, which does not exist
yet: `work_dirs/` is gitignored because it holds about 120 MB of derived data
that regenerates in minutes. They skip until step 1 has run, then join in.

Read the **failure** count, not the pass count. The total changes whenever tests
are added, so it is not a useful check:

| What you see | What it means |
|---|---|
| `0 failed`, 7 skipped | Correct, before preprocessing |
| `0 failed`, 0 skipped | Correct, after preprocessing |
| any failures | Fix before spending GPU hours |
| errors during *collection* | A module is missing — see below |

Collection errors reading `ModuleNotFoundError: No module named 'ggssvt.data'`
mean the repository is out of date. Check with `git ls-files ggssvt/data/`, which
must list five files, and `git log --oneline -1` against the remote.

---

## Before you start: request the two gated models

Both are manual approvals by Meta and can take hours to days, so submit them
first and let them run in the background.

- DINOv3 — <https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m>
- SAM 3 / SAM 3D Objects — <https://huggingface.co/facebook/sam-3d-objects>

```bash
hf auth login
```

Then check what this machine can actually reach:

```bash
python -m ggssvt.cli access
```

It prints the authenticated account, where the token came from, and an OK/BLOCKED
line per model. **Nothing needs a token added to it.** `hf auth login` writes one
to `~/.cache/huggingface`, and `huggingface_hub` and `transformers` pick it up on
their own — no function in this codebase takes a token argument, and no script
should ever contain one.

Two ways an approved model still reads BLOCKED:

**Wrong account.** Approvals are per account. If the settings page says ACCEPTED
but `access` reports BLOCKED, compare the name it prints against the account that
was approved. Fix with `hf auth login --force`.

**`HF_TOKEN` overriding the login.** If that variable is set — in `.bashrc`, a
job script, a conda activation hook — it **takes precedence over `hf auth login`
entirely**, so logging in again changes nothing. `access` reports which source is
in play. To clear it:

```bash
unset HF_TOKEN && python -m ggssvt.cli access
```

For headless or batch jobs, `HF_TOKEN` is the right mechanism — just make sure it
holds the approved account's token, and keep it out of anything committed.

Everything runs without either model; the DINOv3 cells simply report as skipped.

---

## Morning — cheap, and it unblocks the afternoon

### 1. Build both caches (~4 min geometric, ~3 min SAM3D on GPU)

`work_dirs/` is not in the repository, so on a fresh machine there is no cache at
all and everything downstream — baselines, probes, factorial, training — has
nothing to read.

**Both commands are required, and the first one is easy to skip.** The default
`--cache-dir` is `work_dirs/ggssvt/cache`, which only the *geometric* run
writes. Running only the SAM3D command leaves that directory empty, and the next
thing you run fails with `no quality report at .../cache/quality.json`.

```bash
python -m ggssvt.cli preprocess
```

```bash
python -m ggssvt.cli preprocess --segmenter sam3d --cache-dir work_dirs/ggssvt/cache_sam3d --sam-device cuda
```

Then confirm the caches are sane before building on them:

```bash
python -m ggssvt.cli inspect && python -m ggssvt.cli baselines
```

`inspect` audits the raw dataset and reports the missing calibration; `baselines`
prints the leave-one-out table the whole day is measured against. If `baselines`
does not roughly reproduce geometric features at RMSE 0.397 / R² 0.526, the
preprocessing differed from the reference run and everything after it will too.

Expect 28/30 specimens through the quality gate on the geometric cache and 26/30
on SAM3D — SAM3D additionally drops E015 and E019. About 4 s/specimen for the
geometric pass and 5 s/specimen for SAM3D on the GPU, so under five minutes for
both. Then re-run the tests; the seven skipped ones should join in:

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

### 3c. Mesh arm and reconstruction gallery (~5 min, CPU)

```bash
python -m ggssvt.cli mesh --export work_dirs/ggssvt/meshes
```

Marching cubes on the carved occupancy, then biomass from mesh descriptors —
scored under the same leave-one-out protocol as every other method, so the table
is directly comparable. It is currently the **best** method (RMSE 0.359,
R² 0.613, against 0.397 / 0.526 for the voxel features), though the paired
interval spans zero. `--export` also writes OBJ files for MeshLab.

Needs `scikit-image` and `scipy`, both in `requirements.txt`. Nothing else in
the pipeline depends on them.

```bash
python -m ggssvt.cli gallery
```

Contact sheets, PLY point clouds and a self-contained interactive HTML viewer of
every reconstruction under both segmenters. Worth five minutes before committing
to a day of training: it is how the "E001–E010 are mostly pot" problem became
visible in the first place, and it will show immediately if a preprocessing run
on this machine went wrong in a way the quality gate did not catch.

---

## What today's biomass numbers can and cannot say

Read this before interpreting anything the afternoon produces.

The Eucalyptus specimens are two batches with almost no overlap in mass —
E001–E010 average 0.538 kg, E011–E020 average 1.844 kg. **Batch membership alone
explains R² = 0.887**, which is more than any method achieves. Within either
batch, no method clears R² = 0.2.

So the comparison is measuring how well each method separates *size classes*, not
how well it estimates mass among comparable plants. That is still a real
comparison and worth running — a method that separates the classes better is
extracting more from the reconstruction — but the headline claim it supports is
"reconstructed geometry separates plant size classes", not "estimates biomass".

Nothing today fixes this; only a capture batch spanning a continuous mass range
within one species will. Note it in the log so the framing is decided now rather
than at examination.

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

## Pose-free reconstruction — the registration check

```bash
python -m ggssvt.cli posefree --check-only
```

Reports which of DUSt3R, MASt3R and Fast3R are installed, with the clone command
for each. All three sets of weights are open on HuggingFace; it is the
repositories that must be cloned, with submodules:

```bash
git clone --recursive https://github.com/naver/mast3r && cd mast3r && pip install -r requirements.txt && pip install -e .
```

Then:

```bash
python -m ggssvt.cli posefree --methods mast3r --device cuda
```

**Run this for the poses, not for the biomass.** Every camera pose in the project
is estimated from depth, and the azimuth refinement saturates its ±8° search
bound on almost every specimen — so the registration is the least verified
assumption in the pipeline, and nothing inside the pipeline can test it. A
pose-free method shares no failure mode with a depth-based registration, so the
residual after similarity alignment is a real measurement of how wrong the poses
are. **Azimuth RMSE is the number to read.** Under 8° would be reassuring; more
would not be.

Use **MASt3R's metric checkpoint** first. DUSt3R and Fast3R return arbitrary
scale, so their volumes depend on a scale recovered from the Kinect depth, which
makes them only partly independent of the pipeline they are meant to check.
MASt3R-metric has no such loop.

The adapters are written against each project's documented interface but have
**not been run against the real weights** — no GPU was available where they were
written. The maths they feed is tested; the adapters are not. Watch the first run
for import errors and convention drift rather than assuming quiet success.
`sanity_check_result` warns if the cameras stop facing the scene, which is what a
convention change looks like.

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

**Export the poses first — `ns-train` has nothing to read without them.** This
runs in the `ggssvt` environment, not `cropcraft`, and writes `transforms.json`
into each specimen directory:

```bash
conda activate ggssvt && python -m ggssvt.cli nerfstudio && conda activate cropcraft
```

Skip it and `ns-train` fails on a missing `transforms.json`, which reads like a
Nerfstudio problem rather than a missing step.

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

## End of day — collect the results

```bash
python -m ggssvt.cli report --folds work_dirs/ggssvt/reports/folds_geo_cnn.json
```

Writes `report.md`, `comparison.csv`, `metrics.json` and a predicted-versus-true
scatter to `work_dirs/ggssvt/reports/`. Pass `--folds` to fold a GG-SSVT
leave-one-out run into the comparison table alongside the baselines; without it
you get the baselines only.

```bash
python -m ggssvt.cli visualise --plants M003 E001 E011
```

Rig overlays (the world axis projected into every view) and mask overlays. Run
these if any registration looks suspect — if the red axis is not on the plant
stem in all twelve frames, every number for that specimen is meaningless.

---

## Every command

| Command | Needs | Roughly |
|---|---|---|
| `inspect` | nothing | seconds — dataset audit, no computation |
| `access` | network | seconds — HuggingFace account and model access |
| `preprocess` | dataset | 2–3 min per segmenter |
| `baselines` | cache | seconds — LOOCV baseline table |
| `mesh` | cache, scikit-image | ~1 min including the comparison |
| `gallery` | cache | ~2 min — contact sheets, PLY, HTML viewer |
| `visualise` | dataset | seconds per specimen |
| `dino-probe` | cache, network | 5–15 min depending on variant |
| `factorial` | both caches | ~10 min frozen; hours with `--train` |
| `pretrain` | cache, **GPU** | ~25 min at 120 epochs |
| `loocv` | cache, checkpoint, **GPU** | ~40 min at 60 fine-tune epochs |
| `experiment` | cache, **GPU** | one pretrain + one LOOCV per backbone |
| `nerfstudio` | dataset | seconds — writes `transforms.json` |
| `dashboard` | cache, mesh cache | ~1 min — the walkthrough page |
| `posefree` | cloned repos, **GPU** | minutes per specimen per method |
| `report` | cache | seconds |

`experiment` is the single-factor backbone comparison; `factorial` supersedes it
by crossing backbone with segmenter. Use `experiment` only when you want
backbones alone on one cache.

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
