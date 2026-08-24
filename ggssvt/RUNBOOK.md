# One day on the RTX 4080

Everything in order, with what it costs.

> **⚠ The targets changed on 2026-08-22. Every trained number predates it.**
>
> V001–V008 were added and their pot weights are now *measured* rather than
> estimated, which moved their net masses by 2.1–5.6×. The dataset is 38
> specimens, 36 usable. Any checkpoint or result produced before that date was
> fitted against the old targets and cannot be compared with anything produced
> after it.
>
> **Rebuild both caches before running anything**, then re-run the whole
> campaign. The campaign's resume logic skips runs marked `done`, so delete
> `work_dirs/ggssvt/campaign/` first, or it will happily skip every stale run
> you are trying to replace. See [RERUN_V_BATCH.md](RERUN_V_BATCH.md).

**Hardware note.** This was first written for an RTX 4060 (8 GB). The lab card is
an **RTX 4080: 16 GB and roughly three times the compute**. Same Ada
architecture, so the `cu121` torch install is unchanged, but the VRAM ceiling
that shaped the original settings is gone, and the epoch counts below are raised
accordingly.

**Read this first:** the default `--finetune-epochs 200` in a leave-one-out sweep
is 28 folds × 200 epochs × 27 specimens. Even on this card that is most of a day
per condition. Every command below overrides it. If you copy a command from the
README instead of from here, you will start a multi-day run by accident.

**The bottleneck moves to the CPU.** Each specimen means decompressing a 4 MB
archive and back-projecting 2.5 M points. On a 4060 that overlapped with GPU
work; on a 4080 it will not. Run with `--workers 8` throughout, and if
`nvidia-smi` shows the GPU below about 70% utilisation, raise it further, a
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

**Expect `N passed, 7 skipped`. That is correct.** The seven are integration
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
| errors during *collection* | A module is missing, see below |

Collection errors reading `ModuleNotFoundError: No module named 'ggssvt.data'`
mean the repository is out of date. Check with `git ls-files ggssvt/data/`, which
must list five files, and `git log --oneline -1` against the remote.

---

## Before you start: request the two gated models

Both are manual approvals by Meta and can take hours to days, so submit them
first and let them run in the background.

- DINOv3, <https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m>
- SAM 3 / SAM 3D Objects, <https://huggingface.co/facebook/sam-3d-objects>

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
their own, no function in this codebase takes a token argument, and no script
should ever contain one.

Two ways an approved model still reads BLOCKED:

**Wrong account.** Approvals are per account. If the settings page says ACCEPTED
but `access` reports BLOCKED, compare the name it prints against the account that
was approved. Fix with `hf auth login --force`.

**`HF_TOKEN` overriding the login.** If that variable is set, in `.bashrc`, a
job script, a conda activation hook. It **takes precedence over `hf auth login`
entirely**, so logging in again changes nothing. `access` reports which source is
in play. To clear it:

```bash
unset HF_TOKEN && python -m ggssvt.cli access
```

For headless or batch jobs, `HF_TOKEN` is the right mechanism, just make sure it
holds the approved account's token, and keep it out of anything committed.

Everything runs without either model; the DINOv3 cells simply report as skipped.

---

## Morning, cheap, and it unblocks the afternoon

### 1. Build both caches (~4 min geometric, ~3 min SAM3D on GPU)

`work_dirs/` is not in the repository, so on a fresh machine there is no cache at
all and everything downstream (baselines, probes, factorial, training) has
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
does not roughly reproduce geometric features at **RMSE 0.544 / R² 0.030** and
direct 2D at **0.469 / 0.279**, the preprocessing differed from the reference run
and everything after it will too.

**Those are the post-V001–V008 numbers, and 2D being ahead is expected, though
the difference is not statistically resolved, so do not read it as a result.** If you see
geometric features at 0.397 / 0.526 you are running the old 28-specimen cache
without the V batch, check `dataset/ground_truth.csv` has `measured` in the
`pot_weight_source` column for V001–V008, and re-run `preprocess`.

Expect 36/38 specimens through the quality gate on the geometric cache. About 4 s/specimen for the
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
duration below by whatever ratio you actually measure, do not trust these
numbers over your own stopwatch.

### 3. Full frozen-feature factorial (~10 min on GPU)

```bash
python -m ggssvt.cli factorial --variant base --backbones cnn dinov2 dinov3
```

This is the cheap answer to "does DINO or SAM3D help", and it already ran:
DINOv2-base was the best cell, but nothing was statistically resolved at n=26.
Re-running adds the DINOv3 row if your access came through. Descriptors are
cached, so the DINOv2 cells return instantly.

### 3b. DINOv2-large, the best use of the extra VRAM (~15 min)

```bash
python -m ggssvt.cli dino-probe --variant large --backbones dinov2
```

Do this one. The probe already shows DINOv2 small to base improving
(RMSE 0.335 to 0.295), but a single pairwise delta at n=28 was not significant.
**A third scale turns two points into a trend**, and a monotone
small/base/large improvement is far more persuasive than any one comparison.
It is evidence the backbone is doing something rather than that one run got
lucky. ViT-L/14 is 300M frozen parameters; it did not fit comfortably in 8 GB and
fits easily in 16.

### 3b-bis. View-count ablation (~20 min CPU, run once)

Answers "would four images do?" with a physical criterion rather than a quality
score. Build the reduced caches, then score them:

```bash
for v in 3 4 6; do python -m ggssvt.cli preprocess --views $v --cache-dir work_dirs/ggssvt/cache_v$v; done
```

```bash
python -m ggssvt.cli views
```

Expect **0 of 25 plausible at four views** against 8 of 36 at twelve, and a
median implied density of 9.2 kg/m³, lighter than polystyrene. Agreement only
falls from 0.608 to 0.424 over the same range, which is why it is worth printing
both.

### 3b-ter. TSDF depth fusion (~11 min CPU, run once)

The carve cannot be improved past the visual hull, which is why 25 of 36
specimens imply a bulk density one to two orders of magnitude below plant
tissue. Fusing the same depth maps as a signed distance field keeps concavities
and leaves unobserved space empty.

```bash
python -m ggssvt.cli fuse --write-cache
```

Expect **31 of 36 plausible against the carve's 8**, median implied density
529 kg/m3 against 116.8. `--write-cache` also writes `cache_tsdf` at the carve's
own 128^3 and 12 mm, which is the control that isolates the operator from the
grid: 21 of 36 at matched resolution, so the method accounts for most of the gain
and the finer grid adds the rest.

Two things depend on this having run. `cli baselines` offers the `fused geometry`
method only when every specimen is present in `reports/fusion.json`, and the
project page shows TSDF in its method toggle only when `cache_tsdf` exists. Do
not run `fuse --plants ...` on a subset and then expect either: a partial report
is treated as no report, deliberately.

### 3c. Mesh arm and reconstruction gallery (~5 min, CPU)

```bash
python -m ggssvt.cli mesh --export work_dirs/ggssvt/meshes
```

Marching cubes on the carved occupancy, then biomass from mesh descriptors,
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

**Two things cap every biomass number, and both are now measured.**

*The batch confound, weakened but present.* E001–E010 average 0.538 kg and
E011–E020 average 1.844 kg, with almost no overlap; batch membership alone
explained **R² = 0.887**. V001–V008 was the batch that breaks it, 500–1800 g,
overlapping every other batch, and takes it down to **0.744** across the three
Eucalyptus batches, **0.697** across all four. Still high. The comparison is
partly measuring how well each method separates *size classes*.

*The reconstructions are envelopes, not plants.* Measured mass over reconstructed
above-ground volume implies a bulk density; fresh tissue is 300–900 kg/m³. Only
**8 of 36** specimens land in a generous 200–1000 band, 25 imply less (the hull
enclosing air between leaves; all ten Mango at 26–77) and 3 imply more (thin
stems never carved). No amount of training fixes a volume that is the wrong size
for its mass, and this, not any RMSE ordering, is what to report.

**The 3D-versus-2D RMSE comparison is not resolved in either direction.** Paired
bootstrap gives [−0.051, +0.227] on the current set and [−0.168, +0.099] on the
old n=28 one, and the winner flips depending on whether features are whitened
before the ridge. Do not quote an ordering from it.

Decide the framing now rather than at examination:
[RERUN_V_BATCH.md](RERUN_V_BATCH.md) §3 has the argument and the numbers.

---

## Multi-day: the training campaign

With more than a day on the machine, do not drive the training by hand. One
command queues the whole programme, writes a JSON result per run, and **skips
whatever already finished**, so a campaign that dies at 3am is restarted with
the same command and picks up where it stopped.

```bash
python -m ggssvt.campaign --plan smoke --device cuda
```

**Always run `smoke` first.** Two two-epoch runs over four specimens, about ten
minutes on the GPU, and it proves
the loop, the checkpointing, the resume logic and the result writing before you
commit a night to them.

```bash
python -m ggssvt.campaign --plan core --device cuda --workers 8 --batch-size 2
```

```bash
python -m ggssvt.campaign --plan full --device cuda --workers 8 --batch-size 2
```

`--list` prints a plan without running it; `--only NAME [NAME...]` runs a subset;
`--force` re-runs something already marked done.

### What the plans contain, and which question each run closes

Runs are ordered so the hypothesis-answering ones come first, if the campaign is
cut short, what completed is what the write-up needs.

| run | closes | why |
|---|---|---|
| `baseline_cnn` | reference | every comparison below is against this |
| `baseline_fused` | operator | same model on the fused cache. Classical features gained 0.209 kg there with the operator as the only change; whether a trained model inherits that is untested |
| `h2_no_geometry` | **H2** | the geometry-grounding ablation; γ frozen at zero, capacity identical |
| `h1_dinov2` | **H1** | ViT backbone against the CNN stem, inside the trained model |
| `h3_bands_8_freq7` | **H3** | encoding matched to the grid Nyquist, 41.7 cyc/m |
| `h3_bands_6_freq6` | **H3** | half the Nyquist, 20.8 cyc/m, expected to hurt |
| `h3_bands_16_freq10` | **H3** | 8× the Nyquist, 333 cyc/m, expected to add nothing |

**Run `cli fuse --write-cache` before the campaign.** Without `cache_tsdf` the
`baseline_fused` run has nothing to read and fails rather than being skipped.

**Delete `work_dirs/ggssvt/campaign/` before the first re-run.** Resume works by
skipping runs whose status is `done`, which is what you want after a crash and
exactly what you do not want after the targets changed.
| `sam3d_cnn`, `sam3d_dinov2` | factorial | tests whether the probe's interaction survives training |
| `h1_dinov3` | H1 | if access has been granted |

`core` is seven runs, roughly **8–10 hours** at 120 pretrain and 60 fine-tune
epochs. `full` is nine, roughly 13–15 hours. Time one epoch first (step 2) and
scale.

### What to report back from each run

The campaign writes `summary.txt` and a per-run JSON carrying everything below.
For the research status document you need, per run:

- **RMSE, MAE, MARE, R²** with the bootstrap interval on RMSE
- **occupancy AP and best-threshold IoU**: reconstruction quality, threshold-free
- **parameter count**: total and trainable, H3's efficiency claim needs it
- **the paired bootstrap against `baseline_cnn`**: not the two point estimates

And three things that are not in the JSON and must be recorded by hand:

1. **`model.fusion.distance_scales()` after training.** Per-head, per-block γ. If
   it collapses toward zero the distance bias is not being used, and that is a
   finding about H2 regardless of which way the RMSE goes.
2. **Seconds per epoch and total GPU hours** per condition.
3. **Any run that failed or was skipped**, and why. A gated backbone leaving a
   hole in the factorial is a result about availability, not a gap to hide.

---

## Afternoon, the GPU work

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

The full 120 epochs, not the 60 the 8 GB plan called for, on this card you can
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
solution rather than at random. Check the fold errors have actually flattened,
if they are still falling at 60, raise it, because you now have the headroom.

Time this. It sets whether step 6 fits before you go home.

### 6. The trained factorial (~2.5–3 h with these settings)

```bash
python -m ggssvt.cli factorial --train --backbones cnn dinov2 --variant base --epochs 120 --finetune-epochs 60 --workers 8 --batch-size 2 --device cuda
```

Four cells: geometric×cnn, geometric×dinov2, sam3d×cnn, sam3d×dinov2. It prints
the same paired-effect table as the frozen probe, so the two are directly
comparable, **and that comparison is itself a result.** If the trained factorial
reproduces the probe's sign flip (SAM3D alone hurts, SAM3D given DINO helps), the
finding survives moving from pooled descriptors to the full model. If it does
not, the probe was measuring an artefact of pooling.

Add `dinov3` to `--backbones` only if you have both the access and the hours.

---

## Pose-free reconstruction, the registration check

> **Read [POSEFREE.md](POSEFREE.md) first.** All three backends are installed and
> their APIs verified against the real code; the install instructions that were
> here before were wrong for two of the three (DUSt3R and MASt3R ship no
> `setup.py`, so `pip install -e .` fails). That document has the working
> dependency set, the Python 3.11 constraint that `open3d` imposes, and the run
> order, Fast3R first, because it is twenty times cheaper and exercises the same
> comparison code.

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
bound on almost every specimen, so the registration is the least verified
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
**not been run against the real weights**: no GPU was available where they were
written. The maths they feed is tested; the adapters are not. Watch the first run
for import errors and convention drift rather than assuming quiet success.
`sanity_check_result` warns if the cameras stop facing the scene, which is what a
convention change looks like.

---

## Evening, Nerfstudio, in the other environment

```bash
conda activate cropcraft
```

The vendored copy in `nerfstudio/` does not import: `.gitignore` line 1 is
`data/`, which matched `nerfstudio/nerfstudio/data/` and kept the dataparsers out
of the repository. Install upstream instead:

```bash
pip install nerfstudio
```

**Export the poses first, `ns-train` has nothing to read without them.** This
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
measurement of how wrong the registration is, the single most valuable number
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

## End of day, collect the results

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
these if any registration looks suspect, if the red axis is not on the plant
stem in all twelve frames, every number for that specimen is meaningless.

---

## Every command

| Command | Needs | Roughly |
|---|---|---|
| `inspect` | nothing | seconds, dataset audit, no computation |
| `access` | network | seconds, HuggingFace account and model access |
| `preprocess` | dataset | 2–3 min per segmenter |
| `baselines` | cache | seconds, LOOCV baseline table |
| `mesh` | cache, scikit-image | ~1 min including the comparison |
| `gallery` | cache | ~2 min, contact sheets, PLY, HTML viewer |
| `visualise` | dataset | seconds per specimen |
| `dino-probe` | cache, network | 5–15 min depending on variant |
| `factorial` | both caches | ~10 min frozen; hours with `--train` |
| `pretrain` | cache, **GPU** | ~25 min at 120 epochs |
| `loocv` | cache, checkpoint, **GPU** | ~40 min at 60 fine-tune epochs |
| `experiment` | cache, **GPU** | one pretrain + one LOOCV per backbone |
| `nerfstudio` | dataset | seconds, writes `transforms.json` |
| `dashboard` | cache, mesh cache | ~1 min, the walkthrough page |
| `posefree` | cloned repos, **GPU** | minutes per specimen per method |
| `mesh` | cache, scikit-image | ~1 min including the comparison |
| `ggssvt.campaign` | cache, **GPU** | 8–15 h depending on plan |
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
