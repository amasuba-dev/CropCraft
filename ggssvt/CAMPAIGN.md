# The training campaign

Seven runs, eight to ten hours, and the only thing standing between this project
and an answer to H1, H2 and H3. Everything else in the programme has run.

This document is the campaign specifically. [RUNBOOK.md](RUNBOOK.md) is the whole
pipeline; if the caches do not exist yet, start there.

---

## Install

The campaign needs the full requirements, not the subset that gets the classical
arm working. On a fresh machine:

```bash
conda env create -f ggssvt/environment.yml
```

```bash
conda activate ggssvt
```

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

```bash
pip install -r ggssvt/requirements.txt
```

Torch is installed separately and **must** come from the cu121 index. Conda
resolves `torch` to a CPU build, which does not fail: it runs everything at
roughly a hundredth of the speed and you find out eight hours later.

### What each requirement is for

| package | needed by | fatal if missing? |
|---|---|---|
| `numpy`, `scipy` | everything | yes |
| `torch`, `torchvision` | the campaign, DINO, SAM | yes |
| `transformers >= 4.56` | DINOv2, DINOv3, SAM model classes | yes for those arms |
| `huggingface_hub`, `safetensors` | fetching gated weights | yes for those arms |
| `pillow` | every figure and the gallery | yes |
| `scikit-image` | the mesh arm only | no, `cli mesh` says so |
| `scikit-learn` | alternative regressors only | no, preflight says so |
| `pytest`, `ruff` | development | no |

`transformers >= 4.56` is the reason this environment cannot be merged with the
Nerfstudio one: Nerfstudio pins Python 3.8, and 4.56 will not install on it.

### Verify before committing a night

```bash
python -m ggssvt.cli preflight
```

Checks torch and CUDA, the ground truth, optional dependencies, proxy syntax,
the HuggingFace account and each gated repo, and free disk. Findings are `fatal`,
`degraded` or `note`, and it exits non-zero only on fatal. Every environment
failure this project has hit was knowable here in seconds and was instead found
between forty minutes and eight hours in.

Two specific things to look at in its output:

- **`HF_TOKEN` set.** It overrides the cached login, so if it points at the wrong
  account no amount of `hf auth login` will fix it.
- **`dinov3` degraded.** The DINOv3 cells will skip themselves. That is not
  fatal, and `run_all` re-runs the probe on its own once the grant lands.

---

## Before you start

Two things, ten minutes, both of which change what the campaign means.

**1. The probe's control must be repaired.** At n=38 it scored R² −26.997
against +0.312 at n=36. One specimen with a 190 L hull does that; see
[FINDINGS.md](FINDINGS.md) section 7h. Until it is fixed, `h1_dinov2` has
nothing to be compared against.

```bash
python -m ggssvt.cli dino-probe --plants $(python -c "import sys;sys.path.insert(0,'.');from pathlib import Path;from ggssvt.data.preprocess import usable_plant_ids;print(' '.join(p for p in usable_plant_ids(Path('work_dirs/ggssvt/cache')) if p!='V010'))")
```

The CNN control should return near **R² +0.31**. If it does not, stop: something
other than V010 is wrong, and eight hours would be spent on a broken reference.

**2. The smoke plan must run.** Four specimens, two epochs, five minutes.

```bash
python -m ggssvt.campaign --plan smoke --device cuda
```

The smoke plan writes to `work_dirs/ggssvt/campaign_smoke/`, not to
`campaign/`. That separation matters: `run_all` decides the campaign has already
run by looking for `campaign/summary.txt`, so a five-minute plumbing check
sharing that path would mark eight hours of training as done and the unattended
run would skip it and report success. It did, until commit `4c8b0f1`.

**It is expected to report `blocked`.** Two epochs cannot train an occupancy
field, so the volume integral is meaningless and the gate says so. What you are
checking is that the loop, the checkpointing and the resume all work. A run that
predicts around 200 kg for a 0.5 kg plant is the head computing
`density_prior x volume` with an untrained occupancy field at roughly 0.28, which
is arithmetic rather than a bug.

---

## Running it

```bash
nohup python -m ggssvt.run_all --device cuda > run.log 2>&1 &
```

`nohup` matters: a dropped SSH session has already killed one run of this
project's factorial, twice.

To run only the campaign, without the rest of the programme:

```bash
nohup python -m ggssvt.campaign --plan core --device cuda --workers 8 --batch-size 2 > campaign.log 2>&1 &
```

`--batch-size 2` assumes 16 GB. Use 1 below 12 GB, 3 above 20.

### The seven runs, and what each one closes

| run | closes | prediction |
|---|---|---|
| `baseline_cnn` | the reference for everything below | — |
| `baseline_fused` | does a *trained* model inherit the fusion advantage the classical features got, 0.544 to 0.335? | unknown, and this is the interesting one |
| `h2_no_geometry` | **H2**, ablating geometry grounding | grounding helps, or H2 is wrong |
| `h1_dinov2` | **H1**, self-supervised ViT against the CNN stem | — |
| `h3_bands_8_freq7` | **H3**, encoding matched to the grid Nyquist, 41.7 cyc/m | matches the baseline at lower cost |
| `h3_bands_6_freq6` | **H3**, half Nyquist, 20.8 cyc/m | **should hurt** |
| `h3_bands_16_freq10` | **H3**, eight times Nyquist, 333 cyc/m | **should add nothing** |

The H3 trio is the strongest design here because the predictions are written down
before the run. If half-Nyquist hurts and eight-times adds nothing, that is a
parameter-efficiency result. If the 16-band run helps, H3 is wrong, and you learn
that cleanly rather than by argument.

### It is safe to leave

- **Resumable.** Each run writes a completion marker; a campaign that dies at 3am
  restarts and skips what finished.
- **Fingerprinted.** A run whose targets no longer match the ground truth is
  re-run rather than reused, so a stale number cannot survive a data change.
- **Seeded.** Since commit `8605451`. Before that, `TrainConfig.seed` existed and
  nothing read it, so two runs of the same command gave different weights. CUDA
  is still not bitwise deterministic, and that is a deliberate trade.
- **Gated.** A run that collapses onto the training mean, never beats the mean
  predictor, or leaves occupancy at chance is marked `blocked`, not `done`, with
  the failing check named. That matters most overnight.

---

## What each hypothesis needs, and what is missing

The seven runs are necessary and **not sufficient**. Two of the four hypotheses
name a measurement the campaign does not produce, and neither is built. Both are
listed here rather than discovered on Sunday night.

### H1: self-supervised ViTs beat CNNs, from fewer labels

Two claims, and the campaign answers one of them.

| | supplied by | state |
|---|---|---|
| ViT beats CNN | `h1_dinov2` against `baseline_cnn`, paired bootstrap | **in the core plan** |
| DINOv3 variant | `h1_dinov3` | **only in `--plan full`** |
| reaches accuracy from fewer labels | `cli label-efficiency` | **built, and already run** |

```bash
python -m ggssvt.cli label-efficiency
```

Seconds on CPU, because it reads label efficiency off the frozen representation
rather than re-fine-tuning: the backbone never sees a mass, so subsampling labels
changes only the head's training set and stage 1 is not repeated.

First result, [FINDINGS.md](FINDINGS.md) 7j: **DINOv2 reaches the geometric
baseline's full-label accuracy with 8 labels against the baseline's 32.** Two
caveats travel with it. The geometric curve's 8-label point is a numerical
failure rather than a poor fit, flagged and excluded. And both conditions reduce
to 8 principal components, so 1536 dimensions down to 8 is a different operation
from 7 down to 7, and part of the advantage may be that projection regularising
rather than the representation carrying more. A matched-capacity control would
separate them; it does not exist yet.

### H2: geometry grounding gives viewpoint consistency and better reconstruction

Three claims, and the campaign answers one and a half.

| | supplied by | state |
|---|---|---|
| grounding helps | `h2_no_geometry` ablation | **in the core plan** |
| the geometry bias is doing something | `cli attention`, mean distance scale | **built** |
| higher consistency across viewpoints | `cli viewpoint` | **built, and already run** |

```bash
python -m ggssvt.cli viewpoint
```

About 25 minutes on CPU. Holds each view out in turn, reconstructs from the
other eleven, and scores against what the sensor measured at the withheld
azimuth. `reproject` does not do this: it scores against the views a
reconstruction was *built from*, which is self-consistency.

Baseline established, [FINDINGS.md](FINDINGS.md) 7k: the carve scores **0.4070
in-sample against 0.3896 held out, a gap of 4.3 per cent** over 432 held-out
views. Read that as a floor rather than a triumph. Eleven silhouettes nearly
determine a hull, so a small gap is close to guaranteed, and the same hull can
only weigh 8 of 36 plants. Viewpoint consistency and fidelity are independent
and this measures the first.

**Its value is comparative, and the campaign is what supplies the comparison.**
Run the same protocol against a geometry-grounded model and against
`h2_no_geometry`, and the question becomes whether grounding shrinks a gap that
is already small. 4.3 per cent is the number those runs have to beat.

### H3: frequency and geometry grounding together improve parameter efficiency

The best-specified of the four, and fully covered.

| | supplied by | state |
|---|---|---|
| what the target's spectrum contains | `eval/frequency.py` | **built** |
| encoding matched to the grid Nyquist | `h3_bands_8_freq7` | in the core plan |
| below Nyquist should hurt | `h3_bands_6_freq6` | in the core plan |
| far above should add nothing | `h3_bands_16_freq10` | in the core plan |

Nothing missing. Run the frequency analysis first so the Nyquist argument is
measured against the data rather than asserted from the voxel size.

### H4: robustness to occlusion, noise and sparse sampling

**Answered, and it needed no GPU.** See [FINDINGS.md](FINDINGS.md) section 7i.
Sparse sampling from the view ablation, noise and occlusion from
`cli robustness`. Robust to depth noise at four times the sensor's own
characteristic; not robust to sustained occlusion, which destroys 33 of 36
reconstructions at 50%.

---

## Getting the learned arm to 90% before Monday

The learned arm is currently at zero. Ninety per cent means H1, H2 and H3 each
have an experiment that ran and a number that can be reported with an interval.

**Friday evening.** The two pre-checks, then launch. The campaign is the long
pole and everything else fits around it.

```bash
python -m ggssvt.cli preflight && python -m ggssvt.cli dino-probe --plants $(python -c "import sys;sys.path.insert(0,'.');from pathlib import Path;from ggssvt.data.preprocess import usable_plant_ids;print(' '.join(p for p in usable_plant_ids(Path('work_dirs/ggssvt/cache')) if p!='V010'))")
```

```bash
python -m ggssvt.campaign --plan smoke --device cuda
```

```bash
nohup python -m ggssvt.campaign --plan core --device cuda --workers 8 --batch-size 2 > campaign.log 2>&1 &
```

**Friday evening, in parallel, on the CPU.** These do not touch the GPU and
close H3's measurement and H4 entirely.

```bash
python -m ggssvt.cli robustness
```

**Saturday.** The campaign should be finished. Read the summary, run the
attention read for H2, and rebuild the page.

```bash
python -m ggssvt.cli attention --plant M001 && python -m ggssvt.cli report && python -m ggssvt.cli dashboard
```

**Saturday and Sunday, in priority order, if the time exists:**

1. The **label-efficiency curve** for H1. Highest value of the three, because it
   is the half of H1 that makes the method self-supervised rather than merely
   transformer-based.
2. **Held-out-view consistency** for H2. Second, because without it H2 rests on
   an ablation and a single attention statistic.
3. `--plan full`, which adds `h1_dinov3` and the two SAM3D cells. Lowest value:
   it broadens rather than deepens, and DINOv3 is another backbone answering a
   question DINOv2 already answers.

**What 90% looks like on Monday.** H3 complete, H4 complete, H1 answered on the
comparison and open on label efficiency, H2 answered on the ablation and open on
viewpoint consistency. That is three of four hypotheses with real numbers, which
is the difference between a results chapter and a plan.

**What would make it 100%** is the two missing experiments above, and those are
worth a week rather than a weekend. Do not compress them into Sunday night; an
experiment built in a hurry against a deadline is how the reciprocity control got
skipped, and that produced a convincing false positive.

---

## After the campaign: a neural field as a third operator

Not part of the Monday plan, and deliberately sequenced last: it closes no
hypothesis, it extends an operator comparison that is already resolved, and it
competes with the campaign for the card.

```bash
python -m ggssvt.cli nerfstudio          # export transforms, already written
# then, in the cropcraft environment, ns-train per specimen
python -m ggssvt.cli neural-field        # read the fields back, sweep the threshold
```

A neural field gives a **density**, not occupancy, and turning it into a volume
needs a threshold with no physical calibration. That is precisely the free
parameter C1 exists to remove, so `neural-field` does not choose one: it sweeps
21 values across five orders of magnitude and asks whether *any* of them makes
each reconstruction able to weigh its plant.

Both outcomes are results. If no threshold works, the envelope finding
generalises from silhouette hulls to neural fields, which is a broader claim than
the project currently makes. If one works for every specimen, that value is a
measured density calibration for this sensor and subject, which nobody publishes
because nobody has a mass to calibrate against. A threshold that works
per-specimen but not across the set is neither: that is a fitted parameter, and
the report says so rather than quoting the best one.

**Do not modify `nerfstudio/`.** It is unmodified upstream, and the adaptation
belongs on this side. The one thing that goes wrong silently is coordinates:
Nerfstudio re-centres and rescales the scene at load, so the metric voxel grid
has to be mapped through `dataparser_transforms.json`. Skipping that puts every
query in the wrong place and looks like an empty reconstruction rather than a
mis-registration, which is the same class of mistake as the OpenCV to OpenGL
convention in the export.

---

## When it finishes

```bash
python -m ggssvt.cli attention --plant M001
```

For H2, two numbers. **Neighbour preference** above 1 means the fusion prefers
views whose frusta overlap, which is what a model that has learned the rig should
do. **Mean distance scale**: if it has not moved off its initialisation, the
geometry bias is inert and the cross-view attention is ordinary self-attention
wearing a hat. That is evidence *against* H2, and it is worth reporting as such.

```bash
python -m ggssvt.cli report && python -m ggssvt.cli dashboard
```

Rebuilds the tables and the project page from whatever actually ran.

### Reading the summary honestly

Every difference needs a paired bootstrap interval before it is a result. The
project's convention is that an interval spanning zero is reported as
**unresolved**, not as a weak finding. At n=38 in two morphology clusters with a
batch confound, expect several.

**Plan for H1 coming back unresolved.** A representation claim on this dataset is
a hard ask, and every resolved result so far has come from the reconstruction
side rather than the model side. That outcome is not a failed thesis: *the
reconstruction operator, not the representation, is the limiting component* is a
real finding, it is what the evidence already points at, and section 7 of the
feasibility results is written so that result strengthens it.

---

## If something goes wrong

| symptom | cause | fix |
|---|---|---|
| everything is slow, no error | CPU torch wheel | `preflight`; reinstall from cu121 |
| CUDA out of memory | batch size | `--batch-size 1` |
| every DINOv3 cell skips | account not granted | `cli access`; check `HF_TOKEN` |
| run marked `blocked` | the gate caught a run that learned nothing | read the named check before re-running |
| session dies mid-run | no `nohup` | relaunch; it resumes |
| results look stale on the page | artefact predates the cache | the Experiment log flags it red |
