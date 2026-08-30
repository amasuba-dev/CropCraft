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
