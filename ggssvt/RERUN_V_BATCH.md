# The V001–V008 re-run: what the measured pot weights changed

Eight specimens were collected on 2026-08-14 and never processed. Their pots were
weighed after the shoots were cut off, so for the first time in this project the
above-ground mass of a specimen is a **measurement** rather than a total minus an
estimate. Everything below is the pipeline re-run end to end on the enlarged,
corrected set.

**Headline: two results reversed, one new diagnostic explains both, and the new
batch is the most useful one collected so far.**

---

## 1. The estimated pot weights were biased, not noisy

| id | total g | pot estimated | pot **measured** | net estimated | net **measured** |
|---|---|---|---|---|---|
| V001 | 25400 | 21900 | 24400 | 3500 | **1000** |
| V002 | 23700 | 21500 | 23200 | 2200 | **500** |
| V003 | 20500 | 17200 | 19400 | 3300 | **1100** |
| V004 | 32200 | 26800 | 30400 | 5400 | **1800** |
| V005 | 22400 | 18600 | 20600 | 3800 | **1800** |
| V006 | 22300 | 18800 | 21300 | 3500 | **1000** |
| V007 | 33800 | 28400 | 32500 | 5400 | **1300** |
| V008 | 22300 | 18950 | 21700 | 3350 | **600** |

Net shoot mass was overstated by **2.1× to 5.6×**.

The useful part is that the error is systematic:

> **pot estimated / pot measured = 0.891, sd 0.018** across all eight.

A consistent 10.9% under-estimate of pot mass with almost no scatter. That is a
calibration, not a bias to be lamented — it puts an error bar on the E and M net
masses, which are **still estimates**:

| group | net as recorded | if the same −10.9% bias applies |
|---|---|---|
| Mango | 1022 g | 911 g (−11%) |
| E001–E010 | 538 g | 409 g (**−24%**) |
| E011–E020 | 1750 g | 1540 g (−12%) |

E001–E010 is worst affected because the pot is most of the total. Carry this as a
stated uncertainty on every result involving those batches, and weigh the pots
directly on the next capture — it is a scale and ten minutes.

---

## 2. A single pot height could not survive this batch

V pots weigh **17–32 kg** against **0.7–2.2 kg** for E001–E020. The fixed
`POT_HEIGHT_M = 0.28` left a slab of pot counted as plant: V001 reported 15.4 L
of above-ground volume for a 1.0 kg shoot, an implied density of 65 kg/m³ where
fresh tissue is 300–900.

`ggssvt/geometry/pot.py` now estimates the rim per specimen from the occupancy
profile. A rim is a **step**, not a slope, so the test is local — the median
cross-section just below a candidate height against the median just above.

**It refuses more often than it answers, and that is the useful part.**

| batch | n | confident | rim height |
|---|---|---|---|
| E001–E010 | 10 | **1** | — |
| E011–E020 | 8 | 8 | 0.240 ± 0.008 m |
| Mango | 10 | 8 | 0.468 ± 0.055 m |
| V001–V008 | 8 | 8 | 0.354 ± 0.036 m |

E001–E010 carve as single cones — 645 voxels at the floor decaying evenly to 140
and then stopping, with no pot/plant boundary anywhere. Nine of ten now return
`confident=False` and fall back to the constant. That is the truthful answer:
**those reconstructions do not separate plant from pot at all**, which has been
suspected throughout and is now measured.

An earlier version of this estimator thresholded against the widest slice and
returned three different confident rims — 0.432, 0.336, 0.432 m — for three
specimens of the same batch. Worth remembering as a caution: a detector that
always answers is not the same as a detector that works.

---

## 3. The new diagnostic: can the volume physically weigh that much?

Measured mass ÷ reconstructed above-ground volume gives an implied bulk density,
which is checkable against physics rather than against a baseline. Fresh
above-ground tissue is roughly 300–900 kg/m³.

| group | n | plausible (200–1000) | envelope (too light) | missing (too heavy) | median kg/m³ |
|---|---|---|---|---|---|
| E001–E010 | 10 | 1 | 9 | 0 | 127 |
| E011–E020 | 8 | 3 | 2 | 3 | 863 |
| Mango | 10 | **0** | **10** | 0 | **52** |
| V001–V008 | 8 | **4** | 4 | 0 | 248 |
| **all** | **36** | **8** | **25** | **3** | **117** |

**Only 8 of 36 reconstructions are the right size to weigh what the plant
weighs.** Twenty-five are envelopes: the visual hull has enclosed the air between
leaves and branches, so the quantity being regressed is the canopy envelope and
not the plant. Every Mango specimen sits at 26–77 kg/m³ — one to two orders of
magnitude short, which no regressor repairs. Three imply *more* than plausible,
meaning thin stems were never carved; E019 implies 22,569 kg/m³ because
essentially none of it was reconstructed.

This is the same failure the canopy-area hypothesis hit, now measured on volume
as well as surface, and it accounts for the whole pattern of biomass results.
**It is the most useful single number this project has produced**, because it
converts "the fit is poor" into "the reconstructed object is the wrong kind of
object, by this factor, for these species".

---

## 4. Two results reversed

### 3D geometric features no longer beat image-only regression

| set | rim | 3D RMSE / R² | 2D RMSE / R² | winner |
|---|---|---|---|---|
| E+M only (n=28) | constant 0.28 | 0.397 / +0.526 | 0.440 / +0.419 | 3D |
| E+M only (n=28) | per-specimen | 0.434 / +0.435 | 0.440 / +0.419 | 3D, barely |
| **all incl. V (n=36)** | constant 0.28 | 0.512 / +0.140 | 0.469 / +0.279 | **2D** |
| **all incl. V (n=36)** | **per-specimen** | **0.544 / +0.030** | **0.469 / +0.279** | **2D** |

Both changes push the same way; **V dominates**. Full leave-one-out on n=36:

| method | RMSE kg | MAE | MARE% | R² |
|---|---|---|---|---|
| **direct 2D** | **0.469** | 0.370 | 42.8 | **+0.279** |
| mesh geometry | 0.507 | 0.367 | 37.4 | +0.157 |
| geometric features | 0.544 | 0.420 | 43.8 | +0.030 |
| mean | 0.568 | 0.477 | 57.8 | −0.058 |
| volume allometric | 0.592 | 0.495 | 53.0 | −0.150 |
| canopy area allometric | 0.598 | 0.460 | 47.3 | −0.170 |

**Every 3D-derived method now loses to image-only regression.** Section 3 says
why: 28 of 36 reconstructions are not the right size for their mass, so the
volume feature cannot carry mass information.

`test_pipeline.py` now asserts this direction, with the reason, instead of the
direction that was hoped for.

### The batch confound weakened — because V was designed to break it

| set | batch-only R² |
|---|---|
| Eucalyptus, 2 batches (the old n=18) | **0.887** |
| Eucalyptus, 3 batches incl. V (n=26) | 0.744 |
| all four batches (n=36) | **0.697** |

| batch | n | mean | sd | range |
|---|---|---|---|---|
| E001–E010 | 10 | 538 g | 114 | 400–700 |
| E011–E020 | 8 | 1844 g | 348 | 1350–2350 |
| **V001–V008** | 8 | **1138 g** | **484** | **500–1800** |
| Mango | 10 | 1022 g | 284 | 560–1470 |

V has the **widest within-batch spread of any batch** (sd 484 g, CV 43%) and its
range **overlaps every other batch**. That is exactly what the study was missing:
previously the batches sat at separate sizes, so a feature that read batch
membership scored well without measuring mass. V removes that shortcut.

**This is why the 3D advantage disappeared — it was substantially the confound.**
Losing a result to a better-designed sample is a good trade, and it is a
defensible thing to write up: the earlier n=28 comparison was measuring size-class
separation, and the n=36 comparison is measuring mass estimation.

---

## 5. H3 unchanged in direction, strengthened in evidence

V re-run through the spectral characterisation:

| group | n | 95% bandwidth | high-freq share |
|---|---|---|---|
| E001–E010 | 10 | 29.3 ± 2.0 | 0.128 ± 0.010 |
| E011–E020 | 8 | 40.6 ± 2.3 | 0.254 ± 0.045 |
| Mango | 10 | **41.7 ± 0.0** | 0.273 ± 0.020 |
| **V001–V008** | 8 | **41.7 ± 0.0** | 0.257 ± 0.025 |

Grid Nyquist is 41.7 cycles/m. **H3c now covers 18 of 36 specimens, not 10** —
both Mango and V saturate the grid exactly. Resolution, not method, is the
binding constraint for half the dataset. The 12 mm voxel is now the most
defensible thing to change next.

---

## 6. Quality gate

**36 of 38 pass, up from 28 of 30.** All eight V specimens pass. X001 is excluded
throughout: 2 views, one specimen, a species of its own.

---

## What to do with this

**Report readily.** The pot-weight bias calibration (§1). The plausibility
diagnostic and the 8-of-36 result (§3) — this is a genuine methodological
contribution, not a negative result to bury. The confound weakening (§4).

**Reframe.** Anywhere the argument was "3D reconstruction improves biomass
estimation", it now has to be "3D reconstruction of these species by visual hull
produces canopy envelopes, which do not carry mass information; image-only
regression does better". That is a stronger dissertation chapter than the
original claim, because it is supported by a physical measurement rather than a
difference in means at small n.

**Do next, in order.**
1. Weigh pots directly for any future capture. Ten minutes, removes the largest
   single uncertainty in the ground truth.
2. Re-run the campaign on the GPU against the corrected targets — every trained
   number in `RESEARCH_STATUS.md` predates this correction.
3. Test the resolution prediction from §5: a finer voxel grid on the 18
   Nyquist-saturated specimens.
