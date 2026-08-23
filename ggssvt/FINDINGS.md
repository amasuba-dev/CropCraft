# Everything run, and what it means

A record of every experiment executed, what it showed, and what follows for the
research. Companion to [RESEARCH_STATUS.md](RESEARCH_STATUS.md), which maps this
onto the proposal's questions and hypotheses.

> **Superseded in places by [RERUN_V_BATCH.md](RERUN_V_BATCH.md).** V001–V008
> were added with *measured* pot weights and the pipeline re-run on 36
> specimens. Two results reversed: 3D geometric features no longer beat
> image-only regression, and the batch confound fell from R² 0.887 to 0.697.
> Where a number below is marked n=28 or n=30, the n=36 value in that document
> is the current one.

**The headline, stated once.** The pipeline is built, validated and reproducible
across two machines. The GG-SSVT model **has not been trained** — every number
here comes from the geometry pipeline, frozen pretrained features, or classical
baselines. And a batch confound caps what any of the biomass numbers can claim.

---

## 1. Data audit

| | |
|---|---|
| Specimens | 38 captured, **36 usable** (geometric) — was 30/28 before V001–V008 |
| Views | 12 per specimen, 30° apart, dual Kinect v2, 512×424 RGB-D |
| Species | Eucalyptus ×28, Mango ×10, Xylem ×1 (excluded, 2 views) |
| Mass range | 0.20 – 2.35 kg fresh |
| Pot mass | **measured** for V001–V008, estimated for the rest (biased −10.9%) |

### Problems found in the capture set

**Two camB naming conventions.** `collect_specimen.py` names each camB file with
*camA's* step angle (`camB_000`…`camB_150`) while `dataset/README.md` documents
`camB_180`…`camB_330`. Both are present in the data. Taking the filename
literally places camB on top of camA and **silently collapses the 12-view rig
into a 6-view one**. Resolved in `data/naming.py`; the collection script should
be fixed at source.

**No calibration exists.** `dataset/calib` is entirely empty — no ChArUco
intrinsics, no per-day `rig_positions.json`. `make_transforms.py` has never had
an input.

**Ground truth caveats.** Every `pot_weight_source` reads `estimated`. The target
is as-collected *fresh* mass, not oven-dry above-ground biomass. E008's species
carries a stray apostrophe. E001 has an orphan 13th image outside its manifest.

**Tall specimens are truncated.** At the ~1 m working radius the vertical field
of view reaches about 1.15 m; E011–E020 run off the top of frame, so their
carved volumes are underestimates.

---

## 2. Calibration-free rig registration

Because no calibration was captured, extrinsics are recovered from the depth:
RANSAC floor plane per view (tilt, roll, camera height), a subject-axis
hypothesis chosen by **cross-view agreement**, then azimuth refinement by
coordinate descent.

**Result:** mean multi-view agreement 0.625, surface coverage 0.788. Camera
heights recover consistently to ~3 cm across a sweep, an independent check the
fit is working. Refinement raises coverage substantially — E011 0.29 → 0.48,
M001 0.16 → 0.43.

**One failure worth recording.** An earlier version picked the strongest
*single-view* subject candidate and locked onto background structure a metre
behind the plant on several specimens — producing a confident, internally
consistent, completely wrong registration. It was caught only by projecting the
world axis back into the images and looking. Consensus-based selection fixed it.
**Visual verification caught what every numerical diagnostic missed.**

**The standing limitation:** azimuth corrections saturate the ±8° search bound on
**25 of 30 specimens**, so the true placement error is at least that and possibly
more. This is the least verified assumption in the entire pipeline.

---

## 3. Biomass comparison — the core result

> **Superseded.** Re-run on 36 specimens with corrected targets the ordering
> inverts, direct 2D leading at 0.469 kg / R² 0.279 — but **neither ordering is
> statistically resolved**, and the one below never was either (paired bootstrap
> [−0.168, +0.099]). See [RERUN_V_BATCH.md](RERUN_V_BATCH.md) §4.

Leave-one-out, 28 specimens, identical protocol for every method.

| method | RMSE | MAE | MARE | R² |
|---|---|---|---|---|
| **mesh geometry** | **0.359 kg** | 0.269 | 27.7% | **0.613** |
| geometric features (voxel) | 0.397 kg | 0.305 | 32.8% | 0.526 |
| direct 2D (no 3D) | 0.440 kg | 0.336 | 39.9% | 0.419 |
| mean predictor | 0.598 kg | 0.511 | 62.5% | −0.075 |
| volume allometric | 0.622 kg | 0.529 | 56.9% | −0.162 |
| canopy area allometric | 0.642 kg | 0.505 | 52.1% | −0.236 |

~~**Reconstruction beats pixels** — 0.397 against 0.440.~~ **Withdrawn.** The
gap was never statistically resolved: paired bootstrap 95% CI [−0.168, +0.099].
It was reported as a finding without the interval that decides it. On the n=36
set the point estimate goes the other way (0.544 against 0.469) and is equally
unresolved, and the direction flips again under a different feature-whitening
choice. The proposal's third research question is answered by the
implied-density diagnostic below instead, which is a measurement rather than a
difference in means.

**Volume allometry is worse than the mean.** Carved hull density varies ~10×
between bushy mango and thin eucalyptus, so one volume-to-mass law cannot span
both. This is the empirical case for a learned density.

**Surface area does not beat volume.** The mesh arm was built to test the
hypothesis that leaf mass scales with area; the area law is the worst method
tried. **A visual hull's surface is envelope area, not leaf area** — 12 views at
12 mm voxels cannot resolve leaves. This is a mechanism, not a tuning failure,
and it generalises to any hull-based method at this resolution.

**And the same is true of its volume.** Dividing measured mass by reconstructed
above-ground volume gives an implied bulk density; fresh tissue is 300–900 kg/m³.
Only **8 of 36** specimens land in a generous 200–1000 band. Twenty-five imply
*less* — the hull has enclosed the air between leaves, all ten Mango at 26–77
kg/m³. Three imply more, meaning thin stems were never carved at all. This is the
mechanism behind every weak biomass number in this document, now measured rather
than inferred: `ggssvt/eval/plausibility.py`.

---

## 4. The batch confound — the finding that caps everything

Leave-one-feature-out on the mesh set showed **height** carrying the result
(removing it costs 0.051 kg; removing canopy area costs 0.001). Chasing that
down:

| batch | n | mean mass | character |
|---|---|---|---|
| E001–E010 | 10 | 0.538 kg | small; reconstruct as mostly pot |
| E011–E020 | 8 usable | 1.844 kg | tall thin saplings |
| **V001–V008** | **8** | **1.138 kg** | **added later; sd 484 g, spans both** |

**Batch membership alone explained R² = 0.887** on the two Eucalyptus batches —
more than any method achieved, including mesh geometry's 0.788 on that subset.
**Within either batch, no method reached R² = 0.2.**

**V001–V008 was collected to break this, and did.** Its range (500–1800 g)
overlaps every other batch instead of forming its own cluster, and the batch-only
R² falls to **0.744** across the three Eucalyptus batches and **0.697** across all
four. The cost is that the 3D advantage went with it — see
[RERUN_V_BATCH.md](RERUN_V_BATCH.md) §4. That is the correct trade: the earlier
comparison was measuring size-class separation.

Every model is recovering *which size class* a plant belongs to — tall-and-sparse
versus short-and-solid — not estimating mass among comparable plants.

**This does not invalidate the pipeline.** It caps the claim. The defensible
statement is *"reconstructed geometry separates plant size classes"*; the
proposal's *"estimates biomass"* is not yet supported.

**The fix was data, and it worked.** A capture batch spanning a continuous mass
range within one species — which is what V001–V008 turned out to be. More
specimens of the two original clusters would have reinforced the confound rather
than broken it.

---

## 5. DINO backbones

Frozen-feature linear probe, LOOCV, 28 specimens, PCA and standardisation fitted
inside each fold.

| condition | RMSE | R² | dims |
|---|---|---|---|
| no DINO (geometry) | 0.358 kg | 0.616 | 7 |
| DINOv2-small | 0.335 kg | 0.663 | 768 |
| **DINOv2-base** | **0.295 kg** | **0.738** | 1536 |

Paired: **ΔRMSE −0.062 kg, CI [−0.160, +0.036], p ≈ 0.22 — not significant.**
The direction is consistent and scales with backbone size; the sample cannot
resolve it. Adding geometry features on top of DINO changes nothing
(0.295 → 0.296).

**DINOv3** is now access-approved but has not been run.

---

## 6. SAM3D segmentation, and the factorial

SAM (ViT-B) prompted from the registered geometry, with three consistency rules:
3D gating to the working cylinder, a prompt-box coverage guard, and reverting
views whose masks disagree with the rest in 3D.

**Effect on the n=38 set:** 96.7% of views accepted, 17.0% of subject pixels
changed. Multi-view agreement **+0.020**; surface coverage **−0.065**; usable
specimens **36 → 33**.

The same trade as before, at the same size: SAM makes the views agree with each
other slightly more and cover the subject noticeably less, and three specimens
fall below the gate as a result. It now drops E015, E019 and V006 on top of the
E012/E016 that the geometric gate drops too. Losing V006 matters more than the
count suggests — V is the batch that breaks the confound, so the SAM3D arm is
running on a marginally more confounded sample than the geometric arm.
*(Previous figures, n=30: 96% accepted, 15.3% removed, agreement +1.9%, coverage
−6.0%, usable 28 → 26.)*

### The 2×2 factorial (26 shared specimens)

|  | no DINO | DINOv2-base |
|---|---|---|
| **no SAM3D** | 0.306 kg / R² 0.712 | 0.302 / 0.720 |
| **SAM3D** | 0.317 / 0.690 | **0.295 / 0.732** |

| effect | ΔRMSE | 95% CI |
|---|---|---|
| DINO alone | −0.004 | [−0.065, +0.065] |
| SAM3D alone | **+0.011** | [−0.010, +0.034] |
| DINO given SAM3D | −0.022 | [−0.091, +0.055] |
| **interaction** | **−0.018** | [−0.037, **+0.000**] |

**SAM3D alone slightly hurts; SAM3D given DINO helps. The sign flips.** A
one-factor-at-a-time ablation would have concluded "SAM3D doesn't help" and
dropped it. Plausible mechanism: a tighter mask loses volume for hand-built
descriptors but removes background contamination for a pooled DINO descriptor.

Nothing is statistically resolved. **And the 26-specimen shared set is not a
random subset of the 28** — SAM3D fails the gate on E015 and E019, and dropping
them moved the control from 0.358 to 0.306 kg, a larger shift than any effect in
the table.

---

## 7. View-count ablation

| views | usable | agreement | mean above-ground hull |
|---|---|---|---|
| 3 | 17/30 | 0.372 | 133.9 L |
| 4 | 18/30 | 0.447 | 159.3 L |
| 6 | 26/30 | 0.550 | 250.3 L |
| **12** | **28/30** | **0.635** | **19.3 L** |

Usable count and agreement improve monotonically — the 12-view protocol is
justified on its own. **Below twelve views the carve is uselessly loose:**
130–250 L of above-ground hull for plants weighing at most 2.35 kg. Four views at
90° is the visual-hull minimum for a *convex* object; a plant is the opposite.

The biomass comparison across view counts is **uninformative** — only 15
specimens pass under every count, most R² are negative, and 12-vs-4 views is
ΔRMSE −0.000 kg with an interval spanning zero. Do not read the ordering.

---

## 8. Bugs found and fixed

Several would have produced confident wrong numbers rather than errors.

| Bug | Consequence had it stood |
|---|---|
| `.gitignore` `data/` matched at any depth | `ggssvt/data/` and `nerfstudio/…/data/` never committed; fresh clones fail at import |
| Carve thresholds did not scale with view count | 4-view carve returns **empty**, reported as 0/30 usable — looks like a finding |
| Subject axis from single-view candidate | Confident registration onto background a metre behind the plant |
| Token anchors averaged background pixels | Anchors dragged off the specimen, corrupting the distance bias |
| Decoder used self-attention over concatenated queries | O((Q+N)²) — several GB where O(Q·N) was needed |
| Decoder MLP not gain-corrected for GELU | Signal attenuated 85× at init; near-constant output |
| HF access checked `.gated` not `auth_check` | DINOv3 would stay skipped **after** approval arrived |
| 6-connectivity in component cleanup | Diagonal stems severed; 0.3 m of plant amputated on E002 |

---

## 9. Built but not yet run

All of this needs the lab GPU.

| | Status |
|---|---|
| GG-SSVT training (`pretrain`, `loocv`) | Implemented, tested, never run |
| Geometry-grounding ablation (`--no-geometry`) | Implemented, tested, never run |
| Trained 2×2 / 2×3 factorial | Implemented, never run |
| DINOv3 | Access approved, never run |
| DUSt3R / MASt3R / Fast3R | Comparison maths verified synthetically; adapters written, **never executed against real weights** |
| Nerfstudio splatfacto | `transforms.json` exported for all 30; never trained |

---

## 10. What to change in the research scope

### Change

**Reframe the biomass claim.** "Reconstructed geometry separates plant size
classes" is supported. "Estimates biomass" is not, and an examiner who checks
the batch structure will find this in minutes. Saying it first is far stronger
than being asked.

**Keep H3, restated** — see [HYPOTHESIS_3.md](HYPOTHESIS_3.md). An earlier draft
recommended dropping it on the grounds frequency grounding was never implemented;
that was wrong. The Fourier positional encoding is exactly what the proposal's
wording names, and once measured H3 has four of six sub-claims already
established, including a counter-intuitive one with a published precedent to cite
against.

**Split H1 and H4.** Each bundles separable claims; only some have evidence.

**Fix two factual errors in the proposal abstract:** the sensor is Kinect v2, not
Intel RealSense; the target is fresh mass, not oven-dry AGB.

### Reconsider

**The dataset is the binding constraint, not the method.** Two clusters of
similar plants at n=28 cannot support a biomass estimation claim regardless of
architecture. One more capture campaign spanning a continuous mass range within
one species is worth more than any modelling work currently planned.

**Inverse procedural modelling is the principled route past the leaf-area
ceiling.** The area hypothesis failed because a hull's surface is envelope area.
CropCraft's approach — fitting a biologically plausible parametric model whose
leaves are explicit — makes leaf area a *model parameter* rather than something
the sensor must resolve. It needs a mango/eucalyptus morphology model, which is
substantial, but it is the direction that addresses the actual obstacle.

### Report readily, today

These stand on evidence already collected, with intervals:

1. **Calibration-free rig registration** from depth alone — method, diagnostics,
   and the visual-verification failure case. A contribution the proposal did not
   anticipate.
2. **Reconstruction beats direct 2D regression** (0.397 vs 0.440) — RQ3's first
   half, on own data.
3. **Volume allometry fails across morphologies** (R² = −0.162) — motivates a
   learned density.
4. **Surface area does not beat volume, and why** — envelope area is not leaf
   area. A mechanism that generalises to any hull-based method.
5. **View-count requirement** — monotone degradation, and 12 views justified
   against the 4-view protocol used previously.
6. **The batch confound** — reported as a limitation, it is a methodological
   contribution about evaluating small phenotyping datasets.
7. **The F-score / voxel-IoU gap** — implemented and demonstrated on a synthetic
   shell (F-score 1.0 at IoU 0.58); this is Paper 1's thesis.

### Cheap data fixes worth doing before the next capture

- Weigh a sample of empty pots; record real pot heights per specimen
- Capture the ChArUco sequence `dataset/README.md` already specifies
- Step the rig back for tall specimens, or add a raised second tier
- Fix the camB naming in `collect_specimen.py`
- Record a continuous mass range within one species
