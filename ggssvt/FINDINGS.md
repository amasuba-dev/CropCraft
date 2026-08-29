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
across two machines. The GG-SSVT model **has not been trained**, every number
here comes from the geometry pipeline, frozen pretrained features, or classical
baselines. And a batch confound caps what any of the biomass numbers can claim.

---

## 1. Data audit

| | |
|---|---|
| Specimens | 38 captured, **36 usable** (geometric), was 30/28 before V001–V008 |
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

**No calibration exists.** `dataset/calib` is entirely empty, no ChArUco
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
fit is working. Refinement raises coverage substantially, E011 0.29 → 0.48,
M001 0.16 → 0.43.

**One failure worth recording.** An earlier version picked the strongest
*single-view* subject candidate and locked onto background structure a metre
behind the plant on several specimens, producing a confident, internally
consistent, completely wrong registration. It was caught only by projecting the
world axis back into the images and looking. Consensus-based selection fixed it.
**Visual verification caught what every numerical diagnostic missed.**

**The standing limitation:** azimuth corrections saturate the ±8° search bound on
**25 of 30 specimens**, so the true placement error is at least that and possibly
more. This is the least verified assumption in the entire pipeline.

---

## 3. Biomass comparison, the core result

> **Superseded.** Re-run on 36 specimens with corrected targets the ordering
> inverts, direct 2D leading at 0.469 kg / R² 0.279, but **neither ordering is
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

~~**Reconstruction beats pixels**, 0.397 against 0.440.~~ **Withdrawn.** The
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
tried. **A visual hull's surface is envelope area, not leaf area**, 12 views at
12 mm voxels cannot resolve leaves. This is a mechanism, not a tuning failure,
and it generalises to any hull-based method at this resolution.

**And the same is true of its volume.** Dividing measured mass by reconstructed
above-ground volume gives an implied bulk density; fresh tissue is 300–900 kg/m³.
Only **8 of 36** specimens land in a generous 200–1000 band. Twenty-five imply
*less*. The hull has enclosed the air between leaves, all ten Mango at 26–77
kg/m³. Three imply more, meaning thin stems were never carved at all. This is the
mechanism behind every weak biomass number in this document, now measured rather
than inferred: `ggssvt/eval/plausibility.py`.

---

## 4. The batch confound, the finding that caps everything

Leave-one-feature-out on the mesh set showed **height** carrying the result
(removing it costs 0.051 kg; removing canopy area costs 0.001). Chasing that
down:

| batch | n | mean mass | character |
|---|---|---|---|
| E001–E010 | 10 | 0.538 kg | small; reconstruct as mostly pot |
| E011–E020 | 8 usable | 1.844 kg | tall thin saplings |
| **V001–V008** | **8** | **1.138 kg** | **added later; sd 484 g, spans both** |

**Batch membership alone explained R² = 0.887** on the two Eucalyptus batches,
more than any method achieved, including mesh geometry's 0.788 on that subset.
**Within either batch, no method reached R² = 0.2.**

**V001–V008 was collected to break this, and did.** Its range (500–1800 g)
overlaps every other batch instead of forming its own cluster, and the batch-only
R² falls to **0.744** across the three Eucalyptus batches and **0.697** across all
four. The cost is that the 3D advantage went with it, see
[RERUN_V_BATCH.md](RERUN_V_BATCH.md) §4. That is the correct trade: the earlier
comparison was measuring size-class separation.

Every model is recovering *which size class* a plant belongs to, tall-and-sparse
versus short-and-solid, not estimating mass among comparable plants.

**This does not invalidate the pipeline.** It caps the claim. The defensible
statement is *"reconstructed geometry separates plant size classes"*; the
proposal's *"estimates biomass"* is not yet supported.

**The fix was data, and it worked.** A capture batch spanning a continuous mass
range within one species, which is what V001–V008 turned out to be. More
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

Paired: **ΔRMSE −0.062 kg, CI [−0.160, +0.036], p ≈ 0.22, not significant.**
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
count suggests, V is the batch that breaks the confound, so the SAM3D arm is
running on a marginally more confounded sample than the geometric arm.
*(Previous figures, n=30: 96% accepted, 15.3% removed, agreement +1.9%, coverage
−6.0%, usable 28 → 26.)*

### The 2×2 factorial (33 shared specimens, re-run with V)

|  | no DINO | DINOv2-base |
|---|---|---|
| **no SAM3D** | 0.576 kg / R² −0.080 | **0.385 / +0.518** |
| **SAM3D** | **0.778 / −0.967** | 0.390 / +0.505 |

| effect | ΔRMSE | 95% CI | |
|---|---|---|---|
| DINO alone | −0.191 | [−0.404, +0.021] | not resolved |
| SAM3D alone | **+0.201** | [−0.017, +0.394] | not resolved |
| **DINO given SAM3D** | **−0.387** | **[−0.757, −0.039]** | **resolved** |
| SAM3D given DINO | +0.005 | [−0.007, +0.019] | no effect |
| interaction | −0.196 | [−0.393, +0.031] | not resolved |

**The first resolved effect this project has produced.** DINO features help
significantly *when the hull came from SAM3D*, and the reason is visible in the
table: SAM3D on its own is catastrophic for the hand-crafted descriptors
(0.576 → 0.778, R² −0.967, far below the mean-predictor floor), and DINO simply
does not care which segmenter produced the hull (0.385 → 0.390, an effect of
0.005 kg with an interval of ±0.013).

That asymmetry is the finding. **The hand-crafted geometric descriptors are
fragile to the segmentation; the learned image features are not.** Which follows
from §3: those descriptors summarise a volume that is a canopy envelope rather
than a plant, so perturbing the mask moves them a great deal and costs nothing
real, while DINO reads the images and never depended on the volume being
meaningful.

Read the "resolved" carefully. It is resolved partly because SAM3D + no DINO is
so bad, so it evidences *fragility of the descriptors* more than *value of DINO*.
DINO alone against neither remains unresolved at [−0.404, +0.021].

**The 33-specimen shared set is not a random subset of the 36.** SAM3D fails the
gate on E015, E019 and V006 on top of the E012/E016 the geometric gate drops.
Losing V006 matters beyond the count: V is the batch that breaks the mass/batch
confound, so both arms here run on a marginally more confounded sample than the
n=36 baselines table.

*(Previous run, 26 shared specimens: all four cells within 0.295–0.317 kg and
nothing resolved. The spread has widened enormously, which is what adding a batch
that does not share the others' size structure does to features that were reading
size.)*

---

## 7. View-count ablation

| views | usable | agreement | mean above-ground hull | **physically plausible** | median kg/m³ |
|---|---|---|---|---|---|
| 3 | 23/38 | 0.360 | 99.3 L | **1/23** | 9.8 |
| 4 | 25/38 | 0.424 | 126.5 L | **0/25** | 9.2 |
| 6 | 34/38 | 0.521 | 150.4 L | **2/34** | 15.4 |
| **12** | **36/38** | **0.608** | **10.4 L** | **8/36** | **116.8** |

Usable count and agreement improve monotonically, so the 12-view protocol is
justified on its own. But the plausibility column settles it far more sharply
than the agreement column ever could.

**At four views, zero of twenty-five reconstructions are physically capable of
weighing what the plant weighs.** The median implied bulk density is 9.2 kg/m³,
lighter than expanded polystyrene, and thirty to ninety times below fresh plant
tissue. The hulls average 126 L for plants of at most 2.35 kg. These are not
poor reconstructions, they are not reconstructions of the plant at all.

Four views at 90° is the visual-hull minimum for a *convex* object. A plant is
the opposite of convex, and every unsampled azimuth leaves a prism of empty space
uncarved. Twelve views is where the number becomes non-absurd, 10.4 L and 117
kg/m³, and even there only 8 of 36 clear the bar.

**This is the answer to "could we get away with four images?"** No, and the
reason is measurable rather than a matter of taste. It also disposes of the
biomass comparison across view counts, which was uninformative for a better
reason than small n: below twelve views there is nothing to regress against.

*(Previous figures, n=30 with a fixed 0.28 m pot cut: 17/30, 18/30, 26/30, 28/30
usable and 133.9 / 159.3 / 250.3 / 19.3 L. The hull volumes fall throughout
because each specimen's pot is now cut at its own rim.)*

---

## 7b. TSDF depth fusion, which escapes the hull

Space carving intersects silhouette cones, and Laurentini's result says the
visual hull it produces is the maximal solid consistent with those silhouettes.
That is a ceiling, not a resolution problem. A pot's rim casts no silhouette from
anywhere on a circle around it, so the carve fills it; the gap between two leaves
casts none either, so the carve fills that too. Finer voxels give a smoother
envelope, never a gap.

Depth maps are different evidence. A depth pixel does not say "the subject lies
somewhere along this ray", it says "a surface is at exactly this distance".
Integrated as a truncated signed distance field, concavities survive and
unobserved space stays unknown instead of being filled in. One depth pixel spans
3.0 mm at the working distance, so 6 mm voxels at 256 cubed are justified by the
sensor where the 12 mm carving grid was chosen to keep carving tractable.

**The reconstruction result is decisive**, and it is a measurement rather than a
fitted comparison:

| | plausible | median implied density | verdicts |
|---|---|---|---|
| carve, 12 mm hull | 8/36 | 116.8 kg/m³ | 25 envelope, 3 missing |
| **TSDF, 12 mm fusion** | **25/36** | 271.9 kg/m³ | the shipped cache |
| **TSDF, 6 mm fusion** | **31/36** | **529.2 kg/m³** | 1 envelope, 4 missing |

Two counts, because two things improve. Holding the rim fixed at the carve's estimate isolates the occupancy operator and gives **21 of 36**. Letting the rim be re-estimated from the fused profile, which is what the shipped cache does, gives **25 of 36**: a fused vertical profile has a sharper step, so the rim detector refuses less often. Both are at 12 mm; 31 of 36 is the 6 mm figure.

M001 is the clearest case: 25.79 L of hull above the rim for a 0.74 kg shoot,
against 1.18 L fused. The hull implied 28.7 kg/m³, the fusion 629. Mean coverage
is 0.12, so roughly one eighth of the working volume was ever measured and the
rest is honestly absent rather than assumed solid.

**The biomass result is real but unresolved**, which by now is the expected
outcome at this sample size:

| feature set | all 36 | Eucalyptus only (n=26) |
|---|---|---|
| carved geometry (7) | 0.544 / +0.030 | 0.717 / **−0.313** |
| direct 2D (7) | 0.469 / +0.279 | 0.520 / +0.311 |
| **TSDF geometry (7)** | **0.465 / +0.290** | **0.494 / +0.377** |
| TSDF + 2D (14) | **0.429 / +0.396** | |
| mean predictor | 0.552 | 0.626 |

Paired bootstrap, 20,000 resamples: TSDF against carved geometry is −0.079
[−0.242, +0.066] on all 36 and −0.223 [−0.498, +0.021] on Eucalyptus, and against
direct 2D it is −0.004 [−0.134, +0.133]. **Nothing resolves.** Do not report an
ordering from this table.

What is worth reporting is the sign change. **TSDF geometry is the first 3D
feature set to clear the mean-predictor floor within a species.** Carved geometry
sits at R² −0.313 against a floor of −0.082, meaning it carries no usable mass
signal; the fused features sit at +0.377. Crossing the floor is a different
statement from winning a head-to-head, and it is the one that follows from the
plausibility result rather than from a difference in means.

**The limits, stated plainly.** Twelve views leave most leaf undersides
unobserved, so this does not resolve individual leaves and is not a substitute
for a dense photogrammetric capture. The fused interior is a band one truncation
width deep behind each observed surface, not a filled solid, so its volume is a
proxy rather than the plant's volume. And 6 mm is the sensor's limit, not the
plant's: a Eucalyptus leaf is 0.3 mm thick.

Ten minutes for all 36 on one CPU core. `python -m ggssvt.cli fuse`.

---

## 7c. The reconstruction was the bottleneck, not the regressor

Everything in section 3 and section 7b pointed one way, and this is the test that
closes it. Take the seven hand-crafted descriptors unchanged, the same
leave-one-out protocol, the same 12 mm grid, the same twelve views and the same
masks. Change only the operator that turns them into occupancy.

| | carved | fused | paired bootstrap |
|---|---|---|---|
| **geometric features** | 0.544 / +0.030 | **0.335 / +0.632** | −0.209 [−0.363, **−0.066**] **resolved** |
| volume allometric | 0.592 / −0.150 | 0.469 / +0.278 | −0.123 [−0.202, **−0.034**] **resolved** |
| canopy area allometric | 0.598 / −0.170 | 0.494 / +0.201 | |
| mesh geometry | 0.507 / +0.157 | 0.486 / +0.227 | |
| direct 2D | 0.469 / +0.279 | 0.469 / +0.279 | unchanged, as it must be |
| mean predictor | 0.568 | 0.568 | |

**These are the first resolved improvements this project has produced on biomass.**
Everything before them, including the original "reconstruction beats pixels", had
an interval spanning zero. Direct 2D is identical in both columns, which is the
control: it touches no reconstruction, so it must not move, and it does not.

Two methods that sat *below* the mean-predictor floor now clear it. Volume
allometry went from −0.150 to +0.278 and canopy area from −0.170 to +0.201. That
matters more than the ordering. A single volume-to-mass law was said to be
impossible across morphologies because hull density varied tenfold between a
bushy Mango and a thin Eucalyptus; on a fused reconstruction the same law works,
because the volumes are no longer envelopes of wildly different emptiness.

**The canopy-area hypothesis partially survives.** It was declared dead in
section 3 on the grounds that a visual hull's surface is envelope area rather
than leaf area. On a fused surface it clears the floor. The mechanism claimed
there was right, and it was a statement about the instrument rather than about
the biology.

On the Eucalyptus subset of those same pooled predictions, geometric features go
from 0.610 / +0.049 to **0.350 / +0.687** against a floor of 0.626. Note that
this is the subset of a fit on all 36, not a fit on Eucalyptus alone; the
fit-within-species numbers quoted elsewhere in this document are a different
protocol and the two should not be compared.

What this does not do is make the reconstructions good. Coverage is 0.12, most
leaf undersides were never observed, and 5 of 36 fused specimens still fail the
plausibility check. The claim is narrow and it is the one the evidence supports:
**for these species at this resolution, replacing silhouette intersection with
depth integration improves biomass estimation by more than any change to the
regressor has.**

---

## 7d. DITR-style DINO lifting, and why it does not rescue E001-E010

*This is the project's one experiment on the reciprocity Malik et al. argue for
in "The three R's of computer vision" (Pattern Recognition Letters 72, 2016):
that grouping and recognition inform reconstruction and each other. Semantic
features are used to attempt a **reorganization** (plant against pot) in order to
repair a **reconstruction**. It fails, for a reason that is measured rather than
guessed, and a negative result about that interaction is still a result about
it. The third R, **recognition**, is out of scope for this study: one genus per
batch and no category task means there is nothing for a recognition claim to be
tested against. Note also that "the 3Rs" in `POSEFREE.md` refers to the pointmap
models by their shared suffix, which is a different thing entirely; this document
uses "pointmap models" for those.*


Requested by the supervisor, after Knaebel et al., who observe that 3D
segmentation largely ignores 2D foundation models even where calibrated images
sit beside the point cloud. DITR extracts frozen DINOv2 patch features, projects
the points into each camera to look them up, pools across views, and injects the
result into a 3D backbone trained with a supervised loss.

The first half transfers directly and is implemented in
`ggssvt/geometry/dino_lift.py`. The second half does not: DITR trains on
ScanNet, S3DIS and nuScenes, which supply per-point semantic labels, and this
dataset supplies none. The supervised head is therefore replaced by k-means over
the pooled features, which makes the question **can a foundation model separate
plant from pot where excess-green cannot?**

The success criteria were written down before the run, because otherwise any
clustering looks like a result. A useful separation puts the clusters at
different heights, puts most of the volume near the floor in one and most above
the rim in the other, and does so on the batch where colour fails.

| batch | n | mean height gap | agreement with rim | upper cluster above rim | rim confident |
|---|---|---|---|---|---|
| E001-E010 | 10 | 0.212 m | 0.704 | **0.497** | 1/10 |
| E011-E020 | 8 | 0.472 m | 0.905 | 0.786 | 8/8 |
| Mango | 10 | 0.620 m | **0.969** | **0.971** | 8/10 |
| V001-V008 | 8 | 0.365 m | 0.741 | **0.358** | 8/8 |

**The answer is no.** On Mango the lifted clustering recovers essentially the
same boundary the geometric rim detector found, agreeing on 96.9 per cent of
points with 97.1 per cent of the upper cluster above the rim, and M001 alone
reaches 0.998. Two methods sharing no mechanism agreeing that closely is a real
validation of the rim estimator, and it is the useful half of this result.

But E001-E010, the batch this was run for, gives 0.497: the upper cluster falls
half above and half below the rim, which is what a split unrelated to the
pot boundary looks like. V001-V008 gives 0.358 with confident rims, meaning the
clustering there finds a *different* boundary from the geometric one rather than
a better one. **DINO features confirm the split where it is already findable and
do not find it where it is not.**

**The reason is resolution, again.** DINOv2 with patch 14 on a 512 by 416 frame
gives a 37 by 30 patch grid, so one patch spans 13.8 pixels, which is 42 mm at
the 1.1 m working distance. A Eucalyptus stem is 5 to 15 mm. Every patch that
contains stem also contains pot, soil or background, so no pooling of those
features can separate the two. This is the same argument as the 12 mm voxel and
it has the same shape: the instrument is coarser than the structure.

Worth being precise about what this does and does not rule out. It rules out
patch-level DINOv2 at this capture resolution. It does not rule out DITR itself,
which was never given the labelled 3D data it is built for, nor a
higher-resolution capture, nor SAM-family masks lifted the way SAMa lifts them,
which operate on pixels rather than 42 mm patches.

`python -m ggssvt.cli dino-segment`, about 15 seconds a specimen on CPU.

---

## 7e. Does a stronger regressor help? Only while the input is bad

Asked because a previous student used R-squared and RMSE with a ridge-style fit,
and random forests and networks are the obvious alternatives. Measured rather
than argued: the same seven features, the same leave-one-out, only the regressor
changing.

| regressor | carved | fused |
|---|---|---|
| ridge | 0.544 / +0.030 | **0.335 / +0.632** |
| random forest | **0.406 / +0.459** | 0.393 / +0.493 |
| gradient boosting | 0.406 / +0.459 | 0.395 / +0.490 |
| MLP 32-16 | 0.506 / +0.160 | 0.403 / +0.469 |
| mean predictor | 0.552 | 0.552 |

Paired bootstrap against ridge on the same features:

| | difference | 95% interval | |
|---|---|---|---|
| carved, random forest | −0.138 | [−0.281, −0.026] | **resolved** |
| carved, gradient boosting | −0.138 | [−0.288, −0.001] | **resolved** |
| carved, MLP | −0.038 | [−0.114, +0.029] | not resolved |
| fused, random forest | +0.058 | [−0.003, +0.121] | not resolved |
| fused, gradient boosting | +0.059 | [−0.007, +0.127] | not resolved |
| fused, MLP | +0.068 | [+0.007, +0.129] | **resolved, worse** |

**The pattern is the finding.** A nonlinear regressor rescues the carve, resolving
a 0.138 kg improvement, and then stops helping once the reconstruction is fixed.
On the fused features ridge wins and the MLP is resolvably *worse*.

That reads as exactly what it looks like. Hull volume is an envelope whose
relationship to mass is distorted differently for a bushy Mango than for a thin
Eucalyptus, and a forest can carve that space with thresholds where a linear fit
cannot. Once the volumes are no longer envelopes the relationship is closer to
linear, and at n=36 the extra capacity costs more in variance than it buys in
bias. **A stronger model was compensating for a broken input.**

The decision it poses, and the honest answer:

| | RMSE |
|---|---|
| carved + ridge | 0.544 |
| carved + random forest | 0.406 |
| **fused + ridge** | **0.335** |

Fixing the reconstruction beats strengthening the model, and is also the cheaper
claim to defend. But **fused+ridge against carved+RF is −0.071 [−0.175, +0.028],
not resolved**, so the two routes cannot be separated on this data. What can be
said is that fixing the input beats the mean predictor by more, keeps the model
interpretable, and does not spend capacity at a sample size that cannot afford it.

Two cautions. Six comparisons were run and two resolved at 95 per cent, which is
close to what chance alone would produce, so no single row above should be
quoted without that context. And a random forest at n=36 is fitting 35 samples
per fold, which is not where forests are at their best.

**F1 does not apply.** It is a classification metric and above-ground mass is
continuous. Reaching for it would mean binning mass into classes, discarding the
resolution of a ground truth that was obtained destructively, and inventing
boundaries that carry no agronomic meaning. R-squared and RMSE are the right
family; what was missing was never the metric but the interval around it.

---

## 7f. The reconstruction metrics, and why the good reconstruction scores worse

Chamfer distance, Hausdorff and HD95, F-score and PSNR have been in
`eval/metrics.py` since early in the project and were never called once. That was
not an oversight. Those metrics measure distance to a reference, and there is no
reference: destructive harvest produced a mass, not a geometry, and no laser
scan, CT or CAD model of any specimen exists.

Two things they can legitimately do, kept apart here because conflating them
would overstate the result.

**Explanatory power against the captured views.** Project a reconstruction back
into each camera and score what it predicts against what was measured. No
reference needed, because the views are the reference.

| operator | silhouette IoU | depth MAE | depth PSNR | subject pixels explained |
|---|---|---|---|---|
| space carving | **0.407** | 67.9 mm | **32.70 dB** | **0.456** |
| TSDF fusion | 0.219 | 67.4 mm | 32.49 dB | 0.233 |

**The carve wins, and that is the finding.** A visual hull is by construction
consistent with every silhouette it was built from, so a silhouette-agreement
metric structurally favours it. The fusion scores lower because it has holes: it
claims only what a camera measured, and roughly one eighth of the working volume
was ever measured.

Set that beside what the same two operators do on the questions that matter:

| | carve | fusion |
|---|---|---|
| silhouette IoU | **0.407** | 0.219 |
| physically plausible | 8/36 | **25/36** |
| biomass RMSE | 0.544 kg | **0.335 kg** |

**A metric that looks like "reconstruction quality" ranks the worse
reconstruction higher.** This is the clearest argument the project has produced
for the plausibility check: without ground-truth geometry, self-consistency
metrics measure agreement with the input rather than fidelity to the object, and
for a hull that agreement is guaranteed rather than earned. Reporting silhouette
IoU alone would have pointed the whole project in the wrong direction.

Depth error is the honest tie. Where both predict a surface they are equally
accurate, 67.9 against 67.4 mm and 32.70 against 32.49 dB. The two operators
differ in how much they claim, not in how well they place what they claim.

**Agreement between the two operators**, which is a real number and is not
accuracy:

| | value |
|---|---|
| voxel IoU | 0.251 |
| Chamfer distance | 36.6 mm |
| F-score at 20 mm | 0.691 |
| HD95 | 224 mm |
| Hausdorff | 641 mm |

A quarter of the occupied voxels are shared and the worst-case separation is
0.64 m, which is most of the working volume. The two are describing substantially
different objects, and the 0.691 F-score says the disagreement is in the bulk
rather than in a few outliers. Two methods can agree closely and both be wrong;
these do not even agree.

**PSNR without a radiance field.** The conventional use needs rendered views
against captured ones and belongs to the splatfacto arm, which has
`transforms.json` exported for every specimen and has never been trained. The
figure above is depth PSNR from re-projection, which is a different quantity and
is labelled as such wherever it appears.

**One caveat that must travel with the re-projection numbers.** These views built
the reconstruction, so this is self-consistency rather than held-out
generalisation. A volume that fails to explain the images it was carved from is
definitely wrong; one that explains them may still be an envelope. Leave-one-view-
out would be the stronger test and costs twelve carves per specimen.

`python -m ggssvt.cli quality`, about a minute for all 36.

---

## 8. Bugs found and fixed

Several would have produced confident wrong numbers rather than errors.

| Bug | Consequence had it stood |
|---|---|
| `.gitignore` `data/` matched at any depth | `ggssvt/data/` and `nerfstudio/…/data/` never committed; fresh clones fail at import |
| Carve thresholds did not scale with view count | 4-view carve returns **empty**, reported as 0/30 usable, looks like a finding |
| Subject axis from single-view candidate | Confident registration onto background a metre behind the plant |
| Token anchors averaged background pixels | Anchors dragged off the specimen, corrupting the distance bias |
| Decoder used self-attention over concatenated queries | O((Q+N)²), several GB where O(Q·N) was needed |
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

**Keep H3, restated**, see [HYPOTHESIS_3.md](HYPOTHESIS_3.md). An earlier draft
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
CropCraft's approach, fitting a biologically plausible parametric model whose
leaves are explicit, makes leaf area a *model parameter* rather than something
the sensor must resolve. It needs a mango/eucalyptus morphology model, which is
substantial, but it is the direction that addresses the actual obstacle.

### Report readily, today

These stand on evidence already collected, with intervals:

1. **Calibration-free rig registration** from depth alone, method, diagnostics,
   and the visual-verification failure case. A contribution the proposal did not
   anticipate.
2. **Depth fusion beats silhouette carving on biomass**, 0.335 against 0.544,
   paired bootstrap −0.209 [−0.363, −0.066], with direct 2D unchanged as the
   control. This replaces the withdrawn "reconstruction beats pixels" claim
   struck through in section 3: that comparison never resolved, this one does.
3. **Volume allometry fails across morphologies** (R² = −0.162), motivates a
   learned density.
4. **Surface area does not beat volume, and why**, envelope area is not leaf
   area. A mechanism that generalises to any hull-based method.
5. **View-count requirement**, monotone degradation, and 12 views justified
   against the 4-view protocol used previously.
6. **The batch confound**, reported as a limitation. It is a methodological
   contribution about evaluating small phenotyping datasets.
7. **The F-score / voxel-IoU gap**, implemented and demonstrated on a synthetic
   shell (F-score 1.0 at IoU 0.58); this is Paper 1's thesis.

### Cheap data fixes worth doing before the next capture

- Weigh a sample of empty pots; record real pot heights per specimen
- Capture the ChArUco sequence `dataset/README.md` already specifies
- Step the rig back for tall specimens, or add a raised second tier
- Fix the camB naming in `collect_specimen.py`
- Record a continuous mass range within one species
