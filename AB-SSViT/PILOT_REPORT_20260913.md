# AB-SSViT pilot report

**Run date:** 13–14 September 2026  
**Purpose:** Test the thesis-aligned candidate pipeline and map evidence to the
hypothesis and research questions.  
**Protocol:** plant-level leave-one-out evaluation; fold-local PCA,
standardisation, and ridge fitting.

## Executive result

The fresh DINOv2 extraction and AB-SSViT evaluation completed successfully on
**38 specimens**:

| Condition | RMSE (kg) | MAE (kg) | MARE | R2 | Bias (kg) |
|---|---:|---:|---:|---:|---:|
| Fresh frozen DINOv2 + fused geometry (primary) | **0.336** | **0.268** | **0.287** | **0.645** | -0.014 |
| Fresh frozen DINOv3 + fused geometry | 0.361 | 0.269 | 0.272 | 0.588 | -0.026 |
| Enriched geometry/profile/image descriptors | 0.492 | 0.366 | 0.386 | 0.239 | +0.024 |
| Earlier cached DINOv2 + fused geometry | 0.361 | 0.288 | 0.311 | 0.581 | -0.011 |

The fresh primary condition improves on the current 38-specimen fused-geometry
reference (0.430 kg RMSE, R2 0.416) under the same specimen count. This is
stronger evidence for the candidate, but still requires repeated seeds and
campaign-held-out validation.

## Pipeline validation

The two-hour launcher reached the complete preprocessing and TSDF stages:

- 38/38 fused caches written;
- 32/38 TSDF reconstructions passed the existing density plausibility check;
- fused geometry report written successfully;
- fresh DINOv2 descriptors regenerated for 38/38 specimens;
- strict plant-level evaluation completed with fold-local transformations.

The first launcher attempt was blocked by a missing optional dependency and the
second by a stale CLI argument; both were corrected. The final fresh run
completed successfully:

[AB-SSViT pilot JSON](../work_dirs/ab_ssvit/ab_ssvit_pilot.json)  
[pilot launcher log](../work_dirs/ab_ssvit/pilot_20260913.log)  
[Fresh DINO probe](../work_dirs/ab_ssvit/dino_probe_fresh_20260914.json)  
[Fresh AB-SSViT evaluation](../work_dirs/ab_ssvit/ab_ssvit_fresh_20260914.json)  
[DINOv3 probe](../work_dirs/ab_ssvit/dino_probe_dinov3_fresh_20260914.json)  
[DINOv3 AB-SSViT evaluation](../work_dirs/ab_ssvit/ab_ssvit_dinov3_20260914.json)

## Hypothesis and research-question answers

### H1: Self-supervised ViT features contain biomass signal

**Pilot answer: supported provisionally.** Fresh frozen DINOv2 with fold-local
ridge regression produced R2 0.645 and RMSE 0.336 kg on 38 specimens. The
fresh DINOv2 probe also beat its CNN/no-DINO control: RMSE 0.418 versus 2.981
kg in the same run.

This is still one seed and does not establish campaign transfer or causal
benefit from geometry; those remain planned ablations.

### DINOv2 versus DINOv3

**Pilot answer: DINOv2 is currently the better candidate.** Under the same
38-specimen AB-SSViT evaluator, DINOv2 achieved 0.336 kg RMSE and R2 0.645,
while DINOv3 achieved 0.361 kg RMSE and R2 0.588. The difference is not yet
statistically established; repeat seeds and paired bootstrap intervals are
required. DINOv3 remains a valid secondary backbone, not the primary model.

### H2: Geometry adds complementary information

**Pilot answer: unresolved by this run.** The current AB-SSViT implementation
uses DINOv2 plus fused geometry as its primary condition, but it does not emit
a directly paired DINOv2-only condition in this invocation. The existing
inventory reports DINOv2 and DINOv2-plus-geometry as approximately tied in one
earlier probe. The next run must compare both conditions with identical
specimens, folds, and hyperparameters.

### H3: Frequency and sampling should be structure-specific

**Pilot answer: partially supported from prior experiments; not tested by this
pilot.** Existing H3 evidence shows structure-specific occupancy spectra and
view-count requirements. The candidate should therefore use view-count
ablations and a resolution/frequency ablation rather than assume one universal
setting.

### H4: Multi-view fusion improves robustness

**Pilot answer: not yet tested by AB-SSViT.** The current probe uses robust
descriptor aggregation, not a trainable cross-view attention block. This is the
next model stage after the ridge baseline.

### H5: Pot exclusion improves biological validity

**Pilot answer: operationally necessary, predictive effect not isolated here.**
The geometry branch is based on fused plant observations and the existing
above-ground filtering contract. A paired whole-scene versus above-ground
ablation is required before claiming a causal improvement.

## Required next experiments

1. Repeat the fresh DINOv2 extraction with a second seed/cache audit.
2. Emit paired DINOv2-only, geometry-only, and DINOv2-plus-geometry predictions
   from the same folds.
3. Add campaign-held-out validation: train on two groups and test on the third.
4. Implement the small trainable head: geometry projection, azimuth embedding,
   two-layer cross-view attention, gated fusion, and uncertainty head.
5. Pretrain that head with same-plant cross-view consistency and masked-view
   feature prediction, without biomass labels.
6. Evaluate 3/4/6/12 views and 12 mm versus 6 mm voxel resolution.
7. Add verified plant/pot masks before reporting mAP, AP50, or AP75.

## Full-training machine recommendation

Use a GPU with **at least 16 GB VRAM**, 64 GB RAM, and local SSD storage. Keep
DINOv2 frozen for the first full experiment. Train only the projection,
cross-view attention, fusion gate, and regression/uncertainty heads. Then run:

- five random seeds;
- plant-level LOOCV;
- three campaign-held-out folds;
- bootstrap confidence intervals;
- frozen versus adapter/LoRA fine-tuning;
- above-ground versus whole-scene ablation.

Do not start with full DINOv2 fine-tuning or VGGT. The current sample size and
reconstruction-quality failures make those high-variance choices. The
scientifically defensible thesis progression is:

```text
classical fused geometry
 -> frozen DINOv2
 -> frozen DINOv2 + geometry
 -> multi-view self-supervised adaptation
 -> optional parameter-efficient fine-tuning
```
