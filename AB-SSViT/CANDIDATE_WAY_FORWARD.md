# AB-SSViT candidate way forward

## Alignment with the thesis

The thesis hypothesis is that a self-supervised vision-transformer
representation, made multi-view and geometry-aware, can estimate above-ground
biomass more accurately and robustly than image-only, classical geometry, and
pose-free reconstruction baselines.

The current evidence supports this as a testable direction, but not yet as a
confirmed claim:

- frozen DINOv2 is the strongest practical representation probe
  (approximately 0.400 kg RMSE, R2 0.483);
- fused geometry is the strongest all-specimen classical reference
  (0.430 kg RMSE, R2 0.416);
- the existing trained GG-SSVT campaign is weaker (0.557 kg RMSE, R2 0.024);
- the DINOv2-plus-geometry result is promising but must be repeated with one
  consistent specimen set and strict fold-local preprocessing.

### Research-question mapping

| Question | AB-SSViT test | Primary evidence |
|---|---|---|
| Can self-supervised ViT features estimate biomass? | DINOv2-only ridge vs CNN and mean controls | RMSE, MAE, MARE, R2 |
| Does geometry add complementary information? | DINOv2-only vs DINOv2 + above-ground geometry | paired specimen residuals and bootstrap delta |
| Does multi-view fusion improve robustness? | view pooling vs learned cross-view attention | held-out plant metrics by view count |
| Does self-supervised adaptation improve transfer? | frozen DINOv2 vs multi-view consistency pretraining | held-out campaign and label-efficiency results |
| Does pot exclusion improve validity? | whole-scene vs above-ground-only features | pot-retention rate, density plausibility, biomass error |

## Candidate architecture

```text
RGB-D views
  -> rig registration and plant/pot mask
  -> frozen DINOv2 patch tokens
  -> masked patch pooling + azimuth embedding
  -> cross-view attention / robust view pooling
  -> above-ground TSDF and point-cloud geometry branch
  -> gated token fusion
  -> small biomass head with uncertainty
```

The implementation is staged to respect the 38-plant labelled sample size:

1. **Probe:** fold-local PCA + ridge (current reproducible baseline).
2. **Fusion:** train only a small projection, attention, and regression head with
   DINOv2 frozen.
3. **Self-supervised adaptation:** masked-view reconstruction and same-plant
   cross-view consistency without biomass labels.
4. **Optional fine-tuning:** unfreeze only the last DINOv2 block or use LoRA.

The geometry branch must use above-ground features and remain separately
inspectable. No pot-inclusive score is allowed as the primary result.

## Required ablations and validation

- mean, fused geometry, DINOv2-only, DINOv2 + geometry;
- view pooling versus cross-view attention;
- frozen versus adapted backbone;
- above-ground versus whole-scene geometry;
- 3, 4, 6, and 12 views;
- E, M, and V campaign-held-out tests;
- plant-level LOOCV with fold-local PCA, scaling, and hyperparameter selection;
- RMSE, MAE, MARE, R2, signed bias, bootstrap intervals, residual plots;
- segmentation IoU/F1/precision/recall when verified masks are available;
- reconstruction depth error, coverage, Chamfer/F-score, volume plausibility.

## Two-hour pilot interpretation

The pilot is a pipeline and hypothesis-validation run, not a final neural
training claim. It should verify that all caches, modules, fold-local fitting,
reports, and provenance are reproducible on the target machine. A good pilot
must produce:

- a completed fused-geometry baseline;
- DINOv2 probe metrics;
- AB-SSViT fold predictions and residuals;
- a comparison against the mean and classical references;
- explicit pass/fail answers for each research question;
- a full-machine training recommendation.

For a full training machine, use a GPU with at least 16 GB VRAM, 64 GB system
RAM, fast local SSD storage, and enough time to repeat seeds and campaign-held
out tests. Start with frozen DINOv2, then add adapters before attempting full
backbone fine-tuning.
