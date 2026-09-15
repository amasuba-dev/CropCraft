# GG-SSVT and pose-free biomass validation

**Date:** 14 September 2026  
**Evaluation framework:** plant-level leave-one-out validation for biomass;
reprojection against captured RGB-D masks/depth; reconstruction agreement
between independent operators; all biomass geometry filtered above the
specimen-specific pot rim.

## 1. Biomass results

| Arm | n | RMSE (kg) | MAE (kg) | MARE | R2 | Bias (kg) |
|---|---:|---:|---:|---:|---:|---:|
| **GG-SSVT trained baseline_fused** | 38 | 0.557 | 0.347 | 0.336 | 0.024 | +0.042 |
| DINOv3 + above-ground fused geometry | 38 | 0.361 | 0.269 | 0.272 | 0.588 | -0.026 |
| Fast3R above-ground PLY ridge | 38 | 0.523 | 0.426 | 0.483 | 0.139 | +0.014 |
| MASt3R above-ground PLY ridge | 38 | 0.672 | 0.521 | 0.581 | -0.422 | +0.081 |
| DUSt3R above-ground PLY ridge | 38 | 0.727 | 0.535 | 0.567 | -0.663 | +0.059 |

The DINOv3 row is a frozen self-supervised representation plus the
above-ground fused geometry branch. It is not the end-to-end GG-SSVT campaign
model. The GG-SSVT result is the trained volumetric transformer checkpoint.

GG-SSVT source:
[baseline_fused summary](../work_dirs/ggssvt/reports/campaign_visualizations/baseline_fused/summary.json)

DINOv3 source:
[AB-SSViT DINOv3 evaluation](../work_dirs/ab_ssvit/ab_ssvit_dinov3_20260914.json)

Pose-free source:
[above-ground PLY evaluation](../work_dirs/ggssvt/reports/pointcloud_biomass_thesis_20260914.json)

## 2. Reconstruction and segmentation metrics

### Classical RGB-D reference pipeline

| Operator | Silhouette IoU | Depth MAE | PSNR | Coverage |
|---|---:|---:|---:|---:|
| Visual-hull/carving | **0.400 mean (0.132–0.524)** | **0.0707 m** | **32.43 dB** | **0.451** |
| TSDF/fused geometry | 0.216 mean | 0.0696 m | 32.24 dB | 0.231 |

Agreement between the independent carve and fusion reconstructions:

- F-score: **0.664**
- Chamfer: **0.0391 m**
- Voxel IoU: 0.240

These are reprojection and operator-agreement metrics, not ground-truth
reconstruction accuracy. There is no laser/CAD geometry ground truth for the
captured plants.

### GG-SSVT trained volumetric model

The accepted `baseline_fused` GG-SSVT checkpoint reports:

- occupancy AP: **0.414**
- best occupancy IoU: **0.310**
- biomass metrics as reported above

The captured-view reconstruction metrics listed above are the input-pipeline
reference for GG-SSVT. A model-specific held-out reconstruction score requires
running its predicted occupancy through the same reprojection evaluator; the
current campaign summary does not export those per-view predictions.

### DINOv3 accepted backbone

DINOv3 is a feature backbone, not a reconstruction operator. Therefore its
reconstruction/segmentation metrics are reported as the metrics of the
above-ground RGB-D preprocessing and fused-geometry branch it consumes:

- input segmentation/reprojection silhouette IoU: **0.400 mean**
- depth MAE: **0.0707 m**
- PSNR: **32.43 dB**
- coverage: **0.451**
- agreement F-score: **0.664**
- Chamfer: **0.0391 m**

This attribution is intentionally labelled as pipeline-level, not as a claim
that DINOv3 generated the reconstruction.

Numeric segmentation mAP/AP50/AP75 remains pending because a common,
manually verified plant/pot mask set and confidence-ranked predictions have not
yet been created.

## 3. Pot exclusion and direct pose-free retraining

All pose-free PLY arms were evaluated from rig-aligned clouds using the
specimen-specific rim:

```text
read aligned PLY
 -> discard z <= pot_height_m
 -> voxelise above-ground points
 -> volume, height, footprint, spread, density features
 -> plant-level LOOCV ridge/allometric biomass regression
```

Filtered clouds are written under:

`work_dirs/ggssvt/posefree/ply_above_ground_filtered/`

This directly rewires the PLY outputs into the same biomass evaluation
contract and prevents pot/stand points from entering the primary biomass
features.

## 4. Hypothesis litmus tests

| Hypothesis | Test | Result |
|---|---|---|
| H1: self-supervised ViT features contain biomass signal | DINOv3 vs CNN/no-DINO and classical reference | **Supported provisionally**; DINOv3+geometry R2 0.588 and RMSE 0.361 kg |
| H2: geometry adds complementary information | DINO-only vs ViT+above-ground geometry | **Not isolated causally**; must run paired same-fold DINO-only and fusion ablations |
| H3: frequency/view requirements are structure-specific | prior spectral and view-count ablations | **Supported by existing H3 evidence**; not re-estimated in this run |
| H4: learned multi-view fusion improves robustness | pose-free PLY probes versus learned cross-view attention | **Unresolved**; current DINO arms use robust pooling, not a trainable attention head |
| H5: pot exclusion improves biological validity | above-ground PLY filtering and density plausibility | **Operationally supported**; causal biomass delta still needs whole-scene paired ablation |

## 5. Interpretation and next experiment

The strongest current thesis candidate is the frozen DINOv2/DINOv3 feature
branch with above-ground fused geometry, while the trained GG-SSVT campaign
is the architectural self-supervised volumetric baseline. Pose-free methods
are useful reconstruction alternatives but currently do not outperform the
metric RGB-D pipeline.

The next scientifically necessary run is not another broad model sweep. It is
a paired ablation using the same 38 plants:

1. DINOv2-only;
2. DINOv3-only;
3. fused geometry-only;
4. DINOv2 + above-ground geometry;
5. DINOv3 + above-ground geometry;
6. whole-scene versus pot-excluded geometry;
7. campaign-held-out validation by E/M/V group.

All rows should use the same fold-local PCA, scaling, ridge selection, seeds,
and bootstrap comparison.

