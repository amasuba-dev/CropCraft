# CropCraft method and metric inventory

All biomass scores below use plant-level leave-one-out validation unless noted.
The pipeline is:

**image acquisition -> calibration/registration -> segmentation -> reconstruction
or feature extraction -> pot/stand exclusion -> above-ground geometry/features ->
biomass regression -> reconstruction and physical-validity checks**.

## End-to-end method table

| Method | Learning type / ViT use | Description and pipeline | Status | Best biomass result | Reconstruction / segmentation metrics | Relevant outputs |
|---|---|---|---|---|---|---|
| **A. Geometric visual hull** | **Classical deterministic geometry; no learning; no ViT.** | RGB-D -> rig registration -> cylinder/excess-green masks -> silhouette consistency -> voxel space carving -> pot-rim filter -> geometry -> regression. | Complete; E001-E010 pot/stand failure remains. | Geometric features: **RMSE 0.469 kg, MAE 0.375 kg, MARE 0.402, R² 0.309, bias -0.013 kg**. | Carve reprojection: silhouette IoU **0.400 mean** (0.132–0.524), depth MAE **0.0707 m**, PSNR **32.43 dB**, coverage **0.451**; agreement F-score **0.664**, Chamfer **0.0391 m**. | [Gallery](../work_dirs/ggssvt/reports/gallery/reconstructions.html), [quality report](../work_dirs/ggssvt/reports/reconstruction_quality.json) |
| **B. SAM3D carving** | **Pretrained learned segmentation; self-supervised ViT upstream, but no CropCraft fine-tuning; not a CropCraft ViT learner.** | RGB -> SAM3D subject masks -> multi-view mask gating -> voxel carving -> pot exclusion -> geometry -> regression. | Complete, but masks can remove thin shoots or retain the stand. | SAM3D alone: **0.778 kg RMSE, R² -0.967**. SAM3D + DINOv2: **0.390 kg, R² 0.505**. | Best campaign occupancy IoU **0.7088**; per-sample IoU/F-score vary substantially. Standardized mIoU/precision/recall/F1 export remains pending. | [Gallery](../work_dirs/ggssvt/reports/gallery/reconstructions.html), [campaign dashboard](../work_dirs/ggssvt/reports/campaign_visualizations/index.html) |
| **C. TSDF/fused geometry** | **Classical metric reconstruction; no learning; no ViT.** | RGB-D -> registration -> masks and metric depth -> TSDF fusion -> observed surface/occupancy -> above-ground rim filter -> shape features -> regression. | Complete; strongest validated classical method. | **RMSE 0.430 kg, MAE 0.352 kg, MARE 0.393, R² 0.416, bias -0.007 kg**. | Fused reprojection: silhouette IoU **0.216 mean**, depth MAE **0.0696 m**, PSNR **32.24 dB**, coverage **0.231**; agreement metrics are evaluated separately. | [Metrics](../work_dirs/ggssvt/reports/metrics.json), [mesh metrics](../work_dirs/ggssvt/reports/mesh.json) |
| **D. DUSt3R / MASt3R / Fast3R** | **Pretrained learned pointmap transformers; inference-only here; no CropCraft self-supervised training.** | RGB views -> pointmaps/cameras -> metric scale recovery -> Umeyama rig alignment -> pot-rim filtering -> above-ground PLY -> geometry/volume regression. | Complete for 38 plants. | Fast3R: **0.523 kg RMSE, R² 0.139**; MASt3R: 0.672/-0.422; DUSt3R: 0.727/-0.663. | Mean azimuth RMSE: DUSt3R **37.4°**, Fast3R **39.1°**, MASt3R **62.9°**; scale recovered for DUSt3R 38/38. | [Pose-free biomass](../work_dirs/ggssvt/reports/pointcloud_biomass_retrained_20260913.json), [clouds](../work_dirs/ggssvt/posefree/ply_above_ground/) |
| **D+. VGGT** | **Pretrained learned vision transformer; inference-only; no CropCraft self-supervised training.** | RGB views -> confidence-filtered point cloud/cameras -> COLMAP export -> rig alignment -> pot exclusion -> biomass probe. | Incomplete: threshold 5 produced empty clouds; threshold 1 partial before CUDA OOM. | Not reportable. | Valid-completion rate insufficient for comparison; confidence acceptance and point count are the current metrics. | [VGGT log](../work_dirs/ggssvt/reports/vggt_valid_all_20260913.log) |
| **E. GG-SSVT** | **Self-supervised vision transformer; learned occupancy and biomass model.** | RGB-D/world anchors -> CNN/DINO tokens -> Fourier encoding -> geometric cross-view attention -> implicit occupancy decoder -> learned density/residual biomass head. | Campaign implemented; direct pose-free PLY retraining not wired. | Accepted trained campaign: **0.557 kg RMSE, R² 0.024**. | Occupancy IoU is available per fold; best reported **0.7088**. Training loss, calibration and plant-level generalisation remain the principal diagnostics. | [Campaign dashboard](../work_dirs/ggssvt/reports/campaign_visualizations/index.html), [architecture](../work_dirs/ggssvt/reports/architecture/) |
| **F. Frozen DINOv2/DINOv3** | **Self-supervised vision-transformer representations; frozen feature extraction, no CropCraft fine-tuning.** | RGB -> frozen patch tokens -> pooling/PCA -> ridge LOOCV biomass regression; optional geometry concatenation. | Completed probes. | DINOv2-base + geometry: **0.400 kg RMSE, R² 0.483**; DINOv3-base: 0.409 kg, R² 0.459. | Feature/point-lift IoU and recall are auxiliary; no reconstruction volume is claimed. | [DINO report](../work_dirs/ggssvt/reports/dino_probe.json), [screen report](../work_dirs/ggssvt/reports/regressor_screen.json) |
| **G. Mesh geometry** | **Classical deterministic surface modelling; no learning; no ViT.** | Occupancy/TSDF -> marching cubes -> mesh cleanup -> surface/volume/solidity -> biomass regression. | Completed on historical subset. | Approximately **0.359 kg RMSE, R² 0.613** on its subset. | Surface area, enclosed/voxel/convex-hull volume, height, solidity and voxel/mesh ratio. | [Mesh report](../work_dirs/ggssvt/reports/mesh.json) |
| **H. Direct 2D/allometric controls** | **Classical statistical learning: ridge/allometric regression; no ViT.** | Silhouette area, depth profile or volume -> hand-built descriptors -> LOOCV regression, with mean/batch controls. | Complete. | Direct 2D: **0.591 kg RMSE, R² -0.100**; volume allometric: **0.619 kg, R² -0.207**. | Area/profile/volume and density plausibility; batch-only control is a leakage diagnostic. | [Baseline metrics](../work_dirs/ggssvt/reports/metrics.json) |
| **I. Nerfstudio/neural field** | **Self-supervised photometric/implicit scene optimisation; not necessarily a ViT.** | RGB+poses -> NeRF/splat training -> novel-view geometry -> metric scale -> mesh/volume -> biomass. | Pose preparation exists; training/scoring incomplete. | No valid result. | Intended metrics: held-out PSNR/SSIM/LPIPS, depth/geometry error, scale stability and volume bias. | Planned; do not treat generated volume as measured volume. |
| **J. Pheno4D virtual views** | **Classical validation protocol; no learner and no ViT.** | Laser cloud -> synthetic RGB-D views -> identical pipeline -> compare to known geometry/biomass. | Dataset/scoring incomplete. | No valid result. | Intended ground-truth Chamfer, point-to-surface distance, IoU, F-score, volume error and signed bias. | Planned external-validation track. |
| **K. ABVT3R external baseline** | **Classical geometric reconstruction plus supervised regression; no ViT in the reported RF/ANN path.** Optional DINO/neural modules are separate. | RGB-D -> PassThrough/SOR/MLS -> circular coarse registration + ICP -> 7 mm voxel/marching-cubes mesh -> geometric features -> **RF or ANN** biomass prediction. | Reconstruction completed for 40 CropCraft plants. Native RF artifact is non-portable, so shared biomass scores use ABVT3R's documented training report. | CropCraft adapter: mean mesh volume **0.0924 m³**, quality **0.731**, surface area **19.78 m²**, mean ICP RMSE **26.85 mm**, ICP convergence **129/132 = 97.7%**; M002 fallback recorded. | ABVT3R primary RF LOOCV: **MAE 330.1 g, RMSE 401.7 g, bias +10.3 g, nRMSE 0.362, MARE 0.348, R² 0.451, CCC 0.618**, bootstrap RMSE **[320.8, 470.6] g**, R² **[0.141, 0.630]**. ANN: **MAE 408.2 g, RMSE 509.0 g, bias +2.0 g, nRMSE 0.458, MARE 0.441, R² 0.118, CCC 0.335**. | [ABVT3R gallery](../work_dirs/ggssvt/abvt3r/index.html), [adapter](./ABVT3R_ADAPTER.md), [ABVT3R Pipeline](../../ABVT3R/Pipeline.md) |

## Cross-method metric ledger

| Metric family | Metrics covered | Interpretation |
|---|---|---|
| Biomass error | RMSE, MAE, MARE, R², signed bias, nRMSE, CCC, bootstrap CI, leverage, sample count | Predictive accuracy and systematic over/under-estimation. |
| Segmentation | IoU, mIoU, precision, recall, F1/Dice, AP and mAP, occupancy IoU, mask recall, coverage, fragments | Plant/pot pixel or voxel separation. mAP is valid only when class/object ground-truth masks and confidence scores exist. |
| Reconstruction geometry | Volume, surface area, canopy area, height, footprint, solidity, convex-hull volume, voxel/mesh ratio, mesh quality, manifold/closed status, boundary-edge ratio | Whether recovered plant shape is metrically meaningful. |
| Depth/pose | Depth MAE **0.0707 m carve / 0.0696 m fused**, depth PSNR **32.43 / 32.24 dB**, coverage **0.451 / 0.231**, rotation RMSE, maximum rotation error, azimuth RMSE, scale recovery, ICP RMSE and convergence | Sensor/reconstruction consistency and camera-frame reliability. |
| Physical validity | Implied density, plausible-count, mean volume, density variance, signed volume error | Detects hollow or over-carved reconstructions that boundary metrics can miss. |
| Robustness | View-count usability, view agreement, depth-noise degradation, occlusion degradation, fragment count | Stability when observations are sparse, noisy or blocked. |
| Learning diagnostics | Training/validation loss, occupancy IoU, calibration, label efficiency, frequency/positional-encoding ablations | Whether a learned model improves for the claimed reason rather than through leakage or a bundled change. |
| ABVT3R regression diagnostics | RF/ANN MAE, RMSE, bias, nRMSE, MARE, R², CCC, Bland–Altman limits, bootstrap intervals, leverage share, tree/parameter counts | Useful additions for comparing classical supervised regressors fairly; RF and ANN remain separate rows. |
| External validation | Chamfer distance, point-to-surface distance, F-score, IoU, volume error and bias on Pheno4D | Required before making ground-truth reconstruction claims. |

## Metric applicability by experiment

`Reported` means a measured value exists. `Pending` means the metric is
scientifically appropriate but the required labels, confidence scores or
reference geometry have not yet been produced. `N/A` means the metric does
not describe that experiment.

| Experiment | Biomass RMSE/MAE/R² | IoU/mIoU/F1 | AP/mAP | F-score/Chamfer | Pose/depth | Physical/robustness |
|---|---|---|---|---|---|---|
| A. Visual hull | Reported | IoU **0.400 mean**; mIoU/F1 pending standardized export | N/A for detection; segmentation mAP pending masks + ground truth | F-score **0.664 mean**; Chamfer **0.0391 m** | Depth MAE **0.0707 m**, PSNR **32.43 dB**, coverage **0.451** | Reported density, volume and view-count robustness |
| B. SAM3D carving | Reported | Best occupancy IoU **0.7088**; mIoU/F1 pending | Pending; requires per-instance confidence scores and ground-truth masks | Pending standardized reference-geometry comparison | Depth/coverage available per campaign fold | Density and sparse-view metrics reported |
| C. TSDF/fused geometry | Reported | IoU **0.216 mean**; mIoU/F1 pending standardized export | N/A for detection; segmentation mAP pending labels | Pending ground-truth geometry | Depth MAE **0.0696 m**, PSNR **32.24 dB**, coverage **0.231** | Density, volume and robustness reported |
| D. DUSt3R/MASt3R/Fast3R | Reported | N/A unless point clouds are projected against labelled masks | N/A | Pending ground-truth geometry | Azimuth RMSE: DUSt3R **37.4°**, Fast3R **39.1°**, MASt3R **62.9°** | Above-ground volume and density reported |
| D+. VGGT | Not valid yet | Pending | N/A | Pending | Confidence completion incomplete; CUDA OOM | Not reportable |
| E. GG-SSVT | Campaign reported | Occupancy IoU best **0.7088**; mIoU/F1 pending | Pending; model outputs need confidence-calibrated masks | Pending external geometry | Training/registration diagnostics available | Loss, calibration and robustness partially reported |
| F. DINOv2/DINOv3 | Reported | Point-lift IoU/recall auxiliary; mIoU/F1 pending | N/A for biomass regression | N/A unless lifted features are evaluated against reference geometry | N/A as a reconstruction method | Label-efficiency and paired comparisons reported |
| G. Mesh geometry | Historical biomass result | N/A without projected reference masks | N/A | Pending ground-truth mesh | N/A unless mesh is aligned to reference | Mesh-quality metrics reported |
| H. 2D/allometric controls | Reported | Area/profile metrics only; IoU pending labels | N/A | N/A | N/A | Batch-only and LOOCV controls reported |
| I. Nerfstudio | No valid biomass result | Pending rendered-mask evaluation | Pending rendered instances/confidence | Pending reference geometry | Intended PSNR/SSIM/LPIPS and depth metrics | Training throughput/VRAM planned |
| J. Pheno4D | No valid biomass result | Planned ground-truth IoU/mIoU/F1 | N/A unless object detections are introduced | Planned Chamfer, point-to-surface and F-score | Planned known-pose and depth metrics | Planned volume bias and robustness |
| K. ABVT3R RF/ANN | RF and ANN reported separately | Geometric masks exist; mIoU/F1 pending labels | N/A for biomass regression; segmentation mAP pending confidence masks | Agreement metrics only without ground-truth geometry | ICP RMSE **26.85 mm**, convergence **97.7%** | Mesh quality, volume, density and regression CIs reported |

### Why mAP is not yet present as a number

mAP is primarily an object-detection or confidence-ranked instance/semantic
segmentation metric. CropCraft currently has binary heuristic masks and
reconstructed occupancy, but not a common set of manually verified plant/pot
ground-truth masks plus calibrated per-pixel or per-instance confidence
scores for every experiment. Reporting a numeric mAP now would therefore be
fabricated or incomparable. The correct next metric run is a standardized
mask benchmark that exports precision-recall curves, AP50, AP75, AP@[.50:.95],
and mAP for Methods A, B, C, E and K; mAP remains N/A for biomass-only
regressors and pose-only methods.

## Current headline comparison

**Fused geometry is the strongest validated classical pipeline. Frozen DINOv2 is
the strongest practical representation probe. Fast3R is the best pose-free
biomass probe, but remains below fused geometry. IoU/F-score must be reported
alongside volume bias, density plausibility and plant-level splits.**

ABVT3R should be evaluated as an external classical RGB-D/ICP/mesh baseline,
not as a self-supervised vision-transformer result. Its README describes an
optional neural subsystem, but the reported biomass pipeline is the classical
point-cloud, mesh and Random Forest path.

## Presentation evidence pack

These figures are derived from existing reports and reconstructed outputs. They
are diagnostic/presentation visualisations, not additional benchmark runs:

- [ABVT3R segmentation contact sheet](../work_dirs/ggssvt/reports/presentation_evidence/abvt3r_segmentation_contact_sheet.png)
- [M001 pot/shoot exclusion](../work_dirs/ggssvt/reports/presentation_evidence/M001_pot_exclusion.png)
- [Biomass comparison](../work_dirs/ggssvt/reports/presentation_evidence/biomass_comparison.png)
- [Evidence availability and unresolved tracks](../work_dirs/ggssvt/reports/presentation_evidence/evidence_status.png)
- [Figure manifest](../work_dirs/ggssvt/reports/presentation_evidence/manifest.json)

The generator is [presentation_evidence.py](./eval/presentation_evidence.py);
rerun it after replacing or adding source reports.
