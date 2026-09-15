# ABVT3R classical pilot

ABVT3R's reusable classical implementation is in
`/home/titan/aaron/ABVT3R/procedure_alpha`: preprocessing (depth to point cloud,
pass-through ROI, SOR), known-angle coarse registration plus ICP, and the
first-principles grid mesh reconstruction. CropCraft's `camB_000..150` names
are resolved as physical azimuths 180..330 by `ggssvt.data.dataset`; the adapter
passes those physical angles to ABVT3R.

Run from the CropCraft checkout:

```bash
python -m ggssvt.eval.abvt3r_adapter --plant M001 --max-points 1200
```

The pilot writes only derived files to `work_dirs/ggssvt/abvt3r/` and emits
`M001_report.json`. The 12-view M001 pilot completed preprocessing,
registration, segmentation/ROI, and reconstruction. It retained 13,991 input
points, produced 60,575 mesh vertices and 106,723 triangles, with 0.094256 m³
mesh volume and quality 0.721. ICP RMSE averaged 28.969 mm; view 210° reached
the 50-iteration cap.

The rerun also writes one camera-frame filtered cloud
(`M001_camA_000_filtered.ply`, etc.) and one binary grayscale mask
(`M001_camA_000_mask.png`, etc.) per view, plus `M001_gallery.html` linking
these and the merged cloud, mesh vertices, triangles, and normals. Pillow was
available for PNG output; without it the adapter falls back to documented
`.npy` pixel masks.

Biomass prediction is blocked by the legacy RF artifact: loading
`ABVT3R/RF_model/biomass_rf_model.npy` raises
`AttributeError: Can't get attribute 'DecisionTreeRegressor' on __main__`.
`ggssvt.eval.abvt3r_rf` is the CropCraft-side evaluator and the external clone
was not modified. A portable sklearn/joblib export (and feature schema) is the
next prerequisite for RF predictions.

Batch runs use `--batch M002 V001 ... V010`. The adapter catches ABVT3R's
zero-correspondence `UnboundLocalError`, records `fallback_unregistered` in the
plant report, and continues reconstruction. The aggregate
`work_dirs/ggssvt/abvt3r/index.html` recursively links available reports and
galleries, including prior outputs in `abvt3r/full/`.
