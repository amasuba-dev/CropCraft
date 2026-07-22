# Rig calibration → Nerfstudio transforms.json

Standalone toolchain for the dual-Kinect, 12-position rig. Not part of
CropCraft's own pipeline (which relies on Polycam/COLMAP structure-from-motion
instead of known poses) — this exists because you know your rig's geometry
and can skip SfM entirely by solving it directly.

Only dependency is `opencv-python`, already in the repo's `requirements.txt`
— no `opencv-contrib`/ArUco needed, since both calibration steps below use a
plain checkerboard.

## Workflow

1. **Per-camera intrinsics** (`calibrate_intrinsics.py`) — run once for camA,
   once for camB. ~15-20 handheld checkerboard shots per camera, varying
   tilt/distance/position in frame.

2. **Per-position extrinsics** (`calibrate_extrinsics.py`) — run once for the
   whole rig. Place a checkerboard flat where the plant pot will sit (this
   defines your world origin) and, **without moving it**, photograph it from
   all 12 rig positions. Solves each position's camera-to-world pose. This
   is reusable for every future specimen *only if* your 12 positions are
   mechanically repeatable (hard stops/detents) — if not, redo this step per
   capture session.

   Use `--visualize` to sanity-check before trusting it: the plotted camera
   arrows should point roughly inward toward the origin in a ring. If they
   don't, something's wrong (board moved between shots, wrong camera/position
   paired in the manifest, etc.) — don't proceed to NeRF training until this
   looks right.

3. **Per-specimen transforms.json** (`make_transforms.py`) — run once per
   plant, combining the fixed rig calibration with that plant's 12 captured
   frames.

4. **Train + export**, using Nerfstudio directly (not CropCraft's
   `train_nerf.py`, which is a thin wrapper around exactly this call but
   hardcoded for COLMAP-derived field data):
   ```bash
   ns-train depth-nerfacto --data ./data/plant_007 \
       --output-dir ./work_dirs --experiment-name plant_007 \
       --max-num-iterations 20000
   ns-export pointcloud --load-config ./work_dirs/plant_007/.../config.yml \
       --output-dir ./data/plant_007 --num-points 1000000 --remove-outliers True
   ```

## Prerequisites this toolchain assumes, not handles

- **Depth-to-RGB registration.** Kinect v2's depth/IR sensor and color
  sensor are physically offset with different resolutions — raw depth must
  already be aligned to the color frame (e.g. via libfreenect2's
  `Registration` class, or the equivalent SDK call) before it reaches
  `make_transforms.py`. This toolchain just points at whatever registered
  depth PNGs you hand it.
- **Depth units.** Confirm what your registration step actually outputs
  (commonly uint16 millimeters) and pass the matching `--depth_unit_scale` —
  don't assume 0.001 without checking.
- **Segmentation** (plant vs. pot vs. background) is not handled here at
  all; that's a separate step, upstream or downstream depending on whether
  you mask before or after reconstruction.
