# Eucalyptus D455 RGB-D capture

This is a separate collection workflow for single-camera Intel RealSense D455
captures. It does not write to or change the existing Kinect data under
`dataset/`.

## Ubuntu setup

1. Connect one D455 directly to a USB 3 port. Install the Intel RealSense
   SDK/udev rules for the Ubuntu release and verify that
   `rs-enumerate-devices` lists the D455. If device access is denied, apply the
   SDK's udev rules and reconnect the camera.
2. In the Python environment to be used for capture, install the runtime
   packages:

   ```bash
   python3 -m pip install numpy opencv-python pyrealsense2
   ```

3. From the CropCraft repository root, start the capture:

   ```bash
   python3 dataset_d455/capture_d455.py
   ```

If more than one D455 is connected, select the intended camera with
`--serial SERIAL_NUMBER`. Use `python3 dataset_d455/capture_d455.py --help` for
output-directory options.

Before collecting plants, verify one RGB/depth pair reaches the computer:

```bash
python3 dataset_d455/capture_d455.py --check-camera
```

This opens the camera, reports the aligned frame dimensions and nonzero depth
coverage/range, writes no files, then closes the stream.

## Per-plant capture sequence

The script prompts for the plant ID, species, capture date, and every
measurement column in the supplied `measurements.xlsx` sheet, as well as
pot-only mass. Species defaults to `Eucalyptus`; capture date defaults to
today. For other fields, press Enter to record `0` when the value is
unavailable. Measure total plant-plus-pot mass and pot-only mass separately;
the script records `net_plant_mass_g` as their difference when both readings
are nonzero, and records 0 when either reading is unavailable.

For each plant:

1. The D455 opens once and streams aligned color and depth.
2. Set the turntable to its starting position, 0 degrees.
3. For each prompted angle (0, 15, ... 345 degrees), make sure the plant and
   camera are still, then press Enter. Rotate the turntable manually by 15
   degrees after each capture. The final view is at 345 degrees; do not take a
   duplicate 360-degree view.
4. After 24 views, the stream closes. The script records the plant metadata
   and asks whether to begin another plant; a new plant opens a fresh stream.

## Output

Each plant is saved in `dataset_d455/plants/<plant_id>/`:

```text
rgb/000.png ... rgb/345.png       # color PNG
depth/000.png ... depth/345.png   # aligned raw Z16 depth PNG
capture_manifest.json
metadata.json
```

Angles in filenames are turntable angles in degrees. The manifest stores each
angle, the camera serial and firmware, depth scale, and color/depth intrinsics.
Multiply each raw depth value by `depth_scale_m_per_unit` in the manifest to
get distance in metres. Depth images are aligned to color coordinates.

`dataset_d455/ground_truth.csv` contains one row per completed plant with all
sheet fields, pot-only mass, the derived plant-only mass, and camera serial.
The image data and generated CSV are ignored by Git to keep large capture
outputs out of source control. A cancelled or failed sweep may leave a
partial plant folder; it is not added to the CSV, and the script will not
overwrite it.

## Notes for later processing

The saved color frames are lossless RGB PNGs. Depth is not colorized or
converted to millimetres; raw Z16 is retained alongside the per-camera scale
and alignment/intrinsic details for metric reconstruction and voxel-memory
workflows.
