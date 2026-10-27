# D455 + Kinect V2, captured together

One run, one turntable sweep, both cameras, 24 views at 15 degrees. The older
collections in `dataset/` (Kinect) and `dataset_d455/` (D455) are not touched.

```bash
python3 dataset_dual/capture_dual.py --simulate    # rehearse with no cameras (about 2 min)
python3 dataset_dual/capture_dual.py               # the real session
```

## Set-up

```
        backdrop (dark matte cloth, ~0.5 m behind the plant)
     ┌─────────────────────────────┐
     │            plant            │
     │        [ turntable ]        │
     └─────────────────────────────┘
                   ▲  1.1 m, same for both cameras
                   │
              ┌─────────┐   D455, portrait, lens ~0.75 m above the turntable top
              │  D455   │
              ├─────────┤   Kinect directly below it, landscape
              │ Kinect  │
              └─────────┘
```

* **Both cameras on the same side**, one above the other. Facing each other they
  would shine their infrared straight into each other's lens.
* **1.1 m from each lens to the turntable centre**, measured with a tape. Mark the
  tripod feet on the floor.
* **D455 in portrait, lens about 0.75 m above the turntable top, level.** That
  covers about 2.2 m of height at 1.1 m, so a 1.8 m eucalyptus fits. In the old
  landscape set-up at pot height only about 1.05 m above the pot was in frame,
  and 27 of 36 plants lost their tops.
* **Kinect in landscape** covers about 1.3 m of height at 1.1 m. For plants taller
  than that, turn it to portrait too and enter that in the session questions.
* **Separate USB 3 ports, ideally on different controllers** (`lsusb -t`). The
  D455 runs at 15 fps by default to leave bandwidth for the Kinect.
* **Raise the USB buffer for the Kinect** (once per boot):
  `echo 256 | sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb`
* **Indoors or shaded.** The Kinect's time-of-flight depth fails in direct sun.
* **Start the script early.** It opens the Kinect at the start; its depth drifts
  for about 15 minutes while it warms. Weigh and measure the first plant during
  that time. The warm-up minutes are stored with every plant.

## What a session asks and checks

1. **Session questions, once:** operator, location, lighting, backdrop, and for
   each camera the distance, lens height and orientation. Type `g` between
   plants to change them; a new session record starts.
2. **Interference test, once:** each camera's depth with the other off, then on.
   Valid-pixel share and frame-to-frame noise go into `sessions/<id>.json`. The
   D455 projector is always off while the Kinect is read. If the test says the
   Kinect harms the D455, restart with `--kinect-pause`.
3. **Per plant, before the sweep:** tape height, canopy diameters, caliper stem
   diameter at a lower and an upper point, whole potted plant on the scale.
4. **Framing check:** both cameras look once; any image edge the plant touches is
   drawn red in the preview (and in `preview/` in the plant folder). The first
   time the D455 is in portrait it asks which side of the image the top of the
   plant is on, so later steps know which way is up. Move a camera and type `r`
   to recheck.
5. **The sweep:** Enter at each of the 24 angles. A view whose top is cut off is
   flagged as it is saved.
6. **After the sweep:** cut, weigh the shoot and the pot with soil. The script
   computes net mass by difference and warns when it disagrees with the shoot
   on the scale by more than 5% (or 10 g), while you can still re-weigh.

Leave anything not measured **blank** (or type `na`): it is stored empty, never
as 0. Dry mass is left empty until the plants have actually been oven-dried.

## Output

```
dataset_dual/
  ground_truth.csv                     one row per completed plant
  sessions/<session_id>.json           set-up, interference numbers, camera details
  plants/<plant_id>/
    metadata.json                      measurements, session, framing, per-view flags
    preview/framing_d455.jpg, framing_kinect.jpg
    d455/{rgb,depth}/000.png ...       same layout and manifest as dataset_d455
    d455/capture_manifest.json         + orientation and up_edge
    kinect/rgb/, kinect/depth/         512x424 registered pair, as in dataset/
    kinect/color_full/                 1920x1080 colour (--full-colour-format jpg to save space)
    kinect/depth_full/                 depth registered to the 1920x1080 colour, uint16 mm
    kinect/capture_manifest.json       + intrinsics, warm-up minutes, on_old_kinect_grid
```

`up_edge` in each manifest says which image edge is up (`top`, `left` or
`right`); rotate frames so it is at the top before reconstruction. Kinect views
with `on_old_kinect_grid: true` are the 12 at 30-degree steps, matching the
view count of `dataset/`.

A plant takes about 170 MB with lossless colour. Track the plant folders with
DVC, not git.

Plant IDs default to `EUC-<date>-NN` and are checked against `dataset/`,
`dataset_d455/` and this folder, ignoring case, so an ID can never mean two
different plants.
