"""Capture every plant with the RealSense D455 and the Kinect V2 in one run.

    python3 dataset_dual/capture_dual.py                 # a real session, both cameras
    python3 dataset_dual/capture_dual.py --simulate      # rehearse the whole flow, no cameras
    python3 dataset_dual/capture_dual.py --check-only    # framing and interference checks, no plant

Both cameras face the plant from the same side, one above the other, and the
turntable turns the plant 15 degrees between views. At every view the D455 is
read first with its projector on, then its projector is switched off and the
Kinect is read, then the projector goes back on. The two sensors therefore
never fire into each other's frame (the Kinect's time-of-flight light is the
one exception, see --kinect-pause and the interference test).

What this adds over dataset_d455/capture_d455.py and rig_calibration/collect_specimen.py:

* one run, one turntable sweep, both sensors, the same 24 angles;
* the capture geometry (distance, camera height, orientation, backdrop) asked
  once per session and stored with every plant;
* a framing check before each plant that finds the floor edge in the depth image
  and refuses to call a plant framed while it touches the opposite (top) edge,
  the failure that cut the top off 27 of the 36 earlier D455 plants;
* the Kinect's full 1920x1080 colour and the depth registered to it, kept beside
  the usual 512x424 registered pair;
* an interference test at the start of the session, with its numbers saved;
* non-destructive measurements before the sweep, destructive ones after it;
* unknown values stored as blank, never as 0;
* plant IDs checked against every existing dataset in the repository.

Layout, one folder per plant:

    dataset_dual/plants/<plant_id>/
        metadata.json                       everything about the plant and the session
        preview/framing_d455.jpg, framing_kinect.jpg
        d455/rgb/000.png ... 345.png        same layout and manifest as dataset_d455
        d455/depth/000.png ...              aligned raw Z16
        d455/capture_manifest.json
        d455/metadata.json
        kinect/rgb/000.png ...              512x424 colour registered to depth (gamma as before)
        kinect/depth/000.png ...            512x424 depth, uint16 mm
        kinect/color_full/000.png ...       1920x1080 colour as delivered
        kinect/depth_full/000.png ...       depth registered to the full colour, uint16 mm
        kinect/capture_manifest.json
    dataset_dual/ground_truth.csv
    dataset_dual/sessions/<session_id>.json
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import faulthandler
import json
import math
import re
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

DATASET_DIR = Path(__file__).resolve().parent
REPO_ROOT = DATASET_DIR.parent
for _path in (REPO_ROOT, REPO_ROOT / "rig_calibration"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

PLANTS_DIR = DATASET_DIR / "plants"
SESSIONS_DIR = DATASET_DIR / "sessions"
GROUND_TRUTH_CSV = DATASET_DIR / "ground_truth.csv"

# Other datasets whose plant IDs must not be reused.
OTHER_ID_SOURCES = (
    (REPO_ROOT / "dataset_d455" / "plants", REPO_ROOT / "dataset_d455" / "ground_truth.csv"),
    (REPO_ROOT / "dataset" / "plants", REPO_ROOT / "dataset" / "ground_truth.csv"),
)

ANGLES_DEG = tuple(range(0, 360, 15))
D455_SETTLE_FRAMES = 6        # frames dropped after the projector is switched
KINECT_FLUSH_FRAMES = 2       # Kinect frames dropped so the one kept is fresh
KINECT_WARMUP_MINUTES = 15.0  # time-of-flight depth drifts while the sensor warms
KINECT_OPEN_TIMEOUT_S = 45    # libfreenect2 can hang forever opening a wedged Kinect
KINECT_CALL_TIMEOUT_S = 30    # ... or starting/stopping one that stopped responding
KINECT_FRAME_TIMEOUT_S = 10   # no Kinect frame for this long = not streaming
D455_RESET_WAIT_S = 30        # time for the D455 to come back after a hardware reset
INTERFERENCE_FRAMES = 15
ORIENTATIONS = ("landscape", "portrait")
EDGES = ("top", "bottom", "left", "right")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}

CSV_FIELDS = (
    "plant_id", "species_breed", "capture_date", "session_id",
    "total_fresh_with_pot_g", "height_above_rim_mm",
    "canopy_diameter_max_mm", "canopy_diameter_perp_mm",
    "stem_diameter_lower_mm", "stem_diameter_upper_mm", "stem_diameter_mean_mm",
    "harvest_time", "cut_height_above_rim_mm", "shoot_fresh_mass_g",
    "pot_plus_soil_mass_g", "net_plant_mass_g", "minutes_cut_to_weighing",
    "leaf_count", "shoot_dry_mass_g", "watering_state", "measured_by",
    "d455_serial", "kinect_serial",
    "d455_distance_mm", "d455_height_mm", "d455_orientation",
    "kinect_distance_mm", "kinect_height_mm", "kinect_orientation",
    "backdrop", "d455_top_clipped_views", "kinect_top_clipped_views", "notes",
)

# Asked before the sweep: nothing here needs the plant cut.
BEFORE_FIELDS = (
    ("height_above_rim_mm", "Plant height above the pot rim, tape (mm)", "number"),
    ("canopy_diameter_max_mm", "Canopy diameter, widest (mm)", "number"),
    ("canopy_diameter_perp_mm", "Canopy diameter, perpendicular to that (mm)", "number"),
    ("stem_diameter_lower_mm", "Stem diameter, caliper, lower (mm)", "number"),
    ("stem_diameter_upper_mm", "Stem diameter, caliper, upper (mm)", "number"),
    ("total_fresh_with_pot_g", "Scale: whole potted plant before cutting (g)", "number"),
    ("watering_state", "Watering state (e.g. watered this morning)", "text"),
)
# Asked after the sweep, once the plant has been cut and weighed.
AFTER_FIELDS = (
    ("harvest_time", "Harvest time (HH:MM)", "text"),
    ("cut_height_above_rim_mm", "Cut height above the pot rim (mm)", "number"),
    ("shoot_fresh_mass_g", "Scale: cut shoot alone (g)", "number"),
    ("pot_plus_soil_mass_g", "Scale: pot and soil left after cutting (g)", "number"),
    ("minutes_cut_to_weighing", "Minutes from cutting to weighing", "number"),
    ("leaf_count", "Leaf count", "integer"),
)


# ---------------------------------------------------------------------------
# Pure helpers (tested in tests/test_dual_capture.py)
# ---------------------------------------------------------------------------

def parse_optional(raw: str, kind: str = "number") -> float | int | str | None:
    """Blank, 'na' or '-' mean unknown (None). Numbers must be finite and not negative."""
    text = (raw or "").strip()
    if text.lower() in {"", "na", "n/a", "-", "none", "unknown"}:
        return None
    if kind == "text":
        return text
    value = int(text) if kind == "integer" else float(text)
    if not math.isfinite(value) or value < 0:
        raise ValueError("must be a finite number, zero or more")
    return value


def validate_plant_id(plant_id: str) -> str:
    plant_id = (plant_id or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", plant_id):
        raise ValueError("use letters, numbers, underscores or hyphens only")
    return plant_id


def ids_in(plants_dir: Path, csv_path: Path) -> set[str]:
    found: set[str] = set()
    if plants_dir.exists():
        found |= {p.name for p in plants_dir.iterdir() if p.is_dir()}
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            found |= {row["plant_id"] for row in csv.DictReader(handle) if row.get("plant_id")}
    return found


def known_ids(plants_dir: Path = PLANTS_DIR, csv_path: Path = GROUND_TRUTH_CSV,
              others=OTHER_ID_SOURCES) -> set[str]:
    """Every plant ID already used here or in the other datasets, compared case-insensitively."""
    found = ids_in(plants_dir, csv_path)
    for other_dir, other_csv in others:
        found |= ids_in(other_dir, other_csv)
    return {name.upper() for name in found}


def default_plant_id(taken: set[str], day: date | None = None, prefix: str = "EUC") -> str:
    stamp = (day or date.today()).strftime("%Y%m%d")
    number = 1
    while f"{prefix}-{stamp}-{number:02d}".upper() in taken:
        number += 1
    return f"{prefix}-{stamp}-{number:02d}"


def net_mass(total_g: float | None, tare_g: float | None) -> float | None:
    """Plant mass from the potted weight minus the pot and soil left after cutting."""
    if total_g is None or tare_g is None:
        return None
    if tare_g > total_g:
        raise ValueError(f"pot and soil ({tare_g} g) weigh more than the whole plant ({total_g} g)")
    return round(total_g - tare_g, 2)


def mass_disagreement(net_g: float | None, shoot_g: float | None) -> str | None:
    """The two routes to plant mass should agree; say so when they do not."""
    if net_g is None or shoot_g is None:
        return None
    gap = abs(net_g - shoot_g)
    if gap > max(10.0, 0.05 * max(net_g, shoot_g)):
        return (f"net by difference is {net_g:g} g but the cut shoot weighed {shoot_g:g} g "
                f"({gap:g} g apart). Re-weigh before the plant dries.")
    return None


def mean_or_none(*values: float | None) -> float | None:
    known = [v for v in values if v is not None]
    return round(sum(known) / len(known), 2) if known else None


def estimate_plant_depth(depth_m: np.ndarray) -> float | None:
    """The plant's distance when the session distance is unknown: the near layer of the central band."""
    height, width = depth_m.shape
    band = depth_m[int(0.15 * height):int(0.85 * height), int(0.4 * width):int(0.6 * width)]
    values = band[(band > 0.25) & (band < 3.0)]
    if values.size < 50:
        return None
    # The nearest depth layer that holds a real share of the band, not a few flying pixels.
    counts, edges = np.histogram(values, bins=np.arange(0.25, 3.0 + 0.02, 0.02))
    enough = np.where(counts >= max(30, 0.005 * values.size))[0]
    return float((edges[enough[0]] + edges[enough[0] + 1]) / 2) if enough.size else None


def framing(depth_m: np.ndarray, plant_depth_m: float, up_edge: str | None = None,
            band_m: float = 0.4, edge_px: int = 6, min_hits: int = 4) -> dict[str, Any]:
    """Does the plant touch an edge of the frame, and is that edge the top.

    Pixels within ``band_m`` of the plant's distance are the plant, its pot, the
    turntable and any floor right under them. ``up_edge`` says which image edge
    is up (``top`` in landscape; ``left`` or ``right`` in portrait). Without it,
    the edge most covered by those pixels is taken as the floor when it is
    clearly covered, and the opposite edge as the top. When neither is known,
    any edge the plant touches is reported as a cut-off.
    """
    valid = depth_m > 0
    near = valid & (np.abs(depth_m - plant_depth_m) < band_m)
    rows = np.where(valid.mean(axis=1) > 0.2)[0]
    cols = np.where(valid.mean(axis=0) > 0.2)[0]
    if rows.size == 0 or cols.size == 0:
        return {"floor_edge": None, "top_edge": up_edge, "top_clipped": None, "touching": [],
                "note": "no valid depth"}
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    strips = {
        "top": near[r0:r0 + edge_px, c0:c1], "bottom": near[r1 - edge_px:r1, c0:c1],
        "left": near[r0:r1, c0:c0 + edge_px].T, "right": near[r0:r1, c1 - edge_px:c1].T,
    }
    cover = {edge: float(strip.mean()) for edge, strip in strips.items()}

    def central_hits(edge: str) -> int:
        strip = strips[edge]
        span = strip.shape[1]
        return int(strip[:, int(0.2 * span):int(0.8 * span)].sum())

    if up_edge:
        top, floor = up_edge, OPPOSITE[up_edge]
    else:
        floor = max(cover, key=cover.get)
        floor = floor if cover[floor] >= 0.25 else None
        top = OPPOSITE[floor] if floor else None
    touching = [e for e in EDGES if e != floor and central_hits(e) >= min_hits]
    return {
        "floor_edge": floor,
        "top_edge": top,
        "top_clipped": (top in touching) if top else bool(touching),
        "touching": touching,
        "edge_cover": {k: round(v, 3) for k, v in cover.items()},
        "plant_depth_m": round(float(plant_depth_m), 3),
        "note": "" if top else "which edge is up is unknown; any edge contact is reported",
    }


def depth_stats(stack_m: np.ndarray) -> dict[str, float]:
    """Valid share and temporal noise of a burst of depth frames over the central region."""
    n, height, width = stack_m.shape
    roi = stack_m[:, int(0.2 * height):int(0.8 * height), int(0.3 * width):int(0.7 * width)]
    valid = roi > 0
    steady = valid.mean(axis=0) >= 0.8
    noise = float("nan")
    if steady.any():
        masked = np.where(valid, roi, np.nan)[:, steady]
        noise = float(np.nanmedian(np.nanstd(masked, axis=0)) * 1000)
    return {"valid_pct": round(float(valid.mean() * 100), 2), "noise_mm": round(noise, 3)}


def interference_verdict(alone: dict, with_other: dict, sensor: str) -> tuple[bool, str]:
    drop = alone["valid_pct"] - with_other["valid_pct"]
    rise = with_other["noise_mm"] / alone["noise_mm"] if alone["noise_mm"] > 0 else 1.0
    harmed = drop > 3.0 or rise > 1.5
    text = (f"{sensor}: valid {alone['valid_pct']:.1f}% -> {with_other['valid_pct']:.1f}%, "
            f"noise {alone['noise_mm']:.2f} -> {with_other['noise_mm']:.2f} mm"
            + ("  INTERFERENCE" if harmed else "  ok"))
    return harmed, text


def kinect_view_for_old_rig(angle: int) -> bool:
    """The Kinect views on the 30-degree grid of the earlier 12-view dataset."""
    return angle % 30 == 0


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

Ask = Callable[[str], str]


def ask_value(ask: Ask, label: str, kind: str = "number", default: str = "") -> Any:
    hint = f" [{default}]" if default else " [blank = unknown]"
    while True:
        raw = ask(f"  {label}{hint}: ")
        raw = raw if raw.strip() else default
        try:
            return parse_optional(raw, kind)
        except ValueError as exc:
            print(f"    {exc}; try again, or leave blank if unknown.")


def ask_choice(ask: Ask, label: str, choices: tuple[str, ...], default: str) -> str:
    while True:
        raw = ask(f"  {label} {'/'.join(choices)} [{default}]: ").strip().lower() or default
        if raw in choices:
            return raw
        print(f"    choose one of {', '.join(choices)}")


def ask_session(ask: Ask, previous: dict | None = None) -> dict[str, Any]:
    p = previous or {}
    print("\nSession: how the cameras are set up. Asked once; type g between plants to change it.")
    session = {
        "operator": ask_value(ask, "Operator", "text", p.get("operator") or ""),
        "location": ask_value(ask, "Location", "text", p.get("location") or ""),
        "lighting": ask_value(ask, "Lighting and sky (e.g. greenhouse, overcast)", "text",
                              p.get("lighting") or ""),
        "backdrop": ask_value(ask, "Backdrop behind the plant (e.g. black matte cloth, none)", "text",
                              p.get("backdrop") or ""),
    }
    for cam, default_orient in (("d455", "portrait"), ("kinect", "landscape")):
        print(f"  {cam.upper()}: tape from the camera front to the turntable centre.")
        session[f"{cam}_distance_mm"] = ask_value(
            ask, f"{cam.upper()} distance (mm)", "number", _str(p.get(f"{cam}_distance_mm")))
        session[f"{cam}_height_mm"] = ask_value(
            ask, f"{cam.upper()} lens height above the turntable top (mm)", "number",
            _str(p.get(f"{cam}_height_mm")))
        session[f"{cam}_orientation"] = ask_choice(
            ask, f"{cam.upper()} orientation", ORIENTATIONS, p.get(f"{cam}_orientation") or default_orient)
    return session


def _str(value: Any) -> str:
    return "" if value is None else f"{value:g}" if isinstance(value, float) else str(value)


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------

class D455:
    """The existing D455 wrapper, plus the High Accuracy preset and projector control."""

    def __init__(self, serial: str | None, fps: int) -> None:
        self.serial, self.fps = serial, fps

    @staticmethod
    def _base():
        from dataset_d455 import capture_d455
        return capture_d455

    def open(self) -> None:
        try:
            self._open_once()
        except RuntimeError as exc:
            # A D455 left streaming by a killed run fails here ("Failed to set UVC
            # probe control", "xioctl(VIDIOC_S_FMT) failed"). Unlike the Kinect it can
            # be reset from software. Missing or ambiguous cameras are not retried.
            if getattr(getattr(self, "cam", None), "pipeline", None) is None:
                raise
            print(f"  The D455 would not start ({exc}).\n  Resetting it and trying once more ...")
            self._reset()
            self._open_once()

    def _open_once(self) -> None:
        self.close()
        base = self._base()
        base.FPS = self.fps
        self.cam = base.D455Camera(requested_serial=self.serial)
        self.cam.open()
        rs = self.cam._rs
        device = self.cam.pipeline.get_active_profile().get_device()
        self.depth_sensor = device.first_depth_sensor()
        preset = "default"
        try:
            if self.depth_sensor.supports(rs.option.visual_preset):
                self.depth_sensor.set_option(rs.option.visual_preset,
                                             float(int(rs.rs400_visual_preset.high_accuracy)))
                preset = "high_accuracy"
        except Exception as exc:  # noqa: BLE001
            print(f"  Could not set the High Accuracy preset: {exc}")
        self.set_emitter(True)
        self.info = dict(self.cam.camera_info, visual_preset=preset)

    @property
    def serial_number(self) -> str:
        return self.info["serial_number"]

    def set_emitter(self, on: bool) -> None:
        rs = self.cam._rs
        if self.depth_sensor.supports(rs.option.emitter_enabled):
            self.depth_sensor.set_option(rs.option.emitter_enabled, 1.0 if on else 0.0)
        for _ in range(D455_SETTLE_FRAMES):
            self.cam.pipeline.wait_for_frames(5000)

    def grab(self) -> tuple[np.ndarray, np.ndarray]:
        return self.cam.capture()

    def depth_m(self, depth_raw: np.ndarray) -> np.ndarray:
        return depth_raw.astype(np.float32) * float(self.cam.depth_scale_m)

    def _reset(self) -> None:
        rs = self.cam._rs
        serial = self.cam.camera_info.get("serial_number") or self.serial
        devices = [d for d in rs.context().query_devices() if "D455" in d.get_info(rs.camera_info.name)
                   and (not serial or d.get_info(rs.camera_info.serial_number) == serial)]
        if not devices:
            raise RuntimeError("the D455 disappeared; unplug and replug it")
        serial = devices[0].get_info(rs.camera_info.serial_number)
        devices[0].hardware_reset()
        time.sleep(3)  # it drops off USB before coming back
        deadline = time.monotonic() + D455_RESET_WAIT_S
        while time.monotonic() < deadline:
            if any(d.get_info(rs.camera_info.serial_number) == serial for d in rs.context().query_devices()):
                time.sleep(2)
                return
            time.sleep(1)
        raise RuntimeError(f"the D455 did not come back within {D455_RESET_WAIT_S} s of a reset; replug it")

    def close(self) -> None:
        if getattr(self, "cam", None):
            try:
                self.cam.close()
            except Exception:  # noqa: BLE001
                pass


@contextlib.contextmanager
def kinect_watchdog(seconds: float):
    """Exit the process if a libfreenect2 call does not return within `seconds`.

    A wedged Kinect can block inside libfreenect2's native code (openDevice, start,
    stop, close) with no timeout, holding the GIL, so neither a Python thread nor a
    signal handler can interrupt it. faulthandler's timer runs on its own C thread:
    it prints where Python was stuck and calls _exit(1). This is the last resort;
    frame waits use wait_kinect_frames, which fails cleanly instead.
    """
    faulthandler.dump_traceback_later(seconds, exit=True, file=sys.stderr)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()


REPLUG_KINECT = ("Unplug the Kinect's USB cable (or power brick) for 5 s, plug it back in, "
                 "wait 10 s and rerun. Check the adapter box LED is lit.")


class KinectNotStreaming(RuntimeError):
    """The Kinect is open but no frames arrive. Only a power cycle brings it back."""


def wait_kinect_frames(listener, count: int, what: str,
                       timeout_s: float = KINECT_FRAME_TIMEOUT_S) -> Any:
    """Wait for `count` frame sets, releasing all but the last, which is returned.

    pylibfreenect2's waitForNewFrame(milliseconds=...) ignores the timeout and
    blocks forever on a Kinect that stopped streaming, so poll hasNewFrame instead.
    """
    frames = None
    for _ in range(count):
        if frames is not None:
            listener.release(frames)
            frames = None
        deadline = time.monotonic() + timeout_s
        while not listener.hasNewFrame():
            if time.monotonic() > deadline:
                raise KinectNotStreaming(f"no Kinect frame for {timeout_s:.0f} s {what}. {REPLUG_KINECT}")
            time.sleep(0.005)
        frames = listener.waitForNewFrame()
    return frames


class Kinect:
    """One Kinect V2 held open for the whole session, with full-resolution colour kept."""

    def __init__(self, serial: str | None, cpu_pipeline: bool = False) -> None:
        self.serial = serial
        self.cpu_pipeline = cpu_pipeline  # needed by --kinect-pause, see KinectDevice.open
        self.opened_at: float | None = None

    def open(self) -> None:
        import collect_specimen as cs
        if not cs.KINECT_AVAILABLE:
            raise RuntimeError("pylibfreenect2 is not installed; see rig_calibration/install_libfreenect2.sh")
        import pylibfreenect2 as lf
        try:  # keep libfreenect2's packet logs from burying the prompts
            lf.setGlobalLogger(lf.createConsoleLogger(lf.LoggerLevel.Error))
        except Exception:  # noqa: BLE001
            pass
        self.lf, self.cs = lf, cs
        if not self.serial:
            probe = lf.Freenect2()
            count = probe.enumerateDevices()
            if count == 0:
                raise RuntimeError("no Kinect V2 found; check its power brick and USB 3 port")
            serials = [probe.getDeviceSerialNumber(i).decode() for i in range(count)]
            del probe  # one libfreenect2 context at a time
            preferred = [s for s in serials if s in (cs.CAM_A_SERIAL, cs.CAM_B_SERIAL)]
            self.serial = (preferred or serials)[0]
            print(f"  Kinect serial {self.serial} (found {', '.join(serials)})")
        print("  Opening the Kinect ...")
        self.device = cs.KinectDevice("kinect", self.serial)
        with kinect_watchdog(KINECT_OPEN_TIMEOUT_S):
            self.device.open(warmup_frames=0, cpu_pipeline=self.cpu_pipeline)
        self.device.listener.release(
            wait_kinect_frames(self.device.listener, 30, "after opening the Kinect"))
        self.opened_at = time.time()
        ir, colour = self.device.device.getIrCameraParams(), self.device.device.getColorCameraParams()
        pick = lambda obj, names: {n: float(getattr(obj, n)) for n in names if hasattr(obj, n)}  # noqa: E731
        self.info = {
            "model": "Microsoft Kinect V2 (libfreenect2)",
            "serial_number": self.serial,
            "ir_intrinsics": dict(pick(ir, ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "p1", "p2")),
                                  width=512, height=424),
            "color_intrinsics": dict(pick(colour, ("fx", "fy", "cx", "cy")), width=1920, height=1080),
            "depth_encoding": "uint16 millimetres; 0 = no return",
            "depth_decoder": "libfreenect2 " + ("CPU" if self.cpu_pipeline else "OpenGL"),
            "rgb_note": "rgb/ is colour registered to the 512x424 depth, gamma "
                        f"{cs.RGB_GAMMA} as in dataset/; color_full/ is the raw 1920x1080 frame",
            "depth_full_note": "depth_full/ is depth registered to color_full/ (libfreenect2 bigdepth)",
        }

    @property
    def serial_number(self) -> str:
        return self.serial

    def warm_minutes(self) -> float:
        return 0.0 if self.opened_at is None else (time.time() - self.opened_at) / 60

    def pause(self) -> None:
        with kinect_watchdog(KINECT_CALL_TIMEOUT_S):
            self.device.device.stop()

    def resume(self) -> None:
        with kinect_watchdog(KINECT_CALL_TIMEOUT_S):
            self.device.device.start()
        self.device.listener.release(
            wait_kinect_frames(self.device.listener, 5, "after restarting the Kinect (--kinect-pause)"))

    def grab(self) -> dict[str, np.ndarray]:
        lf, cs, dev = self.lf, self.cs, self.device
        frames = wait_kinect_frames(dev.listener, KINECT_FLUSH_FRAMES + 1, "while capturing")
        try:
            colour = frames[lf.FrameType.Color]
            depth = frames[lf.FrameType.Depth]
            undistorted = lf.Frame(512, 424, 4)
            registered = lf.Frame(512, 424, 4)
            bigdepth = lf.Frame(1920, 1082, 4)
            dev.registration.apply(colour, depth, undistorted, registered, enable_filter=True,
                                   bigdepth=bigdepth, color_depth_map=None)
            full = colour.asarray()[..., :3].copy()
            big = bigdepth.asarray(np.float32)[1:-1].copy()
            big[~np.isfinite(big)] = 0
            out = {
                "rgb": cs.gamma_correct(registered.asarray(np.uint8)[..., :3].copy()),
                "depth": np.clip(undistorted.asarray(np.float32), 0, 65535).astype(np.uint16),
                "color_full": full,
                "depth_full": np.clip(big, 0, 65535).astype(np.uint16),
            }
        finally:
            dev.listener.release(frames)
        return out

    @staticmethod
    def depth_m(depth_mm: np.ndarray) -> np.ndarray:
        return depth_mm.astype(np.float32) / 1000.0

    def close(self) -> None:
        if getattr(self, "device", None):
            with kinect_watchdog(KINECT_CALL_TIMEOUT_S):
                self.device.close()


# Simulated cameras, so the whole flow can be rehearsed (and tested) without hardware.

def _scene(height: int, width: int, focal: float, plant_z: float, plant_h_m: float,
           cam_h_m: float, portrait: bool, rng: np.random.Generator) -> np.ndarray:
    if portrait:
        height, width = width, height
    depth = np.full((height, width), plant_z + 0.6, np.float32)            # backdrop
    cy = height / 2
    rows = np.arange(height)
    floor_row = int(cy + focal * cam_h_m / plant_z)                         # turntable top
    depth[rows >= min(floor_row, height), :] = plant_z - 0.1               # floor and turntable
    top_row = int(cy - focal * (plant_h_m - cam_h_m) / plant_z)
    cx = width // 2
    stem = max(2, int(focal * 0.012 / plant_z))
    depth[max(top_row, 0):min(floor_row, height), cx - stem:cx + stem] = plant_z
    depth += rng.normal(0, 0.002, depth.shape).astype(np.float32)
    depth[:4] = 0                                                           # an invalid border row band
    return np.rot90(depth, -1) if portrait else depth


class SimD455:
    def __init__(self, plant_h_m: float, portrait: bool) -> None:
        self.plant_h_m, self.portrait = plant_h_m, portrait
        self.rng = np.random.default_rng(0)
        self.info = {"model": "simulated D455", "serial_number": "SIM-D455", "visual_preset": "sim",
                     "depth_scale_m_per_unit": 0.001}
        self.emitter = True

    serial_number = property(lambda self: "SIM-D455")

    def open(self) -> None: ...
    def close(self) -> None: ...

    def set_emitter(self, on: bool) -> None:
        self.emitter = on

    def grab(self) -> tuple[np.ndarray, np.ndarray]:
        depth = _scene(720, 1280, 646, 1.1, self.plant_h_m, 0.75 if self.portrait else 0.1,
                       self.portrait, self.rng)
        colour = np.dstack([np.clip(depth * 120, 0, 255).astype(np.uint8)] * 3)
        return colour, np.clip(depth * 1000, 0, 65535).astype(np.uint16)

    @staticmethod
    def depth_m(raw: np.ndarray) -> np.ndarray:
        return raw.astype(np.float32) / 1000.0


class SimKinect:
    def __init__(self, plant_h_m: float) -> None:
        self.plant_h_m = plant_h_m
        self.rng = np.random.default_rng(1)
        self.opened_at = time.time()
        self.info = {"model": "simulated Kinect V2", "serial_number": "SIM-KINECT"}

    serial_number = property(lambda self: "SIM-KINECT")

    def open(self) -> None: ...
    def close(self) -> None: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...

    def warm_minutes(self) -> float:
        return 99.0

    def grab(self) -> dict[str, np.ndarray]:
        depth = _scene(424, 512, 366, 1.1, self.plant_h_m, 0.6, False, self.rng)
        mm = np.clip(depth * 1000, 0, 65535).astype(np.uint16)
        big = cv2.resize(mm, (1920, 1080), interpolation=cv2.INTER_NEAREST)
        grey = np.dstack([np.clip(depth * 120, 0, 255).astype(np.uint8)] * 3)
        return {"rgb": grey, "depth": mm, "color_full": cv2.resize(grey, (1920, 1080)), "depth_full": big}

    @staticmethod
    def depth_m(raw: np.ndarray) -> np.ndarray:
        return raw.astype(np.float32) / 1000.0


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _plant_z(depth: np.ndarray, distance_mm: float | None) -> float | None:
    return distance_mm / 1000 if distance_mm else estimate_plant_depth(depth)


def up_edge_of(session: dict, cam: str) -> str | None:
    if session.get(f"{cam}_orientation") == "landscape":
        return "top"
    return session.get(f"{cam}_up_edge")


def check_framing(d455, kinect, session: dict, out_dir: Path | None, preview: bool,
                  ask: Ask | None = None) -> dict:
    colour, raw = d455.grab()
    d455.set_emitter(False)
    k = kinect.grab()
    d455.set_emitter(True)
    views = {"d455": (d455.depth_m(raw), colour, session.get("d455_distance_mm")),
             "kinect": (kinect.depth_m(k["depth"]), k["rgb"], session.get("kinect_distance_mm"))}
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, (_, image, _) in views.items():
            cv2.imwrite(str(out_dir / f"framing_{name}.jpg"), image)
    if preview:
        _show({}, colour, k["rgb"])
    for name in views:
        if ask and session.get(f"{name}_orientation") == "portrait" and not session.get(f"{name}_up_edge"):
            where = f" (or open {out_dir / f'framing_{name}.jpg'})" if out_dir else ""
            session[f"{name}_up_edge"] = ask_choice(
                ask, f"{name.upper()} is in portrait. In the preview{where}, the top of the plant is at the",
                ("left", "right"), "left")
    result = {}
    for name, (depth, image, distance) in views.items():
        z = _plant_z(depth, distance)
        frame = framing(depth, z, up_edge_of(session, name)) if z else {
            "floor_edge": None, "top_edge": None, "top_clipped": None, "touching": [],
            "note": "could not find the plant's distance"}
        result[name] = frame
        touching = frame.get("touching") or []
        if frame.get("top_clipped"):
            state = (f"TOP CUT OFF ({frame['top_edge']} edge)" if frame.get("top_edge")
                     else f"CUT OFF at the {' and '.join(touching)} edge")
        elif frame.get("top_clipped") is False:
            state = "framed" + (f", but touching the {' and '.join(touching)} edge" if touching else "")
        else:
            state = "unclear: " + frame.get("note", "")
        print(f"  {name.upper():6} {state}")
        if out_dir is not None:
            cv2.imwrite(str(out_dir / f"framing_{name}.jpg"), _annotate(image, frame))
    if preview:
        _show(result, colour, k["rgb"])
    return result


def _annotate(image: np.ndarray, frame: dict) -> np.ndarray:
    shown = image.copy()
    h, w = shown.shape[:2]
    lines = {"top": ((0, 0), (w - 1, 0)), "bottom": ((0, h - 1), (w - 1, h - 1)),
             "left": ((0, 0), (0, h - 1)), "right": ((w - 1, 0), (w - 1, h - 1))}
    for edge in frame.get("touching") or []:
        cv2.line(shown, *lines[edge], (0, 0, 255), 12)
    if frame.get("top_edge") and frame["top_edge"] not in (frame.get("touching") or []):
        cv2.line(shown, *lines[frame["top_edge"]], (0, 200, 0), 12)
    return shown


def _show(result: dict, d455_rgb: np.ndarray, kinect_rgb: np.ndarray) -> None:
    try:
        left = _annotate(d455_rgb, result.get("d455", {}))
        right = _annotate(kinect_rgb, result.get("kinect", {}))
        scale = 540 / left.shape[0]
        left = cv2.resize(left, (int(left.shape[1] * scale), 540))
        right = cv2.resize(right, (int(right.shape[1] * 540 / right.shape[0]), 540))
        cv2.imshow("framing: D455 | Kinect (red edge = plant cut off, green = top, clear)", np.hstack([left, right]))
        cv2.waitKey(500)
    except cv2.error:
        pass


def interference_test(d455, kinect_factory) -> tuple[Any, dict]:
    """Measure each sensor with the other off and on. Opens the Kinect and returns it."""
    print("\nInterference test: keep the plant still on the turntable (about 20 s).")
    burst = lambda grab: np.stack([grab() for _ in range(INTERFERENCE_FRAMES)])  # noqa: E731
    d455_alone = depth_stats(burst(lambda: d455.depth_m(d455.grab()[1])))
    kinect = kinect_factory()
    kinect.open()
    d455_with = depth_stats(burst(lambda: d455.depth_m(d455.grab()[1])))
    k_with = depth_stats(burst(lambda: kinect.depth_m(kinect.grab()["depth"])))
    d455.set_emitter(False)
    k_alone = depth_stats(burst(lambda: kinect.depth_m(kinect.grab()["depth"])))
    d455.set_emitter(True)
    d_harm, d_text = interference_verdict(d455_alone, d455_with, "D455 with the Kinect running")
    k_harm, k_text = interference_verdict(k_alone, k_with, "Kinect with the D455 projector on")
    print("  " + d_text + "\n  " + k_text)
    if d_harm:
        print("  The Kinect degrades the D455 here. Restart with --kinect-pause so the Kinect is "
              "stopped while the D455 is read.")
    print("  (The capture always switches the D455 projector off while the Kinect is read.)")
    return kinect, {"d455_alone": d455_alone, "d455_with_kinect": d455_with,
                    "kinect_alone": k_alone, "kinect_with_d455_projector": k_with,
                    "d455_harmed": d_harm, "kinect_harmed": k_harm}


# ---------------------------------------------------------------------------
# One plant
# ---------------------------------------------------------------------------

def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"could not write {path}")


def sweep(plant_dir: Path, d455, kinect, ask: Ask, kinect_pause: bool,
          session: dict, full_ext: str = "png") -> tuple[list[dict], dict]:
    """All 24 views, both cameras. Manifests are rewritten after every view."""
    frames: list[dict] = []
    clipped = {"d455": 0, "kinect": 0}
    z = {"d455": session.get("d455_distance_mm"), "kinect": session.get("kinect_distance_mm")}
    for index, angle in enumerate(ANGLES_DEG, start=1):
        while True:
            answer = ask(f"\nView {index}/{len(ANGLES_DEG)} at {angle:03d} deg. Turntable set and "
                         "plant still? Enter to capture, q to abandon this plant: ").strip().lower()
            if answer in {"", "q"}:
                break
        if answer == "q":
            raise KeyboardInterrupt("plant abandoned")
        stamp = datetime.now(timezone.utc).isoformat()
        if kinect_pause:
            kinect.pause()
        colour, raw = d455.grab()
        if kinect_pause:
            kinect.resume()
        d455.set_emitter(False)
        k = kinect.grab()
        d455.set_emitter(True)

        name = f"{angle:03d}.png"
        write_image(plant_dir / "d455" / "rgb" / name, colour)
        write_image(plant_dir / "d455" / "depth" / name, raw)
        full_name = f"{angle:03d}.{full_ext}"
        for key in ("rgb", "depth", "depth_full"):
            write_image(plant_dir / "kinect" / key / name, k[key])
        write_image(plant_dir / "kinect" / "color_full" / full_name, k["color_full"])

        flags = []
        for cam, depth in (("d455", d455.depth_m(raw)), ("kinect", kinect.depth_m(k["depth"]))):
            plant_z = _plant_z(depth, z[cam])
            if plant_z and framing(depth, plant_z, up_edge_of(session, cam)).get("top_clipped"):
                clipped[cam] += 1
                flags.append(cam.upper())
        frames.append({"angle_deg": angle, "captured_at_utc": stamp,
                       "d455": {"rgb": f"rgb/{name}", "depth": f"depth/{name}"},
                       "kinect": {"rgb": f"rgb/{name}", "depth": f"depth/{name}",
                                  "color_full": f"color_full/{full_name}", "depth_full": f"depth_full/{name}"},
                       "on_old_kinect_grid": kinect_view_for_old_rig(angle),
                       "top_clipped": flags})
        print(f"  saved {index}/{len(ANGLES_DEG)}" + (f"  (top cut off: {', '.join(flags)})" if flags else ""))
    return frames, clipped


def write_manifests(plant_dir: Path, plant_id: str, metadata: dict, frames: list[dict],
                    d455, kinect, framing_result: dict, kinect_pause: bool) -> None:
    d455_manifest = {
        "plant_id": plant_id, "capture_date": metadata["capture_date"],
        "camera": dict(d455.info, emitter="on for D455 frames, off while the Kinect was read"),
        "orientation": metadata.get("d455_orientation"),
        "up_edge": framing_result.get("d455", {}).get("top_edge"),
        "frames": [dict(f["d455"], angle_deg=f["angle_deg"], captured_at_utc=f["captured_at_utc"])
                   for f in frames],
    }
    kinect_manifest = {
        "plant_id": plant_id, "capture_date": metadata["capture_date"],
        "camera": dict(kinect.info, warm_minutes_at_start=round(metadata.get("kinect_warm_minutes", 0), 1)),
        "mode": "paused while the D455 was read" if kinect_pause else "streaming throughout",
        "orientation": metadata.get("kinect_orientation"),
        "up_edge": framing_result.get("kinect", {}).get("top_edge"),
        "frames": [dict(f["kinect"], angle_deg=f["angle_deg"], captured_at_utc=f["captured_at_utc"],
                        on_old_kinect_grid=f["on_old_kinect_grid"]) for f in frames],
    }
    (plant_dir / "d455").mkdir(parents=True, exist_ok=True)
    (plant_dir / "kinect").mkdir(parents=True, exist_ok=True)
    (plant_dir / "d455" / "capture_manifest.json").write_text(json.dumps(d455_manifest, indent=2), "utf-8")
    (plant_dir / "kinect" / "capture_manifest.json").write_text(json.dumps(kinect_manifest, indent=2), "utf-8")
    (plant_dir / "d455" / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), "utf-8")
    (plant_dir / "metadata.json").write_text(json.dumps(
        dict(metadata, framing=framing_result, views=frames), indent=2, default=str), "utf-8")


def append_csv(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in CSV_FIELDS})


def one_plant(d455, kinect, session: dict, ask: Ask, args) -> str | None:
    taken = known_ids(args.output_dir, args.ground_truth_csv)
    suggested = default_plant_id(taken)
    while True:
        raw = ask(f"\nPlant ID [{suggested}]: ").strip() or suggested
        try:
            plant_id = validate_plant_id(raw)
        except ValueError as exc:
            print(f"  {exc}")
            continue
        if plant_id.upper() in taken:
            print(f"  {plant_id} is already used in this or another dataset; choose another.")
            continue
        break
    plant_dir = args.output_dir / plant_id
    meta: dict[str, Any] = {
        "plant_id": plant_id,
        "species_breed": ask_value(ask, "Species", "text", "Eucalyptus"),
        "capture_date": date.today().isoformat(),
        "session_id": session["session_id"],
        **{k: v for k, v in session.items() if k != "session_id"},
    }
    print("Before the sweep (the plant is still whole):")
    for key, label, kind in BEFORE_FIELDS:
        meta[key] = ask_value(ask, label, kind)
    meta["stem_diameter_mean_mm"] = mean_or_none(meta["stem_diameter_lower_mm"], meta["stem_diameter_upper_mm"])

    warm = kinect.warm_minutes()
    meta["kinect_warm_minutes"] = round(warm, 1)
    if warm < KINECT_WARMUP_MINUTES:
        print(f"  The Kinect has been on {warm:.0f} min; its depth settles after about "
              f"{KINECT_WARMUP_MINUTES:.0f}. Capturing now is allowed and the warm-up time is recorded.")

    print("\nFraming check (red edge in the preview = plant cut off):")
    while True:
        frame = check_framing(d455, kinect, session, plant_dir / "preview", args.preview, ask)
        cut = [c for c in ("d455", "kinect") if frame[c].get("top_clipped")]
        prompt = ("  Top cut off on " + " and ".join(c.upper() for c in cut) + ". "
                  if cut else "  Both framed. ")
        answer = ask(prompt + "Enter to start the sweep, r to recheck after moving a camera: ").strip().lower()
        if answer != "r":
            break
    meta["framing_top_clipped"] = cut

    try:
        frames, clipped = sweep(plant_dir, d455, kinect, ask, args.kinect_pause, session,
                                getattr(args, "full_colour_format", "png"))
    except KeyboardInterrupt:
        print(f"\n  {plant_id} abandoned. Files so far stay in {plant_dir}; no CSV row written.")
        return None
    meta["d455_top_clipped_views"] = clipped["d455"]
    meta["kinect_top_clipped_views"] = clipped["kinect"]
    meta["d455_serial"], meta["kinect_serial"] = d455.serial_number, kinect.serial_number

    print("\nAfter the sweep: cut the plant and weigh (blank = not measured).")
    for key, label, kind in AFTER_FIELDS:
        meta[key] = ask_value(ask, label, kind)
    while True:
        try:
            meta["net_plant_mass_g"] = net_mass(meta["total_fresh_with_pot_g"], meta["pot_plus_soil_mass_g"])
            break
        except ValueError as exc:
            print(f"  {exc}")
            meta["total_fresh_with_pot_g"] = ask_value(ask, "Scale: whole potted plant before cutting (g)")
            meta["pot_plus_soil_mass_g"] = ask_value(ask, "Scale: pot and soil left after cutting (g)")
    warning = mass_disagreement(meta["net_plant_mass_g"], meta["shoot_fresh_mass_g"])
    if warning:
        print("  CHECK: " + warning)
        meta["shoot_fresh_mass_g"] = ask_value(ask, "Scale: cut shoot alone, re-weighed (g)", "number",
                                               _str(meta["shoot_fresh_mass_g"]))
    meta["shoot_dry_mass_g"] = None  # filled only after oven drying, never copied from fresh mass
    meta["measured_by"] = ask_value(ask, "Measured by", "text", session.get("operator") or "")
    meta["notes"] = ask_value(ask, "Notes", "text")

    write_manifests(plant_dir, plant_id, meta, frames, d455, kinect, frame, args.kinect_pause)
    append_csv(args.ground_truth_csv, meta)
    blanks = [k for k in ("height_above_rim_mm", "stem_diameter_mean_mm", "shoot_fresh_mass_g",
                          "net_plant_mass_g") if meta.get(k) is None]
    print(f"\nDone: {plant_id}, {len(frames)} views from each camera"
          + (f". Top cut off in D455 {clipped['d455']}, Kinect {clipped['kinect']} views"
             if any(clipped.values()) else "")
          + (f". Still blank: {', '.join(blanks)}" if blanks else "") + ".")
    return plant_id


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

def save_session(session: dict, folder: Path = SESSIONS_DIR) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{session['session_id']}.json").write_text(json.dumps(session, indent=2, default=str), "utf-8")


def new_session_id() -> str:
    return datetime.now().strftime("S%Y%m%d-%H%M%S")


def run(args, ask: Ask = input) -> int:
    print("=" * 70)
    print("  CropCraft dual capture: RealSense D455 + Kinect V2, one turntable sweep")
    print(f"  Output: {args.output_dir}")
    print("=" * 70)
    if args.simulate:
        d455 = SimD455(args.sim_plant_height, portrait=True)
        make_kinect = lambda: SimKinect(args.sim_plant_height)  # noqa: E731
    else:
        d455 = D455(args.d455_serial, args.d455_fps)
        make_kinect = lambda: Kinect(args.kinect_serial, cpu_pipeline=args.kinect_pause)  # noqa: E731
    kinect = None
    try:
        print("\nOpening the D455 ...")
        d455.open()
        session = dict(ask_session(ask), session_id=new_session_id(),
                       started_at=datetime.now().isoformat(timespec="seconds"),
                       kinect_pause=args.kinect_pause, d455_fps=args.d455_fps)
        if args.skip_interference_test:
            kinect = make_kinect()
            kinect.open()
            session["interference"] = "skipped"
        else:
            kinect, session["interference"] = interference_test(d455, make_kinect)
        session["d455"] = d455.info
        session["kinect"] = kinect.info
        sessions = args.output_dir.parent / "sessions"
        save_session(session, sessions)
        if args.check_only:
            check_framing(d455, kinect, session, None, args.preview, ask)
            save_session(session, sessions)
            return 0
        while True:
            one_plant(d455, kinect, session, ask, args)
            save_session(session, sessions)
            answer = ask("\nNext plant? Enter = yes, g = change the camera set-up first, "
                         "q = finish: ").strip().lower()
            if answer == "q":
                break
            if answer == "g":
                previous = {k: v for k, v in session.items() if not k.endswith("_up_edge")}
                session = dict(previous, **ask_session(ask, previous), session_id=new_session_id(),
                               started_at=datetime.now().isoformat(timespec="seconds"))
                save_session(session, sessions)
    except (KeyboardInterrupt, EOFError):
        print("\nStopped.")
        return 130
    except KinectNotStreaming as exc:
        print(f"\nKinect failure: {exc}\nA plant in progress keeps the views saved so far; "
              "no CSV row was written for it. Both cameras are being closed.")
        return 2
    finally:
        for cam in (kinect, d455):
            try:
                if cam is not None:
                    cam.close()
            except Exception:  # noqa: BLE001
                pass
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    print("\nDone; both cameras are closed.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--d455-serial", default=None)
    parser.add_argument("--kinect-serial", default=None, help="default: the first Kinect found")
    parser.add_argument("--d455-fps", type=int, default=15, choices=(5, 15, 30),
                        help="15 leaves USB bandwidth for the Kinect (default 15)")
    parser.add_argument("--kinect-pause", action="store_true",
                        help="stop the Kinect while the D455 is read (use if the interference test says so)")
    parser.add_argument("--skip-interference-test", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="session set-up, checks, no plant")
    parser.add_argument("--no-preview", dest="preview", action="store_false",
                        help="do not open a preview window")
    parser.add_argument("--full-colour-format", choices=("png", "jpg"), default="png",
                        help="Kinect 1920x1080 colour: png is lossless (about 4 MB a view), jpg about 0.5 MB")
    parser.add_argument("--output-dir", type=Path, default=PLANTS_DIR)
    parser.add_argument("--ground-truth-csv", type=Path, default=GROUND_TRUTH_CSV)
    parser.add_argument("--simulate", action="store_true", help="no cameras; synthetic frames")
    parser.add_argument("--sim-plant-height", type=float, default=1.4, help="simulated plant height (m)")
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
