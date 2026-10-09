"""Guided RGB-D collection for Eucalyptus plants with an Intel RealSense D455."""

from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timezone
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

DATASET_DIR = Path(__file__).resolve().parent
PLANTS_DIR = DATASET_DIR / "plants"
GROUND_TRUTH_CSV = DATASET_DIR / "ground_truth.csv"

COLOR_WIDTH = 1280
COLOR_HEIGHT = 720
DEPTH_WIDTH = 1280
DEPTH_HEIGHT = 720
FPS = 30
CAPTURE_ANGLES_DEG = tuple(range(0, 360, 15))

TEXT_FIELDS = {
    "harvest_date",
    "harvest_time",
    "scale_model",
    "watering_state",
    "measured_by",
    "notes",
}
INTEGER_FIELDS = {
    "leaf_count",
    "drying_hours",
}
GROUND_TRUTH_FIELDS = (
    "plant_id",
    "species_breed",
    "capture_date",
    "total_fresh_with_pot_g",
    "harvest_date",
    "harvest_time",
    "cut_height_above_rim_mm",
    "shoot_fresh_mass_g",
    "pot_plus_soil_mass_g",
    "minutes_cut_to_weighing",
    "height_above_cut_mm",
    "canopy_diameter_max_mm",
    "canopy_diameter_perp_mm",
    "stem_basal_diameter_mm",
    "leaf_fresh_mass_g",
    "stem_fresh_mass_g",
    "leaf_count",
    "shoot_dry_mass_g",
    "drying_hours",
    "drying_temp_c",
    "scale_model",
    "scale_resolution_g",
    "repeat_shoot_mass_g",
    "watering_state",
    "hours_since_watering",
    "measured_by",
    "notes",
    "pot_weight_g",
    "net_plant_mass_g",
    "camera_serial_number",
)


def validate_plant_id(plant_id: str) -> str:
    """Return a filesystem-safe plant ID or raise a useful error."""
    plant_id = plant_id.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", plant_id):
        raise ValueError("Use letters, numbers, underscores, or hyphens only.")
    return plant_id


def calculate_net_plant_mass(
    total_with_bag_and_soil_g: float,
    bag_and_soil_tare_g: float,
) -> float:
    """Estimate plant mass by subtracting the matching bag-and-soil tare."""
    total = float(total_with_bag_and_soil_g)
    tare = float(bag_and_soil_tare_g)

    if not math.isfinite(total) or not math.isfinite(tare):
        raise ValueError("Scale readings must be finite numbers.")
    if total < 0 or tare < 0:
        raise ValueError("Scale readings cannot be negative.")
    if total == 0 or tare == 0:
        return 0.0
    if total < tare:
        raise ValueError(
            "Total fresh weight must be at least the bag-and-soil tare."
        )
    return total - tare


def _prompt_text(label: str, default: str = "0") -> str:
    answer = input(f"{label} [{default}]: ").strip()
    return answer if answer else default


def _prompt_number(label: str, integer: bool = False) -> int | float:
    while True:
        raw = _prompt_text(label)
        try:
            value = int(raw) if integer else float(raw)
        except ValueError:
            print("  Enter a number, or press Enter to record 0.")
            continue

        if not math.isfinite(value) or value < 0:
            print("  Measurements cannot be negative; enter 0 if unavailable.")
            continue
        return value


def collect_plant_metadata() -> dict[str, Any]:
    """Prompt for metadata and calculate mass using the bag-and-soil tare."""
    print("\nPlant metadata (press Enter for 0 when a value is unavailable).")

    while True:
        try:
            plant_id = validate_plant_id(input("Plant ID: "))
            break
        except ValueError as exc:
            print(f"  Invalid plant ID: {exc}")

    species = _prompt_text("Species name", "Eucalyptus")
    capture_date = _prompt_text(
        "Capture date (YYYY-MM-DD)",
        date.today().isoformat(),
    )

    metadata: dict[str, Any] = {
        "plant_id": plant_id,
        "species_breed": species,
        "capture_date": capture_date,
    }

    field_labels = {
        "total_fresh_with_pot_g": (
            "Total fresh weight with plant, bag, and soil (g)"
        ),
        "harvest_date": "Harvest date (YYYY-MM-DD, or 0)",
        "harvest_time": "Harvest time (HH:MM, or 0)",
        "cut_height_above_rim_mm": "Cut height above pot rim (mm)",
        "shoot_fresh_mass_g": "Shoot fresh mass (g)",
        "pot_plus_soil_mass_g": "Bag/container plus soil tare (g)",
        "minutes_cut_to_weighing": "Minutes from cutting to weighing",
        "height_above_cut_mm": "Height above cut (mm)",
        "canopy_diameter_max_mm": "Maximum canopy diameter (mm)",
        "canopy_diameter_perp_mm": "Perpendicular canopy diameter (mm)",
        "stem_basal_diameter_mm": "Stem basal diameter (mm)",
        "leaf_fresh_mass_g": "Leaf fresh mass (g)",
        "stem_fresh_mass_g": "Stem fresh mass (g)",
        "leaf_count": "Leaf count",
        "shoot_dry_mass_g": "Shoot dry mass (g)",
        "drying_hours": "Drying duration (hours)",
        "drying_temp_c": "Drying temperature (C)",
        "scale_model": "Scale model",
        "scale_resolution_g": "Scale resolution (g)",
        "repeat_shoot_mass_g": "Repeat shoot mass (g)",
        "watering_state": "Watering state",
        "hours_since_watering": "Hours since watering",
        "measured_by": "Measured by",
        "notes": "Notes",
    }

    for field in GROUND_TRUTH_FIELDS:
        if field in metadata or field in {
            "net_plant_mass_g",
            "camera_serial_number",
            "pot_weight_g",
        }:
            continue

        if field in TEXT_FIELDS:
            metadata[field] = _prompt_text(field_labels[field])
        else:
            metadata[field] = _prompt_number(
                field_labels[field],
                integer=field in INTEGER_FIELDS,
            )

    # Retained for compatibility with the existing CSV schema. No pot-only
    # measurement is available for plants grown in bags.
    metadata["pot_weight_g"] = 0.0

    while True:
        try:
            metadata["net_plant_mass_g"] = calculate_net_plant_mass(
                float(metadata["total_fresh_with_pot_g"]),
                float(metadata["pot_plus_soil_mass_g"]),
            )
            break
        except ValueError as exc:
            print(f"  {exc} Please check both scale readings.")
            metadata["total_fresh_with_pot_g"] = _prompt_number(
                field_labels["total_fresh_with_pot_g"]
            )
            metadata["pot_plus_soil_mass_g"] = _prompt_number(
                field_labels["pot_plus_soil_mass_g"]
            )

    print(
        "  Plant-only mass: "
        f"{metadata['net_plant_mass_g']} g "
        "(0 means one or both scale readings were unavailable)."
    )
    return metadata


def _intrinsics_dict(video_profile: Any) -> dict[str, Any]:
    intrinsics = video_profile.get_intrinsics()
    return {
        "width": intrinsics.width,
        "height": intrinsics.height,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "ppx": intrinsics.ppx,
        "ppy": intrinsics.ppy,
        "model": str(intrinsics.model),
        "coeffs": list(intrinsics.coeffs),
    }


class D455Camera:
    """Own one RealSense pipeline for the duration of a single plant capture."""

    def __init__(self, requested_serial: str | None = None) -> None:
        self.requested_serial = requested_serial
        self.pipeline: Any = None
        self.aligner: Any = None
        self.depth_scale_m: float | None = None
        self.camera_info: dict[str, Any] = {}
        self.started = False
        self._rs: Any = None

    def open(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "pyrealsense2 is not installed. On Ubuntu run "
                "`python3 -m pip install pyrealsense2` in this environment."
            ) from exc

        self._rs = rs
        context = rs.context()
        devices = list(context.query_devices())
        d455s = [
            device
            for device in devices
            if "D455" in device.get_info(rs.camera_info.name)
        ]

        if not d455s:
            found = [
                device.get_info(rs.camera_info.name)
                for device in devices
            ]
            raise RuntimeError(
                "No Intel RealSense D455 was detected. "
                f"Detected devices: {found or 'none'}."
            )

        if self.requested_serial:
            d455s = [
                device
                for device in d455s
                if device.get_info(rs.camera_info.serial_number)
                == self.requested_serial
            ]
            if not d455s:
                raise RuntimeError(
                    f"No connected D455 has serial {self.requested_serial}."
                )

        if len(d455s) != 1:
            serials = [
                device.get_info(rs.camera_info.serial_number)
                for device in d455s
            ]
            raise RuntimeError(
                "More than one D455 is connected; select one with "
                f"--serial. Available serials: {serials}."
            )

        selected = d455s[0]
        serial = selected.get_info(rs.camera_info.serial_number)
        self.pipeline = rs.pipeline(context)
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(
            rs.stream.color,
            COLOR_WIDTH,
            COLOR_HEIGHT,
            rs.format.bgr8,
            FPS,
        )
        config.enable_stream(
            rs.stream.depth,
            DEPTH_WIDTH,
            DEPTH_HEIGHT,
            rs.format.z16,
            FPS,
        )

        profile = self.pipeline.start(config)
        self.started = True

        try:
            device = profile.get_device()

            # Configure color controls only on the color sensor. These controls
            # may help with global color balance, but cannot correct localized
            # color casts caused by lighting or an optical/sensor issue.
            try:
                color_sensor = device.first_color_sensor()
                if color_sensor.supports(rs.option.enable_auto_exposure):
                    color_sensor.set_option(
                        rs.option.enable_auto_exposure,
                        1.0,
                    )
                if color_sensor.supports(
                    rs.option.enable_auto_white_balance
                ):
                    color_sensor.set_option(
                        rs.option.enable_auto_white_balance,
                        1.0,
                    )
            except (AttributeError, RuntimeError) as exc:
                print(f"  Could not configure color auto controls: {exc}")

            depth_sensor = device.first_depth_sensor()
            self.depth_scale_m = float(depth_sensor.get_depth_scale())
            self.aligner = rs.align(rs.stream.color)

            self.camera_info = {
                "model": device.get_info(rs.camera_info.name),
                "serial_number": device.get_info(rs.camera_info.serial_number),
                "firmware_version": device.get_info(
                    rs.camera_info.firmware_version
                ),
                "depth_scale_m_per_unit": self.depth_scale_m,
                "depth_encoding": (
                    "aligned Z16; raw values multiplied by "
                    "depth_scale_m_per_unit give metres"
                ),
                "color_intrinsics": _intrinsics_dict(
                    profile.get_stream(rs.stream.color)
                    .as_video_stream_profile()
                ),
                "depth_intrinsics": _intrinsics_dict(
                    profile.get_stream(rs.stream.depth)
                    .as_video_stream_profile()
                ),
                "color_resolution": [COLOR_WIDTH, COLOR_HEIGHT],
                "depth_resolution": [DEPTH_WIDTH, DEPTH_HEIGHT],
                "fps": FPS,
            }

            # Let automatic exposure and white balance settle before capture.
            for _ in range(FPS):
                self.pipeline.wait_for_frames(5000)

        except BaseException:
            self.close()
            raise

        print(
            f"  Opened {self.camera_info['model']} "
            f"(serial {self.camera_info['serial_number']}); "
            f"depth scale {self.depth_scale_m} m/unit."
        )

    def capture(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.started:
            raise RuntimeError("The D455 stream is not open.")

        frames = self.pipeline.wait_for_frames(5000)
        aligned = self.aligner.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            raise RuntimeError(
                "The D455 did not return both color and depth frames."
            )

        color_bgr = np.asanyarray(color_frame.get_data())
        depth_z16 = np.asanyarray(depth_frame.get_data())

        if color_bgr.shape[:2] != depth_z16.shape[:2]:
            raise RuntimeError(
                "Aligned depth dimensions do not match the color dimensions: "
                f"{depth_z16.shape[:2]} vs {color_bgr.shape[:2]}."
            )

        return color_bgr, depth_z16

    def close(self) -> None:
        if self.started:
            try:
                self.pipeline.stop()
            finally:
                self.started = False


def _existing_plant_ids(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()

    with csv_path.open(newline="", encoding="utf-8") as source:
        return {
            row["plant_id"]
            for row in csv.DictReader(source)
            if row.get("plant_id")
        }


def _write_ground_truth(
    csv_path: Path,
    metadata: dict[str, Any],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    with csv_path.open("a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=GROUND_TRUTH_FIELDS,
            extrasaction="ignore",
        )
        if write_header:
            writer.writeheader()
        writer.writerow(metadata)


def capture_plant(
    metadata: dict[str, Any],
    output_dir: Path = PLANTS_DIR,
    camera: D455Camera | None = None,
) -> Path:
    plant_id = validate_plant_id(str(metadata["plant_id"]))
    plant_dir = output_dir / plant_id

    if plant_dir.exists():
        raise FileExistsError(
            f"{plant_dir} already exists; refusing to overwrite an existing capture."
        )

    if camera is None:
        camera = D455Camera()
    if not camera.started:
        camera.open()

    manifest_frames: list[dict[str, Any]] = []
    created = False

    try:
        plant_dir.mkdir(parents=True, exist_ok=False)
        created = True

        rgb_dir = plant_dir / "rgb"
        depth_dir = plant_dir / "depth"
        rgb_dir.mkdir()
        depth_dir.mkdir()

        for index, angle in enumerate(CAPTURE_ANGLES_DEG, start=1):
            print(
                f"\nView {index}/24 at {angle:03d} degrees. "
                "Align the turntable; then press Enter to capture "
                "(rotate it +15 degrees afterwards)."
            )
            input("  Press Enter when ready ... ")

            color_bgr, depth_z16 = camera.capture()
            rgb_path = rgb_dir / f"{angle:03d}.png"
            depth_path = depth_dir / f"{angle:03d}.png"

            if not cv2.imwrite(str(rgb_path), color_bgr):
                raise OSError(f"Could not write RGB image {rgb_path}.")
            if not cv2.imwrite(str(depth_path), depth_z16):
                raise OSError(f"Could not write depth image {depth_path}.")

            manifest_frames.append(
                {
                    "angle_deg": angle,
                    "rgb": f"rgb/{rgb_path.name}",
                    "depth": f"depth/{depth_path.name}",
                    "captured_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            print(f"  Saved {index}/24 views.")

        manifest = {
            "plant_id": plant_id,
            "capture_date": metadata["capture_date"],
            "camera": camera.camera_info,
            "frames": manifest_frames,
        }
        (plant_dir / "capture_manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        (plant_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

    except BaseException:
        if created:
            print(
                f"\nCapture stopped before completion. Partial files, if any, "
                f"remain under {plant_dir}; no ground-truth row was appended."
            )
        raise
    finally:
        camera.close()

    return plant_dir


def _confirm_another_plant() -> bool:
    answer = input("\nCapture another plant? [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


def _check_camera(serial: str | None) -> int:
    camera = D455Camera(requested_serial=serial)

    try:
        camera.open()
        color_bgr, depth_z16 = camera.capture()
        valid_depth = depth_z16[depth_z16 > 0]

        print(
            f"Color frame: {color_bgr.shape}, {color_bgr.dtype}; "
            f"depth frame: {depth_z16.shape}, {depth_z16.dtype}."
        )

        if valid_depth.size == 0:
            print(
                "Camera returned no nonzero depth pixels.",
                file=sys.stderr,
            )
            return 1

        scale = float(camera.depth_scale_m)
        valid_percent = 100.0 * valid_depth.size / depth_z16.size
        print(
            f"Nonzero depth: {valid_percent:.1f}% of pixels; "
            f"range {valid_depth.min() * scale:.3f}-"
            f"{valid_depth.max() * scale:.3f} m. "
            "No capture files were written."
        )
        return 0

    except Exception as exc:
        print(f"\nD455 camera check failed: {exc}", file=sys.stderr)
        return 1
    finally:
        camera.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Capture 24 manually rotated RGB-D views per plant "
            "on an Intel D455."
        )
    )
    parser.add_argument(
        "--serial",
        help=(
            "D455 serial number "
            "(required only when more than one D455 is connected)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PLANTS_DIR,
        help=f"Plant data directory (default: {PLANTS_DIR})",
    )
    parser.add_argument(
        "--ground-truth-csv",
        type=Path,
        default=GROUND_TRUTH_CSV,
        help=f"Metadata CSV path (default: {GROUND_TRUTH_CSV})",
    )
    parser.add_argument(
        "--check-camera",
        action="store_true",
        help=(
            "open the D455, acquire one RGB/depth pair, "
            "report stream health, and exit"
        ),
    )
    args = parser.parse_args(argv)

    if args.check_camera:
        return _check_camera(args.serial)

    print("=" * 68)
    print("  CropCraft Intel RealSense D455 Eucalyptus collection")
    print(f"  Output: {args.output_dir}")
    print("  24 views per plant, one every 15 degrees (0 through 345).")
    print("=" * 68)

    while True:
        try:
            metadata = collect_plant_metadata()
        except (KeyboardInterrupt, EOFError):
            print("\nMetadata entry cancelled.")
            return 130

        plant_id = metadata["plant_id"]

        if (args.output_dir / plant_id).exists():
            print(
                f"  Plant folder already exists for {plant_id}; "
                "choose another ID."
            )
            continue

        if plant_id in _existing_plant_ids(args.ground_truth_csv):
            print(
                f"  Plant ID {plant_id} already exists in the ground-truth CSV."
            )
            continue

        camera = D455Camera(requested_serial=args.serial)
        try:
            camera.open()
        except Exception as exc:
            print(f"\nCould not start D455 capture: {exc}", file=sys.stderr)
            return 1

        metadata["camera_serial_number"] = camera.camera_info["serial_number"]

        try:
            plant_dir = capture_plant(metadata, args.output_dir, camera)
        except (KeyboardInterrupt, EOFError):
            print("\nCapture cancelled; the camera stream has been closed.")
            return 130
        except Exception as exc:
            print(f"\nCapture failed: {exc}", file=sys.stderr)
            return 1

        try:
            _write_ground_truth(args.ground_truth_csv, metadata)
        except OSError as exc:
            print(
                f"\nImages saved to {plant_dir}, but ground truth could not be "
                f"written to {args.ground_truth_csv}: {exc}",
                file=sys.stderr,
            )
            return 1

        print(
            f"\nCompleted {plant_id}: 24 RGB/depth pairs saved under "
            f"{plant_dir}."
        )
        if not _confirm_another_plant():
            break

    print("\nDone; all D455 streams are closed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())