import json

import cv2
import numpy as np
import pytest

from dataset_d455.capture_d455 import (
    CAPTURE_ANGLES_DEG,
    GROUND_TRUTH_FIELDS,
    calculate_net_plant_mass,
    capture_plant,
    validate_plant_id,
)


def test_capture_angles_are_24_evenly_spaced_views_without_duplicate_360():
    assert len(CAPTURE_ANGLES_DEG) == 24
    assert CAPTURE_ANGLES_DEG == tuple(range(0, 360, 15))
    assert CAPTURE_ANGLES_DEG[-1] == 345


def test_workbook_fields_and_additional_pot_measurement_are_in_csv_schema():
    assert GROUND_TRUTH_FIELDS[:3] == (
        "plant_id",
        "species_breed",
        "capture_date",
    )
    assert "pot_weight_g" in GROUND_TRUTH_FIELDS
    assert "net_plant_mass_g" in GROUND_TRUTH_FIELDS
    assert "camera_serial_number" in GROUND_TRUTH_FIELDS
    assert len(GROUND_TRUTH_FIELDS) == len(set(GROUND_TRUTH_FIELDS))


@pytest.mark.parametrize(
    ("total_with_pot_g", "pot_weight_g", "expected"),
    [(1500, 500, 1000), (0, 500, 0), (1500, 0, 0), (0, 0, 0)],
)
def test_net_mass_is_only_derived_when_both_weights_are_available(
    total_with_pot_g, pot_weight_g, expected
):
    assert calculate_net_plant_mass(total_with_pot_g, pot_weight_g) == expected


def test_net_mass_rejects_pot_heavier_than_total():
    with pytest.raises(ValueError, match="at least the pot-only weight"):
        calculate_net_plant_mass(400, 500)


@pytest.mark.parametrize("plant_id", ["E044", "plant_02", "tree-7"])
def test_valid_plant_ids_are_returned_trimmed(plant_id):
    assert validate_plant_id(f" {plant_id} ") == plant_id


@pytest.mark.parametrize("plant_id", ["", "../outside", "plant/02", "a b"])
def test_plant_ids_cannot_escape_the_capture_directory(plant_id):
    with pytest.raises(ValueError):
        validate_plant_id(plant_id)


class FakeCamera:
    camera_info = {"serial_number": "test-d455"}

    def __init__(self, fail_on_capture=None):
        self.capture_count = 0
        self.close_count = 0
        self.fail_on_capture = fail_on_capture

    def capture(self):
        self.capture_count += 1
        if self.capture_count == self.fail_on_capture:
            raise RuntimeError("simulated camera failure")
        color = np.full((2, 2, 3), self.capture_count, dtype=np.uint8)
        depth = np.full((2, 2), 1000, dtype=np.uint16)
        return color, depth

    def close(self):
        self.close_count += 1


def test_capture_writes_24_rgb_depth_pairs_and_closes_camera(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("builtins.input", lambda _prompt: "")
    camera = FakeCamera()
    metadata = {
        "plant_id": "E099",
        "capture_date": "2026-10-08",
        "camera_serial_number": camera.camera_info["serial_number"],
    }

    plant_dir = capture_plant(metadata, tmp_path, camera)

    manifest = json.loads(
        (plant_dir / "capture_manifest.json").read_text(encoding="utf-8")
    )
    assert [frame["angle_deg"] for frame in manifest["frames"]] == list(
        CAPTURE_ANGLES_DEG
    )
    assert len(list((plant_dir / "rgb").glob("*.png"))) == 24
    assert len(list((plant_dir / "depth").glob("*.png"))) == 24
    assert cv2.imread(str(plant_dir / "depth" / "000.png"), cv2.IMREAD_UNCHANGED)[
        0, 0
    ] == 1000
    assert json.loads(
        (plant_dir / "metadata.json").read_text(encoding="utf-8")
    )["camera_serial_number"] == "test-d455"
    assert camera.capture_count == 24
    assert camera.close_count == 1


def test_failed_capture_closes_camera_and_keeps_partial_folder(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("builtins.input", lambda _prompt: "")
    camera = FakeCamera(fail_on_capture=2)
    metadata = {"plant_id": "E100", "capture_date": "2026-10-08"}

    with pytest.raises(RuntimeError, match="simulated camera failure"):
        capture_plant(metadata, tmp_path, camera)

    plant_dir = tmp_path / "E100"
    assert len(list((plant_dir / "rgb").glob("*.png"))) == 1
    assert not (plant_dir / "capture_manifest.json").exists()
    assert camera.close_count == 1
