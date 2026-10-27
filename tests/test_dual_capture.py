import csv
import json
from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest

from dataset_dual import capture_dual as cd


def scene(plant_h, cam_h, portrait=False, height=720, width=1280, focal=646.0, z=1.1):
    return cd._scene(height, width, focal, z, plant_h, cam_h, portrait, np.random.default_rng(0))


# -- values and IDs ---------------------------------------------------------

@pytest.mark.parametrize("raw", ["", "  ", "na", "N/A", "-", "unknown"])
def test_blank_and_na_mean_unknown_never_zero(raw):
    assert cd.parse_optional(raw) is None


def test_numbers_parse_and_negative_is_refused():
    assert cd.parse_optional("12.5") == 12.5
    assert cd.parse_optional("7", "integer") == 7
    assert cd.parse_optional("note", "text") == "note"
    with pytest.raises(ValueError):
        cd.parse_optional("-3")


def test_ids_are_checked_across_every_dataset_case_insensitively(tmp_path):
    own, other = tmp_path / "dual" / "plants", tmp_path / "d455" / "plants"
    (own / "EUC-20261027-01").mkdir(parents=True)
    (other / "e01").mkdir(parents=True)
    kinect_csv = tmp_path / "kinect.csv"
    kinect_csv.write_text("plant_id,x\nE001,1\n", encoding="utf-8")
    taken = cd.known_ids(own, tmp_path / "none.csv",
                         others=((other, tmp_path / "x.csv"), (tmp_path / "k", kinect_csv)))
    assert {"EUC-20261027-01", "E01", "E001"} <= taken
    assert cd.default_plant_id(taken, date(2026, 10, 27)) == "EUC-20261027-02"


def test_plant_id_rejects_unsafe_names():
    with pytest.raises(ValueError):
        cd.validate_plant_id("../E01")
    assert cd.validate_plant_id(" EUC-1 ") == "EUC-1"


# -- masses -----------------------------------------------------------------

def test_net_mass_is_blank_when_a_reading_is_missing():
    assert cd.net_mass(None, 700) is None
    assert cd.net_mass(1200, None) is None
    assert cd.net_mass(1200, 700) == 500


def test_net_mass_refuses_tare_heavier_than_total():
    with pytest.raises(ValueError):
        cd.net_mass(700, 1200)


def test_the_two_routes_to_mass_are_compared():
    assert cd.mass_disagreement(500, 498) is None
    assert "apart" in cd.mass_disagreement(500, 300)
    assert cd.mass_disagreement(None, 300) is None


def test_stem_mean_uses_whatever_was_measured():
    assert cd.mean_or_none(10.0, 8.0) == 9.0
    assert cd.mean_or_none(None, 8.0) == 8.0
    assert cd.mean_or_none(None, None) is None


# -- framing ----------------------------------------------------------------

def test_low_camera_finds_the_floor_and_flags_a_tall_plant():
    tall = cd.framing(scene(1.4, 0.1), 1.1)
    assert tall["floor_edge"] == "bottom" and tall["top_edge"] == "top"
    assert tall["top_clipped"] is True


def test_low_camera_short_plant_is_framed():
    short = cd.framing(scene(0.6, 0.1), 1.1)
    assert short["floor_edge"] == "bottom"
    assert short["top_clipped"] is False


def test_portrait_with_known_up_edge():
    depth = scene(1.4, 0.75, portrait=True)
    framed = cd.framing(depth, 1.1, up_edge="right")
    assert framed["top_edge"] == "right" and framed["top_clipped"] is False
    too_tall = cd.framing(scene(2.2, 0.75, portrait=True), 1.1, up_edge="right")
    assert too_tall["top_clipped"] is True


def test_landscape_up_edge_flags_a_cut_top_even_without_a_visible_floor():
    depth = scene(1.6, 0.75)
    depth[depth < 1.05] = 2.0     # no floor or turntable in the near band
    assert cd.framing(depth, 1.1, up_edge="top")["top_clipped"] is True


def test_unknown_up_edge_reports_any_contact():
    depth = scene(1.6, 0.75)
    depth[depth < 1.05] = 2.0
    result = cd.framing(depth, 1.1)
    assert result["top_edge"] is None and result["top_clipped"] is True
    assert "top" in result["touching"]


def test_empty_depth_is_unclear_not_a_crash():
    assert cd.framing(np.zeros((50, 60), np.float32), 1.1)["top_clipped"] is None


def test_plant_depth_estimate_finds_the_plant_not_the_backdrop():
    z = cd.estimate_plant_depth(scene(1.4, 0.75))
    assert z == pytest.approx(1.1, abs=0.15)


# -- interference -----------------------------------------------------------

def test_depth_stats_and_interference_verdict():
    rng = np.random.default_rng(0)
    calm = 1.1 + rng.normal(0, 0.001, (10, 60, 80)).astype(np.float32)
    noisy = 1.1 + rng.normal(0, 0.004, (10, 60, 80)).astype(np.float32)
    holes = calm.copy()
    holes[:, 20:40, 30:50] = 0
    a, b, c = cd.depth_stats(calm), cd.depth_stats(noisy), cd.depth_stats(holes)
    assert a["valid_pct"] == 100 and c["valid_pct"] < 90
    assert cd.interference_verdict(a, a, "x")[0] is False
    assert cd.interference_verdict(a, b, "x")[0] is True
    assert cd.interference_verdict(a, c, "x")[0] is True


def test_old_kinect_grid_is_every_second_view():
    assert [a for a in cd.ANGLES_DEG if cd.kinect_view_for_old_rig(a)] == list(range(0, 360, 30))


# -- a whole simulated session ----------------------------------------------

def scripted(answers):
    queue = list(answers)

    def ask(prompt):
        assert queue, f"ran out of answers at: {prompt}"
        return queue.pop(0)
    ask.left = queue
    return ask


SESSION = ["Aaron", "Farm", "greenhouse overcast", "black cloth",
           "1100", "750", "", "1100", "600", ""]
BEFORE = ["", "1400", "420", "380", "10.5", "8.5", "1200", "watered 07:00"]
AFTER = ["10:42", "20", "498", "700", "4", "41", "", ""]


def test_simulated_session_writes_both_cameras_and_blank_unknowns(tmp_path):
    args = SimpleNamespace(simulate=True, sim_plant_height=1.4, d455_serial=None, kinect_serial=None,
                           d455_fps=15, kinect_pause=False, skip_interference_test=False,
                           check_only=False, preview=False, output_dir=tmp_path / "plants",
                           ground_truth_csv=tmp_path / "ground_truth.csv")
    answers = SESSION + ["", *BEFORE, "right", ""] + [""] * 24 + AFTER + ["q"]
    ask = scripted(answers)
    assert cd.run(args, ask) == 0
    assert ask.left == []

    plant = next((tmp_path / "plants").iterdir())
    assert plant.name.startswith("EUC-")
    for sub, count in (("d455/rgb", 24), ("d455/depth", 24), ("kinect/rgb", 24), ("kinect/depth", 24),
                       ("kinect/color_full", 24), ("kinect/depth_full", 24)):
        assert len(list((plant / sub).glob("*.png"))) == count, sub

    d455 = json.loads((plant / "d455" / "capture_manifest.json").read_text())
    kinect = json.loads((plant / "kinect" / "capture_manifest.json").read_text())
    assert d455["up_edge"] == "right" and d455["orientation"] == "portrait"
    assert kinect["up_edge"] == "top"
    assert sum(f["on_old_kinect_grid"] for f in kinect["frames"]) == 12

    row = next(csv.DictReader((tmp_path / "ground_truth.csv").open()))
    assert row["net_plant_mass_g"] == "500.0"
    assert row["stem_diameter_mean_mm"] == "9.5"
    assert row["shoot_dry_mass_g"] == ""          # never copied from fresh mass
    assert row["measured_by"] == "Aaron"
    assert row["d455_top_clipped_views"] == "0"    # portrait, raised camera: whole plant in frame
    assert int(row["kinect_top_clipped_views"]) == 24   # landscape Kinect at 0.6 m cuts a 1.4 m plant
    assert row["d455_orientation"] == "portrait" and row["d455_distance_mm"] == "1100.0"
    sessions = list((tmp_path / "sessions").glob("*.json"))
    assert sessions and "interference" in json.loads(sessions[0].read_text())


def test_a_reused_id_is_refused(tmp_path):
    (tmp_path / "plants" / "E01").mkdir(parents=True)
    args = SimpleNamespace(simulate=True, sim_plant_height=1.0, d455_serial=None, kinect_serial=None,
                           d455_fps=15, kinect_pause=True, skip_interference_test=True, check_only=False,
                           preview=False, output_dir=tmp_path / "plants",
                           ground_truth_csv=tmp_path / "ground_truth.csv")
    answers = SESSION + ["e01", "NEW-1", *BEFORE, "right", ""] + [""] * 24 + AFTER + ["q"]
    ask = scripted(answers)
    assert cd.run(args, ask) == 0
    assert (tmp_path / "plants" / "NEW-1" / "metadata.json").exists()


def test_abandoning_a_plant_writes_no_csv_row(tmp_path):
    args = SimpleNamespace(simulate=True, sim_plant_height=1.0, d455_serial=None, kinect_serial=None,
                           d455_fps=15, kinect_pause=False, skip_interference_test=True, check_only=False,
                           preview=False, output_dir=tmp_path / "plants",
                           ground_truth_csv=tmp_path / "ground_truth.csv")
    answers = SESSION + ["", *BEFORE, "right", "", "", "q", "q"]
    ask = scripted(answers)
    assert cd.run(args, ask) == 0
    assert not (tmp_path / "ground_truth.csv").exists()
