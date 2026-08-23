"""The implied-density plausibility check."""

from __future__ import annotations

import math

from ggssvt.eval.plausibility import classify, summarise


def test_a_sensible_volume_for_the_mass_is_plausible():
    # 1 kg in 2 litres -> 500 kg/m^3, squarely fresh plant tissue.
    result = classify("P1", mass_kg=1.0, volume_m3=0.002)

    assert result.plausible
    assert result.density_kg_m3 == 500.0


def test_a_canopy_envelope_is_caught():
    """Mango's carve: 25 L of hull for a 0.74 kg shoot."""
    result = classify("M001", mass_kg=0.74, volume_m3=0.02579)

    assert result.verdict == "envelope"
    assert result.density_kg_m3 < 50


def test_an_unreconstructed_specimen_is_caught():
    """E019: 0.09 L of volume for a 1.95 kg plant."""
    result = classify("E019", mass_kg=1.95, volume_m3=0.00009)

    assert result.verdict == "missing"
    assert result.density_kg_m3 > 10_000


def test_an_empty_volume_is_reported_rather_than_dividing_by_zero():
    result = classify("P0", mass_kg=1.0, volume_m3=0.0)

    assert result.verdict == "no volume"
    assert math.isinf(result.density_kg_m3)
    assert not result.plausible


def test_the_band_edges_are_inclusive():
    assert classify("lo", 1.0, 1.0 / 200.0).plausible
    assert classify("hi", 1.0, 1.0 / 1000.0).plausible


def test_summary_uses_the_median_so_one_outlier_cannot_set_it():
    results = [
        classify("a", 1.0, 0.002),        # 500
        classify("b", 1.0, 0.0025),       # 400
        classify("c", 1.0, 0.00009),      # 11111, the outlier
    ]

    summary = summarise(results)

    assert summary["n"] == 3
    assert summary["n_plausible"] == 2
    assert summary["median_density_kg_m3"] == 500.0
    assert summary["verdicts"]["missing"] == 1


def test_summary_of_nothing_is_not_an_error():
    assert summarise([])["n"] == 0
