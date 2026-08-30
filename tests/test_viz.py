"""The configurable visualisation entry point.

Most of this is drawing, which tests cannot judge. What they can pin is that the
config refuses nonsense loudly rather than rendering something misleading, and
that the deliberate choices, greyscale support and the absence of jet, do not
quietly regress.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from ggssvt.eval.viz import COLORMAPS, LAYERS, SOURCES, VizConfig, ramp


def test_an_unknown_layer_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown layer"):
        VizConfig(layers=("occupancy", "hologram"))


def test_an_unknown_source_is_refused():
    with pytest.raises(ValueError, match="unknown source"):
        VizConfig(source="nerf")


def test_jet_is_refused_with_the_reason():
    """A perceptually non-uniform ramp invents boundaries in a depth cue.

    Refusing it in the constructor rather than silently substituting means a
    figure cannot be made with it by accident.
    """
    with pytest.raises(ValueError, match="luminance edges"):
        VizConfig(cmap="jet")


def test_every_advertised_layer_has_a_panel():
    """LAYERS drives the CLI help; a name with no implementation would crash."""
    from ggssvt.eval.viz import _PANELS

    assert set(LAYERS) == set(_PANELS)


def test_every_advertised_source_maps_to_a_cache_directory():
    for source in SOURCES:
        assert VizConfig(source=source).cache_dir.name.startswith("cache")


def test_viridis_is_perceptually_ordered():
    """Monotonic in luminance, which is what makes the depth cue readable."""
    samples = ramp(np.linspace(0.0, 1.0, 16), "viridis").astype(float)
    luminance = samples @ np.array([0.2126, 0.7152, 0.0722])
    assert all(a < b for a, b in pairwise(luminance))


def test_greys_runs_dark_to_light_and_its_reverse_is_the_mirror():
    """A thesis is often printed in greyscale; a colour-only figure fails there."""
    values = np.linspace(0.0, 1.0, 8)
    forward = ramp(values, "greys").astype(int)
    reverse = ramp(values, "greys_r").astype(int)

    assert forward[0, 0] > forward[-1, 0], "greys should darken with value"
    assert np.array_equal(forward[::-1], reverse)


def test_every_colormap_returns_three_uint8_channels():
    for cmap in COLORMAPS:
        out = ramp(np.linspace(0, 1, 5), cmap)
        assert out.shape == (5, 3)
        assert out.dtype == np.uint8


def test_values_outside_the_unit_range_are_clipped_not_wrapped():
    """Wrapping would put the far end of the ramp on an outlier."""
    low = ramp(np.array([-5.0]), "viridis")
    high = ramp(np.array([5.0]), "viridis")
    assert np.array_equal(low, ramp(np.array([0.0]), "viridis"))
    assert np.array_equal(high, ramp(np.array([1.0]), "viridis"))
