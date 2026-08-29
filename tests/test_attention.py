"""Attention capture, and the guarantees that make it safe to leave in.

The weights are (B, heads, N, N) with N the token count across twelve views, so
capturing them unconditionally would be a real memory cost during training. The
properties worth pinning are therefore about when capture is *off*, not only
that it works when on.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ggssvt.eval.attention import (
    AttentionReading,
    capture_attention,
    render_per_view,
)
from ggssvt.models.attention import (
    CrossViewFusion,
    CrossViewGeometricAttention,
)


def _fusion(depth: int = 2):
    return CrossViewFusion(embed_dim=32, num_heads=4, depth=depth)


def _inputs(n_tokens: int = 24):
    return (
        torch.randn(1, n_tokens, 32),
        torch.randn(1, n_tokens, 3),
        torch.ones(1, n_tokens, dtype=torch.bool),
    )


def _slots(model):
    return [m for m in model.modules()
            if isinstance(m, CrossViewGeometricAttention)]


def test_capture_is_off_by_default():
    """Nothing is retained unless someone asks, or training pays for it."""
    fusion = _fusion()
    fusion(*_inputs())
    assert all(m._capture is None for m in _slots(fusion))


def test_capture_collects_one_matrix_per_block():
    fusion = _fusion(depth=3)
    with capture_attention(fusion) as captured:
        fusion(*_inputs())
    assert sorted(captured) == [0, 1, 2]
    for passes in captured.values():
        assert passes[0].shape == (1, 4, 24, 24)


def test_capture_is_cleared_even_when_the_body_raises():
    """A failed pass must not leave a model quietly accumulating weights."""
    fusion = _fusion()
    with pytest.raises(RuntimeError), capture_attention(fusion):
        raise RuntimeError("boom")
    assert all(m._capture is None for m in _slots(fusion))


def test_captured_weights_are_detached():
    """Holding graph references would pin activations for the whole capture."""
    fusion = _fusion()
    with capture_attention(fusion) as captured:
        fusion(*_inputs())
    assert not captured[0][0].requires_grad


def test_rows_of_the_attention_matrix_sum_to_one():
    """It is a softmax; if this stops holding the reduction is meaningless."""
    fusion = _fusion()
    with capture_attention(fusion) as captured:
        fusion(*_inputs())
    weights = captured[0][0]
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)),
                          atol=1e-5)


def test_neighbour_preference_is_about_one_with_no_structure():
    """The metric must not manufacture a result from flat attention.

    Above 1 is the claim that the fusion prefers views whose frusta overlap, so
    a uniform reading has to score 1 or the metric is biased toward finding H2
    supported.
    """
    flat = np.full((2, 12), 1.0 / 12)
    reading = AttentionReading(flat, np.zeros((0, 12, 0, 0)),
                               np.zeros((2, 4)), 12, (4, 4))
    assert reading.neighbour_preference() == pytest.approx(1.0)


def test_neighbour_preference_rises_when_neighbours_dominate():
    weight = np.full(12, 0.01)
    weight[[5, 7]] = 0.4          # the two neighbours of view 6
    weight[6] = 0.5
    reading = AttentionReading(weight[None, :], np.zeros((0, 12, 0, 0)),
                               np.zeros((1, 4)), 12, (4, 4))
    assert reading.neighbour_preference() > 5


def test_render_returns_an_image_and_refuses_an_empty_reading():
    reading = AttentionReading(np.full((3, 12), 1 / 12), np.zeros((0, 12, 0, 0)),
                               np.zeros((3, 4)), 12, (4, 4))
    assert render_per_view(reading, size=200).shape[1] == 200

    empty = AttentionReading(np.zeros((0, 12)), np.zeros((0, 12, 0, 0)),
                             np.zeros((0, 4)), 12, (0, 0))
    with pytest.raises(ValueError, match="no blocks"):
        render_per_view(empty)
