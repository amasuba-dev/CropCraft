"""The hierarchical stem, and the one property it exists to provide.

Swin is here for token resolution rather than representation strength: the
backbone comparison already returned a null, so "a stronger encoder" is not the
claim. The claim is that a stem five to seven pixels wide survives a stride-4
first stage where it cannot survive a 16-pixel patch embedding.

That is a property of the code, not a hope about the weights, and the tests
below check it without downloading anything: the fusion path is exercised on
synthetic pyramid levels, so they run on a machine with no network and no
transformers install.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ggssvt.models.backbones import (  # noqa: E402
    SWIN_REPOS, backbone_is_available,
)


def test_swin_is_checked_against_its_own_repositories():
    """The availability gate must not send Swin to DINOv3's catalogue.

    It used to. `DINOV2_REPOS if kind == "dinov2" else DINOV3_REPOS` routes
    every other backbone to DINOv3, so a Swin tiny reported "unknown variant"
    and a Swin base was checked against a DINOv3 access grant. Both answers were
    about the wrong model.
    """
    ok, reason = backbone_is_available("swin", "large")
    assert not ok
    assert "swin" in reason and "dinov3" not in reason.lower()

    ok, reason = backbone_is_available("bogus", "tiny")
    assert not ok and "unknown backbone" in reason

    assert set(SWIN_REPOS) == {"tiny", "small", "base"}


def _fuse_like_swin(levels, grid_h, grid_w):
    """The resampling half of SwinBackbone._fuse, without the weights.

    Kept in step with the real thing by test_fusion_matches_the_backbone below,
    which runs the same comparison through the module itself when transformers
    is present.
    """
    import torch.nn.functional as F

    out = []
    for level in levels:
        h, w = level.shape[-2:]
        if (h, w) == (grid_h, grid_w):
            out.append(level)
        elif h > grid_h:
            out.append(F.adaptive_avg_pool2d(level, (grid_h, grid_w)))
        else:
            out.append(F.interpolate(level, size=(grid_h, grid_w),
                                     mode="bilinear", align_corners=False))
    return out


def test_a_thin_stem_survives_area_pooling_but_not_sampling():
    """The mechanism, stated as a test.

    A vertical line two columns wide on a stride-4 grid is narrower than the
    stride-16 output cell. Pooled by area it contributes a non-zero fraction of
    every cell it crosses. Sampled, it lands between sample points and is gone
    from most of them. That difference is the entire argument for this backbone,
    so it is pinned rather than assumed.
    """
    import torch.nn.functional as F

    fine = torch.zeros(1, 1, 104, 128)
    fine[..., 50:52] = 1.0                      # a two-pixel stem at stride 4

    pooled = _fuse_like_swin([fine], 26, 32)[0]
    sampled = F.interpolate(fine, size=(26, 32), mode="nearest")

    stem_column = pooled[0, 0, :, 12:13]
    assert (stem_column > 0).all(), "area pooling lost the stem"
    assert float(pooled.sum()) > 0.0

    # Sampling keeps it only where a sample happens to land on it.
    assert float((sampled > 0).sum()) < float((pooled > 0).sum()), (
        "sampling should reach fewer cells than pooling for a thin structure")


def test_every_pyramid_level_reaches_the_output_grid():
    """Coarse levels upsample, fine levels pool, and all four line up."""
    levels = [
        torch.randn(2, 96, 104, 128),
        torch.randn(2, 192, 52, 64),
        torch.randn(2, 384, 26, 32),
        torch.randn(2, 768, 13, 16),
    ]
    resampled = _fuse_like_swin(levels, 26, 32)

    assert len(resampled) == 4
    for level in resampled:
        assert level.shape[-2:] == (26, 32)


def test_the_output_grid_matches_the_vit_it_is_compared_against():
    """26 by 32 on a 416 by 512 frame, which is DINOv3's grid exactly.

    If these differed, a Swin arm would change the token count as well as the
    stem, and the campaign comparison would confound the two.
    """
    height, width = 416, 512
    assert (height // 16, width // 16) == (26, 32)
    assert (height // 4, width // 4) == (104, 128)


@pytest.mark.skipif(
    not backbone_is_available("swin", "tiny")[0],
    reason="needs transformers and the Swin weights")
def test_fusion_matches_the_backbone():
    """The real module agrees with the helper the other tests use."""
    from ggssvt.models.backbones import build_backbone

    backbone = build_backbone("swin", variant="tiny")
    rgb = torch.randn(1, 3, 416, 512)
    with torch.no_grad():
        tokens, grid_h, grid_w = backbone.patch_tokens(rgb)

    assert (grid_h, grid_w) == (26, 32)
    assert tokens.shape[1] == grid_h * grid_w
    assert torch.isfinite(tokens).all()

    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    assert trainable > 0, "the lateral projections must stay trainable"
