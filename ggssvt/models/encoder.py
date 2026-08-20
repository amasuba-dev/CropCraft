"""Per-view token encoder with geometry-driven token selection.

Two design points matter here.

**RGB-D patches, not RGB patches.** The stem consumes colour, metric depth and a
validity channel together. Depth is what makes a token's 3D anchor computable at
all, and the validity channel stops the network reading the Kinect's zero-return
holes as "surface at range zero" -- those holes are common on the thin stems and
dark pots in this capture set.

**Subject-driven token pruning.** The plant occupies about four percent of each
512x424 frame; the rest is greenhouse floor and background. Keeping every patch
would mean roughly ten thousand tokens per specimen, whose pairwise distance
matrix does not fit the 4 GB laptop GPU this work is developed on, and would
spend almost all of the attention budget on concrete. Because the views are
registered, the subject mask is already known, so the encoder keeps the ``K``
patches per view with the most subject coverage and discards the rest. This is
not a heuristic bolted on for speed -- it is only available *because* the model
is geometry-grounded, and it is what brings the cross-view attention in
:mod:`ggssvt.models.attention` within reach of a single consumer GPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import MODEL
from .embedding import GeometryGroundedEmbedding, patch_centroids


class PatchStem(nn.Module):
    """Convolutional patch embedding over RGB, depth and validity.

    Shape:
        - Input: ``(N, 5, H, W)``
        - Output: ``(N, C, H // patch, W // patch)``
    """

    def __init__(self, embed_dim: int = MODEL.embed_dim, patch_size: int = MODEL.patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.project = nn.Conv2d(5, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.GroupNorm(1, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.project(x))


class Dinov2Stem(nn.Module):
    """DINOv2 patch tokens as the appearance backbone.

    Loaded from ``torch.hub``, so it needs network access the first time. The
    depth and validity channels cannot pass through a frozen RGB backbone, so
    they are embedded separately and summed onto the DINOv2 tokens -- the
    appearance prior stays intact while the geometry still reaches the encoder.
    """

    def __init__(
        self,
        embed_dim: int = MODEL.embed_dim,
        name: str = MODEL.dinov2_name,
        freeze: bool = MODEL.freeze_backbone,
    ):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", name)
        self.patch_size = self.backbone.patch_size
        backbone_dim = self.backbone.embed_dim

        if freeze:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
            self.backbone.eval()

        self.frozen = freeze
        self.project = nn.Linear(backbone_dim, embed_dim)
        self.depth_stem = nn.Conv2d(
            2, embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )

    def train(self, mode: bool = True):  # noqa: D102 - keeps a frozen backbone in eval
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb, depth = x[:, :3], x[:, 3:]

        context = torch.no_grad() if self.frozen else torch.enable_grad()
        with context:
            features = self.backbone.forward_features(rgb)["x_norm_patchtokens"]

        tokens = self.project(features)

        grid_h = x.shape[-2] // self.patch_size
        grid_w = x.shape[-1] // self.patch_size
        tokens = tokens.transpose(1, 2).reshape(x.shape[0], -1, grid_h, grid_w)

        return tokens + self.depth_stem(depth)


def _select_top_patches(
    scores: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Indices of the ``k`` highest-scoring patches per view.

    Args:
        scores: ``(B, V, P)`` subject coverage per patch.
        k: patches to keep.

    Returns:
        ``(index, kept_score)``, both ``(B, V, k)``.
    """
    k = min(k, scores.shape[-1])
    kept_score, index = scores.topk(k, dim=-1)
    return index, kept_score


def _gather_tokens(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Gather ``(B, V, P, D)`` along the patch axis with a ``(B, V, k)`` index."""
    expanded = index.unsqueeze(-1).expand(*index.shape, source.shape[-1])
    return source.gather(2, expanded)


class ViewEncoder(nn.Module):
    """Encodes every view to a pruned set of geometry-grounded tokens.

    Args:
        embed_dim: token width.
        patch_size: patch side in pixels.
        depth: transformer layers applied within each view, before fusion.
        tokens_per_view: how many subject patches to keep per view.
        backbone: ``"cnn"`` for the trainable RGB-D stem, ``"dinov2"`` for the
            frozen foundation-model backbone.
    """

    def __init__(
        self,
        embed_dim: int = MODEL.embed_dim,
        patch_size: int = MODEL.patch_size,
        depth: int = MODEL.encoder_depth,
        num_heads: int = MODEL.num_heads,
        mlp_ratio: float = MODEL.mlp_ratio,
        dropout: float = MODEL.dropout,
        tokens_per_view: int = 64,
        backbone: str = MODEL.backbone,
    ):
        super().__init__()
        self.tokens_per_view = tokens_per_view
        self.backbone_kind = backbone

        if backbone == "cnn":
            self.stem = PatchStem(embed_dim=embed_dim, patch_size=patch_size)
            self.patch_size = patch_size
        elif backbone == "dinov2":
            self.stem = Dinov2Stem(embed_dim=embed_dim)
            self.patch_size = self.stem.patch_size
        else:
            raise ValueError(f"unknown backbone {backbone!r}; expected 'cnn' or 'dinov2'")

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.within_view = nn.TransformerEncoder(
            layer, num_layers=depth, enable_nested_tensor=False
        )
        self.grounding = GeometryGroundedEmbedding(embed_dim=embed_dim)

    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        points_world: torch.Tensor,
        subject: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            rgb: ``(B, V, 3, H, W)`` in [0, 1].
            depth: ``(B, V, 1, H, W)`` metres, 0 where invalid.
            points_world: ``(B, V, 3, H, W)`` world coordinates per pixel.
            subject: ``(B, V, 1, H, W)`` subject mask, float or bool.

        Returns:
            ``(tokens, anchors, anchor_valid)`` of shapes ``(B, V, K, C)``,
            ``(B, V, K, 3)`` and ``(B, V, K)``.
        """
        batch, views = rgb.shape[:2]
        subject = subject.to(rgb.dtype)
        valid = (depth > 0).to(rgb.dtype)

        stacked = torch.cat([rgb, depth, valid], dim=2)
        flat = stacked.reshape(batch * views, 5, *stacked.shape[-2:])

        features = self.stem(flat)
        grid_h, grid_w = features.shape[-2:]
        n_patches = grid_h * grid_w

        tokens = features.reshape(batch, views, -1, n_patches).permute(0, 1, 3, 2)

        # Anchor each patch on its *subject* pixels only. Averaging over every
        # valid pixel instead lets a patch that straddles the plant edge take
        # its 3D position partly from the greenhouse wall metres behind, which
        # drags the anchor off the specimen and corrupts the distance bias that
        # the whole architecture rests on.
        centroids, patch_valid = patch_centroids(
            points_world, (depth > 0) & (subject > 0), self.patch_size
        )
        coverage = (
            F.avg_pool2d(
                (subject * valid).reshape(batch * views, 1, *subject.shape[-2:]),
                self.patch_size,
            )
            .reshape(batch, views, n_patches)
        )

        index, kept_coverage = _select_top_patches(coverage, self.tokens_per_view)
        tokens = _gather_tokens(tokens, index)
        anchors = _gather_tokens(centroids, index)
        anchor_valid = patch_valid.gather(2, index) & (kept_coverage > 0)

        tokens = self.grounding(tokens, anchors, anchor_valid)

        # Refine within each view before the views are allowed to talk.
        shape = tokens.shape
        tokens = self.within_view(tokens.reshape(batch * views, shape[2], shape[3]))
        tokens = tokens.reshape(shape)

        return tokens, anchors, anchor_valid


__all__ = ["Dinov2Stem", "PatchStem", "ViewEncoder"]
