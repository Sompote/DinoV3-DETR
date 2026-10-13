# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the DINOv3 encoder. All encoders are randomly initialised, so no gated Hub weights are needed."""

import pytest
import torch

from rfdetr.config import DinoV3DETRNanoConfig, DinoV3DETRSegNanoConfig
from rfdetr.models.backbone import _BACKBONE_REGISTRY
from rfdetr.models.backbone.dinov3 import DINOV3_ENCODER_NAMES, DinoV3, DinoV3Backbone

_H, _W = 8, 12  # patch grid of a 128 x 192 input


def _encoder(**kwargs: object) -> DinoV3:
    """Build a randomly initialised DINOv3-S encoder in eval mode.

    Examples:
        >>> _encoder(use_windowed_attn=False).patch_size
        16
    """
    return DinoV3("small", [3, 6, 9, 12], load_pretrained_weights=False, **kwargs).eval()  # type: ignore[arg-type]


@pytest.fixture
def image() -> torch.Tensor:
    """A batch of two 128 x 192 images."""
    return torch.randn(2, 3, 16 * _H, 16 * _W)


class TestDinoV3Encoder:
    def test_global_path_matches_transformers(self, image: torch.Tensor) -> None:
        encoder = _encoder(use_windowed_attn=False)
        with torch.no_grad():
            feats = encoder(image)
            hidden_states = encoder.model(image, output_hidden_states=True).hidden_states
        offset = len(hidden_states) - 12
        refs = [
            encoder.model.norm(hidden_states[i - 1 + offset])[:, 5:].transpose(1, 2).reshape(2, -1, _H, _W)
            for i in (3, 6, 9, 12)
        ]
        assert all(torch.allclose(f, r, atol=1e-5) for f, r in zip(feats, refs))

    def test_feature_shapes(self, image: torch.Tensor) -> None:
        with torch.no_grad():
            feats = _encoder(num_windows=2)(image)
        assert [tuple(f.shape) for f in feats] == [(2, 384, _H, _W)] * 4

    def test_window_partition_merge_is_inverse(self) -> None:
        encoder = _encoder(num_windows=2)
        tokens = torch.randn(2, 5 + _H * _W, 384)
        merged = encoder._window_merge(encoder._window_partition(tokens, _H, _W), 2, _H, _W)
        assert torch.equal(merged, tokens)

    def test_windowed_layer_matches_per_window_reference(self, image: torch.Tensor) -> None:
        encoder = _encoder(num_windows=2)
        layer = encoder.blocks[0]
        with torch.no_grad():
            cos, sin = encoder.model.rope_embeddings(image)
            hidden = encoder.model.embeddings(image)
            windowed_rope = (encoder._window_rope(cos, 2, _H, _W), encoder._window_rope(sin, 2, _H, _W))
            out = encoder._window_merge(
                layer(encoder._window_partition(hidden, _H, _W), position_embeddings=windowed_rope), 2, _H, _W
            )

            patches = hidden[:, 5:].reshape(2, _H, _W, 384)
            cos_grid, sin_grid = cos.reshape(_H, _W, -1), sin.reshape(_H, _W, -1)
            ref_patches = torch.zeros_like(patches)
            ref_prefix = torch.zeros_like(hidden[:, :5])
            for a in range(2):
                for b in range(2):
                    rows, cols = slice(a * _H // 2, (a + 1) * _H // 2), slice(b * _W // 2, (b + 1) * _W // 2)
                    tokens = torch.cat([hidden[:, :5], patches[:, rows, cols].reshape(2, -1, 384)], dim=1)
                    rope = (
                        cos_grid[rows, cols].reshape(-1, cos.shape[-1]),
                        sin_grid[rows, cols].reshape(-1, sin.shape[-1]),
                    )
                    window_out = layer(tokens, position_embeddings=rope)
                    ref_patches[:, rows, cols] = window_out[:, 5:].reshape(2, _H // 2, _W // 2, 384)
                    ref_prefix += window_out[:, :5] / 4
        ref = torch.cat([ref_prefix, ref_patches.reshape(2, -1, 384)], dim=1)
        assert torch.allclose(out, ref, atol=1e-5)

    def test_gradient_checkpointing_matches_plain(self, image: torch.Tensor) -> None:
        plain = _encoder(num_windows=2).train()
        checkpointed = _encoder(num_windows=2, gradient_checkpointing=True).train()
        checkpointed.load_state_dict(plain.state_dict())
        assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(plain(image), checkpointed(image)))

    def test_every_layer_has_schedulable_drop_path(self) -> None:
        assert all(hasattr(layer.drop_path, "drop_prob") for layer in _encoder().blocks)

    def test_rejects_indivisible_input(self) -> None:
        with pytest.raises(AssertionError, match="divisible by 32"):
            _encoder(num_windows=2)(torch.randn(1, 3, 48, 64))


class TestDinoV3Registration:
    @pytest.mark.parametrize("encoder", DINOV3_ENCODER_NAMES)
    def test_encoder_registered(self, encoder: str) -> None:
        assert _BACKBONE_REGISTRY[encoder] is DinoV3Backbone

    def test_config_selects_dinov3(self) -> None:
        config = DinoV3DETRNanoConfig()
        assert (config.encoder, config.patch_size, config.pretrain_weights) == ("dinov3_windowed_small", 16, None)

    def test_segmentation_config_selects_dinov3(self) -> None:
        config = DinoV3DETRSegNanoConfig()
        assert (config.encoder, config.patch_size, config.resolution, config.segmentation_head) == (
            "dinov3_windowed_small",
            16,
            416,
            True,
        )

    def test_rejects_non_16_patch_size(self) -> None:
        with pytest.raises(ValueError, match="patch_size=16"):
            DinoV3Backbone(
                "dinov3_windowed_small",
                out_feature_indexes=[3, 6, 9, 12],
                projector_scale=["P4"],
                load_dinov2_weights=False,
                patch_size=14,
            )
