# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""DINOv3 ViT encoder for RF-DETR.

DINOv3 differs from DINOv2 in three ways that matter here: its patch size is 16, it has four register tokens next to
the CLS token, and it encodes position with 2D RoPE on the patch tokens instead of a learned absolute table. RoPE is
computed from the input grid on every call, so any resolution divisible by ``16 * num_windows`` works without
interpolating weights, and export needs no position-table freeze.

Windowed attention follows RF-DETR's DINOv2 design. Layers not listed in ``out_feature_indexes`` attend inside
``num_windows x num_windows`` windows, the others attend globally. Each window keeps its patches' global RoPE
coordinates, so a windowed layer computes exactly the attention of the global layer restricted to the window. The
prefix (CLS + registers) is copied into every window and averaged back afterwards.
"""

from __future__ import annotations

from typing import Any, cast, get_args

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from rfdetr.config import DinoV3EncoderName
from rfdetr.models.backbone.backbone import Backbone
from rfdetr.models.backbone.dinov2 import compute_window_block_indexes
from rfdetr.utilities.logger import get_logger

logger = get_logger()

__all__ = ["DINOV3_ENCODER_NAMES", "DinoV3", "DinoV3Backbone"]

#: Architecture of each LVD-1689M DINOv3 ViT, as given by ``convert_dinov3_vit_to_hf.py`` in transformers. Only the
#: fields that differ from the ``DINOv3ViTConfig`` defaults are listed.
_SIZE_TO_CONFIG: dict[str, dict[str, Any]] = {
    "small": dict(hidden_size=384, intermediate_size=1536, num_hidden_layers=12, num_attention_heads=6),
    "splus": dict(
        hidden_size=384,
        intermediate_size=1536,
        num_hidden_layers=12,
        num_attention_heads=6,
        use_gated_mlp=True,
        hidden_act="silu",
    ),
    "base": dict(hidden_size=768, intermediate_size=3072, num_hidden_layers=12, num_attention_heads=12),
    "large": dict(hidden_size=1024, intermediate_size=4096, num_hidden_layers=24, num_attention_heads=16),
    "hplus": dict(
        hidden_size=1280,
        intermediate_size=5120,
        num_hidden_layers=32,
        num_attention_heads=20,
        use_gated_mlp=True,
        hidden_act="silu",
    ),
}

_SIZE_TO_HUB_ID: dict[str, str] = {
    "small": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "splus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "hplus": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
}

#: Every ``ModelConfig.encoder`` value served by :class:`DinoV3Backbone`.
DINOV3_ENCODER_NAMES: tuple[str, ...] = get_args(DinoV3EncoderName)


def build_dinov3_config(size: str, drop_path_rate: float = 0.0) -> DINOv3ViTConfig:
    """Build the transformers config of a DINOv3 ViT size.

    RoPE coordinate augmentation (``pos_embed_rescale``, on by default in DINOv3 pretraining) is switched off, so the
    detector sees the same patch coordinates in training and evaluation.

    Args:
        size: One of ``small``, ``splus``, ``base``, ``large``, ``hplus``.
        drop_path_rate: Stochastic-depth rate written into every layer.

    Returns:
        The DINOv3 ViT config.

    Examples:
        >>> build_dinov3_config("base").hidden_size
        768
    """
    if size not in _SIZE_TO_CONFIG:
        raise ValueError(f"Unknown DINOv3 size {size!r}; expected one of {sorted(_SIZE_TO_CONFIG)}.")
    return DINOv3ViTConfig(
        patch_size=16,
        num_register_tokens=4,
        drop_path_rate=drop_path_rate,
        pos_embed_shift=None,
        pos_embed_jitter=None,
        pos_embed_rescale=None,
        **_SIZE_TO_CONFIG[size],
    )


class _DropPath(nn.Module):
    """Per-sample stochastic depth with a mutable ``drop_prob``, which ``LWDETR.update_drop_path`` schedules.

    transformers renamed its own DINOv3 drop-path class between 5.x releases, so it is not imported.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class DinoV3(nn.Module):
    """DINOv3 ViT returning per-stage patch feature maps, with optional windowed attention.

    Args:
        size: DINOv3 size (``small``, ``splus``, ``base``, ``large``, ``hplus``).
        out_feature_indexes: Stage numbers to return, where stage ``i`` is the output of layer ``i`` (1-indexed).
        use_windowed_attn: Run layers outside ``out_feature_indexes`` (or in ``window_block_indexes``) windowed.
        num_windows: Windows per image side.
        window_block_indexes: Explicit 0-indexed windowed layers; ``None`` derives them as in DINOv2.
        load_pretrained_weights: Load LVD-1689M weights from the Hugging Face Hub (gated: accept the licence on the
            model page and log in with ``hf auth login`` first).
        gradient_checkpointing: Checkpoint each layer's activations in training.
        drop_path_rate: Stochastic-depth rate.
    """

    def __init__(
        self,
        size: str = "small",
        out_feature_indexes: list[int] | None = None,
        use_windowed_attn: bool = True,
        num_windows: int = 2,
        window_block_indexes: list[int] | None = None,
        load_pretrained_weights: bool = True,
        gradient_checkpointing: bool = False,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if not out_feature_indexes:
            out_feature_indexes = [3, 6, 9, 12]
        config = build_dinov3_config(size, drop_path_rate)
        num_layers = config.num_hidden_layers
        if any(not (1 <= i <= num_layers) for i in out_feature_indexes):
            raise ValueError(f"out_feature_indexes must lie in [1, {num_layers}] for DINOv3-{size}.")

        if load_pretrained_weights:
            self.model = DINOv3ViTModel.from_pretrained(_SIZE_TO_HUB_ID[size], config=config)
        else:
            self.model = DINOv3ViTModel(config)

        # transformers uses nn.Identity when drop_path_rate == 0; a module with ``drop_prob`` lets the drop-path
        # schedule (LWDETR.update_drop_path) reach every layer.
        for layer in self.blocks:
            layer.drop_path = _DropPath(drop_path_rate)

        self.patch_size = int(config.patch_size)
        self.num_prefix_tokens = 1 + config.num_register_tokens
        self.num_windows = num_windows if use_windowed_attn else 1
        self.out_feature_indexes = list(out_feature_indexes)
        self.window_block_indexes = (
            set(compute_window_block_indexes(self.out_feature_indexes, window_block_indexes))
            if use_windowed_attn and num_windows > 1
            else set()
        )
        self.gradient_checkpointing = gradient_checkpointing
        self._out_feature_channels = [config.hidden_size] * len(self.out_feature_indexes)
        self._export = False

    @property
    def blocks(self) -> nn.ModuleList:
        """Transformer layers, the layout ``LWDETR.update_drop_path`` looks for.

        transformers moved them from ``DINOv3ViTModel.layer`` to ``DINOv3ViTModel.model.layer`` within 5.x.
        """
        container = self.model.model if hasattr(self.model, "model") else self.model
        return cast(nn.ModuleList, container.layer)

    def export(self) -> None:
        """Mark the encoder as exported; RoPE is recomputed from the input shape, so nothing is baked."""
        self._export = True

    def _window_partition(self, x: Tensor, h: int, w: int) -> Tensor:
        """``(B, P+h*w, C)`` -> ``(B*n*n, P+(h/n)*(w/n), C)`` with the prefix copied into every window."""
        b, _, c = x.shape
        n = self.num_windows
        prefix, patches = x[:, : self.num_prefix_tokens], x[:, self.num_prefix_tokens :]
        patches = patches.reshape(b, n, h // n, n, w // n, c).permute(0, 1, 3, 2, 4, 5)
        patches = patches.reshape(b * n * n, (h // n) * (w // n), c)
        prefix = prefix.repeat_interleave(n * n, dim=0)
        return torch.cat([prefix, patches], dim=1)

    def _window_merge(self, x: Tensor, b: int, h: int, w: int) -> Tensor:
        """Inverse of :meth:`_window_partition`; the per-window prefix copies are averaged."""
        n = self.num_windows
        c = x.shape[-1]
        prefix, patches = x[:, : self.num_prefix_tokens], x[:, self.num_prefix_tokens :]
        prefix = prefix.reshape(b, n * n, self.num_prefix_tokens, c).mean(dim=1)
        patches = patches.reshape(b, n, n, h // n, w // n, c).permute(0, 1, 3, 2, 4, 5).reshape(b, h * w, c)
        return torch.cat([prefix, patches], dim=1)

    def _window_rope(self, rope: Tensor, b: int, h: int, w: int) -> Tensor:
        """``(h*w, D)`` -> ``(B*n*n, 1, (h/n)*(w/n), D)``: each window keeps its patches' global coordinates."""
        n = self.num_windows
        d = rope.shape[-1]
        rope = rope.reshape(n, h // n, n, w // n, d).permute(0, 2, 1, 3, 4).reshape(n * n, 1, -1, d)
        return rope.repeat(b, 1, 1, 1)

    def _run_layer(self, layer: nn.Module, x: Tensor, rope: tuple[Tensor, Tensor]) -> Tensor:
        if self.gradient_checkpointing and self.training:
            return cast(Tensor, checkpoint(layer, x, None, rope, use_reentrant=False))
        return cast(Tensor, layer(x, position_embeddings=rope))

    def forward(self, x: Tensor) -> list[Tensor]:
        block_size = self.patch_size * self.num_windows
        assert x.shape[2] % block_size == 0 and x.shape[3] % block_size == 0, (
            f"Backbone requires input shape to be divisible by {block_size}, but got {x.shape}"
        )
        b = x.shape[0]
        h, w = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size

        x = x.to(self.model.embeddings.patch_embeddings.weight.dtype)
        rope = self.model.rope_embeddings(x)
        windowed_rope = None
        if self.window_block_indexes:
            windowed_rope = (self._window_rope(rope[0], b, h, w), self._window_rope(rope[1], b, h, w))
        hidden = self.model.embeddings(x)

        feats = []
        last = self.out_feature_indexes[-1]
        for i, layer in enumerate(self.blocks[:last]):
            if i in self.window_block_indexes:
                assert windowed_rope is not None
                hidden = self._window_partition(hidden, h, w)
                hidden = self._run_layer(layer, hidden, windowed_rope)
                hidden = self._window_merge(hidden, b, h, w)
            else:
                hidden = self._run_layer(layer, hidden, rope)
            if i + 1 in self.out_feature_indexes:
                patches = self.model.norm(hidden)[:, self.num_prefix_tokens :]
                feats.append(patches.transpose(1, 2).reshape(b, -1, h, w).contiguous())
        return feats


class DinoV3Backbone(Backbone):
    """RF-DETR backbone with a DINOv3 encoder, selected by ``encoder="dinov3[_windowed]_<size>"``."""

    def _build_encoder(
        self,
        name: str,
        *,
        out_feature_indexes: list[int] | None,
        gradient_checkpointing: bool,
        load_pretrained_weights: bool,
        patch_size: int,
        num_windows: int,
        drop_path: float,
        window_block_indexes: list[int] | None,
        **kwargs: Any,
    ) -> nn.Module:
        if name not in DINOV3_ENCODER_NAMES:
            raise ValueError(f"Unknown DINOv3 encoder {name!r}; expected one of {DINOV3_ENCODER_NAMES}.")
        if patch_size != 16:
            raise ValueError(f"DINOv3 encoders use patch_size=16, got patch_size={patch_size}.")
        parts = name.split("_")
        return DinoV3(
            size=parts[-1],
            out_feature_indexes=out_feature_indexes,
            use_windowed_attn="windowed" in parts,
            num_windows=num_windows,
            window_block_indexes=window_block_indexes,
            load_pretrained_weights=load_pretrained_weights,
            gradient_checkpointing=gradient_checkpointing,
            drop_path_rate=drop_path,
        )
