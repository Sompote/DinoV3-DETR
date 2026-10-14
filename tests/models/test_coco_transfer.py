# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Transfer of RF-DETR COCO weights into a DINOv3 model, without the encoder."""

import argparse
from pathlib import Path

import pytest
import torch

import rfdetr.models.backbone.dinov3 as dinov3_module
from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import DinoV3DETRSegNanoConfig, TrainConfig
from rfdetr.models.lwdetr import LWDETR, build_model
from rfdetr.models.weights import load_pretrain_weights


@pytest.fixture(autouse=True)
def _random_dinov3(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build DINOv3 encoders from random weights, so no gated Hub download is needed."""
    monkeypatch.setattr(
        dinov3_module.DINOv3ViTModel,
        "from_pretrained",
        classmethod(lambda cls, name, config: cls(config)),
    )


def _model(config: DinoV3DETRSegNanoConfig) -> LWDETR:
    """Build the bare network for *config*.

    Examples:
        Builds a full network, so the example is not run:

        >>> model = _model(DinoV3DETRSegNanoConfig(coco_pretrain_weights=None))  # doctest: +SKIP
    """
    args = _namespace_from_configs(config, TrainConfig(dataset_dir=None, output_dir="output"))
    model = build_model(args)
    assert isinstance(model, LWDETR)
    return model


def _fake_rfdetr_checkpoint(model: LWDETR, path: Path) -> dict[str, torch.Tensor]:
    """Write an RF-DETR-style checkpoint: the model's non-encoder weights shifted by +1, plus DINOv2 encoder keys.

    Examples:
        Needs a built model, so the example is not run:

        >>> state = _fake_rfdetr_checkpoint(model, tmp_path / "x.pt")  # doctest: +SKIP
    """
    state = {k: v.clone() + 1.0 for k, v in model.state_dict().items() if not k.startswith("backbone.0.encoder.")}
    # DINOv2 encoder names and an incompatible patch size, as in a real RF-DETR Seg checkpoint.
    state["backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight"] = torch.zeros(384, 3, 12, 12)
    state["backbone.0.encoder.encoder.embeddings.position_embeddings"] = torch.zeros(1, 677, 384)
    args = argparse.Namespace(patch_size=12, segmentation_head=True, num_queries=100, group_detr=13)
    torch.save({"model": state, "args": args}, path)
    return state


def test_transfer_loads_everything_but_the_encoder(tmp_path: Path) -> None:
    config = DinoV3DETRSegNanoConfig(coco_pretrain_weights=None)
    model = _model(config)
    encoder_before = {k: v.clone() for k, v in model.backbone[0].encoder.state_dict().items()}
    checkpoint = _fake_rfdetr_checkpoint(model, tmp_path / "rf-detr-seg-nano.pt")

    load_pretrain_weights(model, config, weights_path=str(tmp_path / "rf-detr-seg-nano.pt"), skip_encoder=True)

    after = model.state_dict()
    transferred = [k for k in checkpoint if k in after and not k.startswith("backbone.0.encoder.")]
    assert any(k.startswith("backbone.0.projector.") for k in transferred)
    assert any(k.startswith("transformer.decoder.") for k in transferred)
    assert any(k.startswith("segmentation_head.") for k in transferred)
    assert all(torch.equal(after[k], checkpoint[k]) for k in transferred)
    assert all(torch.equal(v, encoder_before[k]) for k, v in model.backbone[0].encoder.state_dict().items())


def test_transfer_skips_projector_of_a_wider_encoder(tmp_path: Path) -> None:
    small = _model(DinoV3DETRSegNanoConfig(coco_pretrain_weights=None))
    _fake_rfdetr_checkpoint(small, tmp_path / "rf-detr-seg-nano.pt")
    config = DinoV3DETRSegNanoConfig(coco_pretrain_weights=None, encoder="dinov3_windowed_base")
    base = _model(config)
    small_shapes = {k: v.shape for k, v in small.state_dict().items()}
    wide = {
        k: v.clone()
        for k, v in base.state_dict().items()
        if k in small_shapes and v.shape != small_shapes[k] and not k.startswith("backbone.0.encoder.")
    }
    assert wide, "a 768-wide encoder must change some projector shapes"

    load_pretrain_weights(base, config, weights_path=str(tmp_path / "rf-detr-seg-nano.pt"), skip_encoder=True)

    # Tensors whose shape follows the encoder width cannot transfer and keep their own init.
    after = base.state_dict()
    assert all(torch.equal(after[k], v) for k, v in wide.items())


def test_variants_default_to_their_coco_checkpoint() -> None:
    assert DinoV3DETRSegNanoConfig().coco_pretrain_weights == "rf-detr-seg-nano.pt"
