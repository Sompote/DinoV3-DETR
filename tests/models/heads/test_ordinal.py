# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the mixture-of-experts ordinal grading head and its model and loss wiring."""

import types
from typing import Any, cast

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from rfdetr.config import DinoV3DETRNanoConfig
from rfdetr.models.criterion import SetCriterion
from rfdetr.models.heads.ordinal import OrdinalMoEHead
from rfdetr.models.matcher import HungarianMatcher


def _loop_corn_loss(z: torch.Tensor, level: torch.Tensor) -> torch.Tensor:
    """Reference CORN loss written as the per-threshold loop of the original implementation.

    Examples:
        >>> round(_loop_corn_loss(torch.zeros(1, 4), torch.tensor([0])).item(), 4)
        0.6931
    """
    total, n = torch.zeros(()), 0
    for k in range(z.shape[-1]):
        m = level >= k
        if m.any():
            total = total + F.binary_cross_entropy_with_logits(z[m, k], (level[m] > k).float(), reduction="sum")
            n += int(m.sum())
    return total / max(n, 1)


class TestOrdinalMoEHead:
    def test_neutral_biases_give_uniform_grades(self) -> None:
        # Expert weights start random (they must differ to specialise), so the biases alone set the neutral point.
        head = OrdinalMoEHead(dim=16, num_levels=5, num_subclasses=3)
        probs = OrdinalMoEHead.level_probs(head(torch.zeros(4, 16)))
        assert torch.allclose(probs, torch.full_like(probs, 0.2), atol=1e-5)

    def test_cumulative_probabilities_are_monotone(self) -> None:
        head = OrdinalMoEHead(dim=16, num_levels=6, num_experts=4)
        for p in head.parameters():
            torch.nn.init.normal_(p, std=2.0)
        cum = torch.cumprod(torch.sigmoid(head(torch.randn(64, 16)).float()), dim=-1)
        assert bool((cum[..., 1:] <= cum[..., :-1] + 1e-6).all())

    def test_grade_probabilities_sum_to_one(self) -> None:
        z = torch.randn(32, 4)
        assert torch.allclose(OrdinalMoEHead.level_probs(z).sum(-1), torch.ones(32), atol=1e-4)

    def test_mixture_reproduces_gated_cumulative_probabilities(self) -> None:
        head = OrdinalMoEHead(dim=8, num_levels=5, num_experts=3)
        torch.nn.init.normal_(head.gate.weight)
        h = torch.randn(10, 8)
        gate = torch.softmax(head.gate(h), -1)
        expected = torch.stack(
            [gate[:, e, None] * torch.cumprod(torch.sigmoid(head.experts[e](h)), -1) for e in range(3)]
        ).sum(0)
        assert torch.allclose(torch.cumprod(torch.sigmoid(head(h)), -1), expected, atol=1e-5)

    def test_single_expert_is_plain_corn(self) -> None:
        head = OrdinalMoEHead(dim=8, num_levels=5, num_experts=1)
        h = torch.randn(10, 8)
        assert torch.allclose(head(h), head.experts[0](h), atol=1e-4)

    def test_corn_loss_matches_loop_reference(self) -> None:
        z, level = torch.randn(50, 4), torch.randint(0, 5, (50,))
        assert torch.allclose(OrdinalMoEHead.corn_loss(z, level), _loop_corn_loss(z, level), atol=1e-6)

    def test_corn_loss_is_zero_without_instances(self) -> None:
        assert OrdinalMoEHead.corn_loss(torch.zeros(0, 4), torch.zeros(0, dtype=torch.long)).item() == 0.0

    def test_fuse_keeps_best_score_and_subclass(self) -> None:
        head = OrdinalMoEHead(dim=8, num_levels=5, num_subclasses=3)
        logits = torch.full((1, 1, 17), -8.0)
        logits[0, 0, 4] = 2.0  # level 1 (B), sub-class 1
        z = torch.tensor([[[12.0, 12.0, -12.0, -12.0]]])  # ordinal head says level 2 (C)
        fused = head.fuse(logits, z)
        # p_class(B) dominates but p_ord(B) ~ 0 → product picks C; score 2.0 and sub-class 1 are kept.
        assert int(fused[0, 0, :15].argmax()) == 2 * 3 + 1
        assert torch.isclose(fused[0, 0, 7], torch.tensor(2.0), atol=1e-4)

    def test_fuse_leaves_ungraded_classes_untouched(self) -> None:
        head = OrdinalMoEHead(dim=8, num_levels=5, num_subclasses=3)
        logits, z = torch.randn(2, 6, 17), torch.randn(2, 6, 4)
        assert torch.equal(head.fuse(logits, z)[..., 15:], logits[..., 15:])

    def test_fuse_keeps_one_graded_class_per_query(self) -> None:
        head = OrdinalMoEHead(dim=8, num_levels=5, num_subclasses=3)
        fused = head.fuse(torch.randn(2, 6, 17), torch.randn(2, 6, 4))
        assert bool(((fused[..., :15] > -12.0).sum(-1) == 1).all())

    def test_rejects_single_level(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            OrdinalMoEHead(dim=8, num_levels=1)


class TestOrdinalConfig:
    def test_disabled_by_default(self) -> None:
        assert DinoV3DETRNanoConfig().ordinal_levels == 0

    def test_rejects_block_larger_than_label_set(self) -> None:
        with pytest.raises(ValueError, match="exceed num_classes"):
            DinoV3DETRNanoConfig(num_classes=10, ordinal_levels=5, ordinal_subclasses=3)


class TestOrdinalLoss:
    def test_loss_only_uses_matched_graded_queries(self) -> None:
        criterion = SetCriterion(
            17,
            matcher=HungarianMatcher(),
            weight_dict={},
            focal_alpha=0.25,
            losses=["ordinal"],
            ordinal_levels=5,
            ordinal_subclasses=3,
        )
        z = torch.randn(1, 10, 4)
        targets = [{"labels": torch.tensor([7, 15, 0])}]  # level 2, ungraded class 15, level 0
        indices = [(torch.tensor([3, 5, 8]), torch.tensor([0, 1, 2]))]
        loss = criterion.loss_ordinal({"pred_ordinal": z}, targets, indices, torch.tensor(3.0))["loss_ordinal"]
        expected = _loop_corn_loss(z[0, [3, 8]], torch.tensor([2, 0]))
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_layers_without_head_output_are_skipped(self) -> None:
        criterion = SetCriterion(17, matcher=HungarianMatcher(), weight_dict={}, focal_alpha=0.25, losses=["ordinal"])
        assert criterion.loss_ordinal({}, [], [], torch.tensor(1.0)) == {}


def test_builder_attaches_head_and_loss() -> None:
    from rfdetr.models.lwdetr import build_criterion_and_postprocessors

    args = types.SimpleNamespace(
        device="cpu",
        num_classes=16,
        cls_loss_coef=2.0,
        bbox_loss_coef=5.0,
        giou_loss_coef=2.0,
        segmentation_head=False,
        aux_loss=True,
        dec_layers=3,
        two_stage=True,
        focal_alpha=0.25,
        group_detr=1,
        use_varifocal_loss=False,
        use_position_supervised_loss=False,
        ia_bce_loss=True,
        num_select=100,
        set_cost_class=2.0,
        set_cost_bbox=5.0,
        set_cost_giou=2.0,
        ordinal_levels=5,
        ordinal_subclasses=3,
        ordinal_loss_weight=1.5,
    )
    criterion, _ = build_criterion_and_postprocessors(cast(Any, args))
    assert "ordinal" in criterion.losses
    assert criterion.weight_dict["loss_ordinal"] == pytest.approx(3.0)
    assert criterion.weight_dict["loss_ordinal_1"] == pytest.approx(3.0)
