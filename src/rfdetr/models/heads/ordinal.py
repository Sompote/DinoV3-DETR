# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Mixture-of-experts ordinal grading head.

Some label sets carry an ordered grade inside the class: rock-mass codes ``A01 ... E03`` grade A to E, damage classes
grade slight to severe. The stock class head scores every class with an independent sigmoid, so its loss treats a
one-grade error exactly like a four-grade error. This head sits *beside* the class head, reads the same decoder query
and predicts the grade with rank-consistent ordinal regression. It never changes which objects are detected, their
scores, boxes or masks; at inference it only decides the grade of each detection, by fusing its grade distribution
with the grade evidence of the class head.

Each of ``K`` experts is a CORN head (Shi et al., 2023): one hyperplane per threshold and cumulative probabilities
``P(L > k)`` built as a chain product of conditional sigmoids, so they are monotone by construction. A softmax gate
read from the query mixes the experts on the cumulative probabilities, which keeps the mixture monotone, and the head
returns the conditional logits of the mixture so that the loss and the fusion are written once for a single CORN
head. With ``num_experts=1`` the head is a plain CORN head.

Graded classes must be the first ``num_levels * num_subclasses`` class ids, level-major (level 0 sub-classes first,
then level 1, ...). Class ids beyond that block (for example a shotcrete class) are not graded and pass through.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

__all__ = ["OrdinalMoEHead"]

#: Logit written into the non-selected graded classes at fusion: sigmoid(-12) ~ 6e-6, so they never pass a threshold.
_SUPPRESSED_LOGIT = -12.0


class OrdinalMoEHead(nn.Module):
    """``num_experts`` CORN experts mixed by a softmax gate on the cumulative probabilities.

    Args:
        dim: Width of the decoder query.
        num_levels: Number of ordered grades (5 for A to E).
        num_subclasses: Classes per grade, laid out level-major in the first ``num_levels * num_subclasses`` ids.
        num_experts: Number of CORN experts; 1 gives a single CORN head without a gate effect.

    Examples:
        >>> head = OrdinalMoEHead(dim=8, num_levels=5, num_subclasses=3, num_experts=3)
        >>> z = head(torch.zeros(2, 8))
        >>> tuple(z.shape)
        (2, 4)
        >>> [round(p, 3) for p in OrdinalMoEHead.level_probs(z)[0].tolist()]  # neutral start: uniform grades
        [0.2, 0.2, 0.2, 0.2, 0.2]
    """

    def __init__(self, dim: int, num_levels: int, num_subclasses: int = 1, num_experts: int = 3) -> None:
        super().__init__()
        if num_levels < 2:
            raise ValueError(f"num_levels must be at least 2, got {num_levels}.")
        if num_subclasses < 1 or num_experts < 1:
            raise ValueError("num_subclasses and num_experts must be at least 1.")
        self.num_levels = num_levels
        self.num_subclasses = num_subclasses
        self.num_experts = num_experts
        thresholds = num_levels - 1
        self.experts = nn.ModuleList([nn.Linear(dim, thresholds) for _ in range(num_experts)])
        # Biases whose chain product gives a uniform grade distribution: logit((t - j) / (t + 1 - j)).
        neutral = torch.tensor(
            [math.log(c / (1 - c)) for c in [(thresholds - j) / (thresholds + 1 - j) for j in range(thresholds)]]
        )
        with torch.no_grad():
            for expert in self.experts:
                expert.bias.copy_(neutral)
        # Zero gate: equal mixing at the start; expert weights keep their different default inits so they can split.
        self.gate = nn.Linear(dim, num_experts)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    @property
    def num_graded_classes(self) -> int:
        """Number of leading class ids that carry a grade."""
        return self.num_levels * self.num_subclasses

    def forward(self, h: Tensor) -> Tensor:
        """Map queries ``(..., dim)`` to conditional ordinal logits ``(..., num_levels - 1)``."""
        gate = torch.softmax(self.gate(h).float(), dim=-1)
        cum = torch.stack([torch.cumprod(torch.sigmoid(e(h).float()), dim=-1) for e in self.experts], dim=-2)
        c = (gate[..., None] * cum).sum(dim=-2).clamp(1e-6, 1 - 1e-6)  # P(L > k), monotone in k
        cond = torch.cat([c[..., :1], c[..., 1:] / c[..., :-1]], dim=-1).clamp(1e-6, 1 - 1e-6)
        return (torch.log(cond) - torch.log1p(-cond)).to(h.dtype)

    @staticmethod
    def level_probs(z: Tensor) -> Tensor:
        """Conditional logits ``(..., T)`` -> grade probabilities ``(..., T + 1)``."""
        c = torch.cumprod(torch.sigmoid(z.float()), dim=-1)
        return torch.cat([1 - c[..., :1], c[..., :-1] - c[..., 1:], c[..., -1:]], dim=-1).clamp_min(1e-6)

    @staticmethod
    def corn_loss(z: Tensor, level: Tensor) -> Tensor:
        """CORN loss: threshold ``k`` is trained on instances with ``level >= k`` against ``1[level > k]``.

        Args:
            z: Conditional logits ``(N, T)`` of the matched, graded instances.
            level: Their true grades ``(N,)``.

        Returns:
            Mean binary cross-entropy over all (instance, threshold) terms; zero (with a graph) when ``N == 0``.

        Examples:
            >>> z = torch.tensor([[5.0, -5.0, -5.0, -5.0]])  # confident "grade 1"
            >>> round(OrdinalMoEHead.corn_loss(z, torch.tensor([1])).item(), 4)
            0.0067
        """
        thresholds = torch.arange(z.shape[-1], device=z.device)
        active = level[:, None] >= thresholds[None, :]
        target = (level[:, None] > thresholds[None, :]).float()
        bce = F.binary_cross_entropy_with_logits(z.float(), target, reduction="none")
        return (bce * active).sum() / active.sum().clamp_min(1)

    def fuse(self, class_logits: Tensor, z: Tensor) -> Tensor:
        """Replace the grade of each query by ``argmax(p_class(level) * p_ordinal(level))``.

        The query keeps its best graded score and that score's sub-class; the other graded classes are suppressed.
        Ungraded classes are untouched, so detection, scores and masks do not change.

        Args:
            class_logits: ``(..., C)`` class-head logits.
            z: ``(..., num_levels - 1)`` conditional logits of this head.

        Returns:
            Fused logits with the shape and dtype of ``class_logits``.
        """
        n_codes = self.num_graded_classes
        scores = torch.sigmoid(class_logits[..., :n_codes].float()).unflatten(-1, (self.num_levels, self.num_subclasses))
        p_class = scores.amax(dim=-1)
        p_class = p_class / p_class.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        level = (p_class * self.level_probs(z)).argmax(dim=-1)
        best, best_index = scores.flatten(-2).max(dim=-1)
        best = best.clamp(1e-6, 1 - 1e-6)
        code = level * self.num_subclasses + best_index % self.num_subclasses
        fused = torch.full_like(scores.flatten(-2), _SUPPRESSED_LOGIT)
        fused = fused.scatter(-1, code[..., None], (torch.log(best) - torch.log1p(-best))[..., None])
        return torch.cat([fused, class_logits[..., n_codes:].float()], dim=-1).to(class_logits.dtype)
