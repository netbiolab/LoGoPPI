"""Symmetric Maxsim scoring and Step 2 training modules for LoGoPPI."""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _validate_embeddings(
    embeddings: Tensor,
    mask: Tensor,
    name: str,
) -> None:
    """Validate embedding shapes and ensure that every protein has residues."""
    if embeddings.ndim != 3:
        raise ValueError(
            f"{name} embeddings must have shape [batch, length, dim]"
        )

    if mask.shape != embeddings.shape[:2]:
        raise ValueError(
            f"{name} mask must match embeddings batch/length"
        )

    has_residue = mask.bool().any(dim=1)

    if not has_residue.all().item():
        raise ValueError(
            f"{name} contains a protein with no residues"
        )


def symmetric_maxsim(
    embeddings_a: Tensor,
    mask_a: Tensor,
    embeddings_b: Tensor,
    mask_b: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute both directional Maxsim scores and their symmetric mean.

    Masks must be true only for residues and false for padding and special
    tokens. All similarity calculations are performed in FP32.
    """
    _validate_embeddings(embeddings_a, mask_a, "A")
    _validate_embeddings(embeddings_b, mask_b, "B")

    if embeddings_a.shape[0] != embeddings_b.shape[0]:
        raise ValueError("A and B batch sizes differ")

    if embeddings_a.shape[2] != embeddings_b.shape[2]:
        raise ValueError("A and B embedding dimensions differ")

    valid_a = mask_a.bool()
    valid_b = mask_b.bool()

    # Disable CUDA autocast so that Maxsim is always computed in FP32.
    if embeddings_a.device.type == "cuda":
        precision_context = torch.autocast(
            device_type="cuda",
            enabled=False,
        )
    else:
        precision_context = contextlib.nullcontext()

    with precision_context:
        embeddings_a_fp32 = embeddings_a.float()
        embeddings_b_fp32 = embeddings_b.float()

        # [B, LA, D] x [B, D, LB] -> [B, LA, LB].
        similarity = torch.bmm(
            embeddings_a_fp32,
            embeddings_b_fp32.transpose(1, 2),
        )

        # Prevent masked positions from being selected as maximum values.
        lowest_value = torch.finfo(similarity.dtype).min

        # A -> B: find the best matching residue in B for every residue in A.
        invalid_b = ~valid_b.unsqueeze(1)
        similarity_ab = similarity.masked_fill(
            invalid_b,
            lowest_value,
        )
        best_match_a = similarity_ab.max(dim=2).values
        del similarity_ab

        # B -> A: find the best matching residue in A for every residue in B.
        invalid_a = ~valid_a.unsqueeze(2)
        similarity_ba = similarity.masked_fill(
            invalid_a,
            lowest_value,
        )
        best_match_b = similarity_ba.max(dim=1).values
        del similarity_ba

        # Average the best matches over true residue positions only.
        score_sum_a = (best_match_a * valid_a.float()).sum(dim=1)
        residue_count_a = valid_a.sum(dim=1).float()
        score_ab = score_sum_a / residue_count_a

        score_sum_b = (best_match_b * valid_b.float()).sum(dim=1)
        residue_count_b = valid_b.sum(dim=1).float()
        score_ba = score_sum_b / residue_count_b

        symmetric_score = (score_ab + score_ba) * 0.5

    return symmetric_score, score_ab, score_ba


def inverse_softplus(value: float) -> float:
    """Return the input whose softplus equals ``value``."""
    if value <= 0:
        raise ValueError("inverse softplus input must be positive")

    return math.log(math.expm1(value))


class MaxSimAdapter(nn.Module):
    """Apply a bias-free residue projection initialized to the identity."""

    def __init__(self, dimension: int = 512) -> None:
        super().__init__()

        if dimension <= 0:
            raise ValueError("adapter dimension must be positive")

        self.linear = nn.Linear(
            dimension,
            dimension,
            bias=False,
        )

        # Identity initialization preserves residue representations initially.
        with torch.no_grad():
            identity_matrix = torch.eye(dimension)
            self.linear.weight.copy_(identity_matrix)

    def forward(self, embeddings: Tensor) -> Tensor:
        return self.linear(embeddings)


@dataclass
class MaxSimStep2Output:
    """Store Step 2 scores, loss, and affine parameters."""

    loss: Tensor | None
    maxsim_logit: Tensor
    maxsim_score: Tensor
    maxsim_score_ab: Tensor
    maxsim_score_ba: Tensor
    maxsim_scale: Tensor
    maxsim_bias: Tensor


class MaxSimStep2Model(nn.Module):
    """Train an optional adapter and temporary affine Maxsim parameters."""

    def __init__(
        self,
        dimension: int = 512,
        use_adapter: bool = True,
        pos_weight: float = 10.0,
    ) -> None:
        super().__init__()

        if pos_weight <= 0:
            raise ValueError("pos_weight must be positive")

        self.dimension = int(dimension)

        # Share one adapter between both proteins.
        if use_adapter:
            self.adapter = MaxSimAdapter(dimension)
        else:
            self.adapter = nn.Identity()

        # Initialize the positive scale to approximately one after softplus.
        initial_scale_raw = inverse_softplus(1.0)

        self.maxsim_scale_raw = nn.Parameter(
            torch.tensor(initial_scale_raw, dtype=torch.float32)
        )
        self.maxsim_bias = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float32)
        )

        self.register_buffer(
            "pos_weight",
            torch.tensor([pos_weight], dtype=torch.float32),
        )

    def forward(
        self,
        embeddings_a: Tensor,
        mask_a: Tensor,
        embeddings_b: Tensor,
        mask_b: Tensor,
        labels: Tensor | None = None,
    ) -> MaxSimStep2Output:
        # 1. Transform both residue representations with the shared adapter.
        adapted_a = self.adapter(embeddings_a)
        adapted_b = self.adapter(embeddings_b)

        # 2. Compute directional Maxsim scores and their symmetric mean.
        maxsim_score, score_ab, score_ba = symmetric_maxsim(
            embeddings_a=adapted_a,
            mask_a=mask_a,
            embeddings_b=adapted_b,
            mask_b=mask_b,
        )

        # 3. Convert the Maxsim score into a logit for BCE training.
        scale = F.softplus(self.maxsim_scale_raw.float()) + 1e-8
        bias = self.maxsim_bias.float()

        normalized_score = maxsim_score / math.sqrt(self.dimension)
        logit = scale * normalized_score + bias

        # 4. Compute the loss only when labels are supplied.
        loss = None

        if labels is not None:
            targets = labels.float().reshape_as(logit)

            loss = F.binary_cross_entropy_with_logits(
                logit,
                targets,
                pos_weight=self.pos_weight.float(),
            )

        return MaxSimStep2Output(
            loss=loss,
            maxsim_logit=logit,
            maxsim_score=maxsim_score,
            maxsim_score_ab=score_ab,
            maxsim_score_ba=score_ba,
            maxsim_scale=scale,
            maxsim_bias=self.maxsim_bias,
        )
