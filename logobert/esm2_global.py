"""ESM-2 encoder and Global branch for LoGoPPI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoModel, EsmConfig, EsmModel, PreTrainedModel
from transformers.utils import ModelOutput

from .config import ESM2PPIConfig


@dataclass
class ESM2PPIOutput(ModelOutput):
    """Model output with logits [B] and pooled representations [B, D]."""

    loss: Tensor | None = None
    logits: Tensor | None = None
    logits_ab: Tensor | None = None
    logits_ba: Tensor | None = None
    pooled_a: Tensor | None = None
    pooled_b: Tensor | None = None


class ESM2ForPPI(PreTrainedModel):
    """Encode two proteins with ESM-2 and average their AB and BA logits.

    Evaluation mode computes symmetric scores for the input order.
    In training mode, repeated calls may differ due to dropout.
    """

    config_class = ESM2PPIConfig
    base_model_prefix = "encoder"
    main_input_name = "input_a"
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: ESM2PPIConfig,
        encoder: nn.Module | None = None,
    ) -> None:
        super().__init__(config)

        # Build the encoder from the stored configuration when none is supplied.
        if encoder is None:
            if config.encoder_config is not None:
                # This creates the architecture without loading pretrained weights.
                esm_config = EsmConfig.from_dict(config.encoder_config)
                encoder = EsmModel(
                    esm_config,
                    add_pooling_layer=False,
                )
            else:
                encoder = AutoModel.from_pretrained(
                    config.base_model_name,
                    revision=config.base_model_revision,
                    add_pooling_layer=False,
                )
                config.encoder_config = encoder.config.to_dict()

        encoder_config = getattr(encoder, "config", None)
        hidden_size = getattr(encoder_config, "hidden_size", None)

        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError("encoder.config.hidden_size must be positive")

        self.encoder = encoder

        # Project each residue representation to the shared embedding dimension.
        self.projection = nn.Linear(
            hidden_size,
            config.projection_dim,
        )
        self.projection_dropout = nn.Dropout(config.projection_dropout)

        # Classifier input: [pooled_a, pooled_b, |pooled_a - pooled_b|].
        pair_dim = 3 * config.projection_dim

        self.global_layer_norm = nn.LayerNorm(pair_dim)
        self.global_mlp = nn.Sequential(
            nn.Linear(pair_dim, config.global_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.global_hidden_dim, 1),
        )

        # Store the fixed loss weight with the model and move it across devices.
        self.register_buffer(
            "pos_weight",
            torch.tensor([config.pos_weight], dtype=torch.float32),
        )

        self._freeze_unused_encoder_parameters()
        self._initialize_added_modules()

    def _freeze_unused_encoder_parameters(self) -> None:
        """Freeze encoder parameters that are not used for PPI scoring."""
        unused_names = {
            "pooler.dense.weight",
            "pooler.dense.bias",
            "contact_head.regression.weight",
            "contact_head.regression.bias",
        }

        position_type = getattr(
            self.encoder.config,
            "position_embedding_type",
            None,
        )

        if position_type == "rotary":
            unused_names.add("embeddings.position_embeddings.weight")

        for name, parameter in self.encoder.named_parameters():
            if name in unused_names:
                parameter.requires_grad_(False)

    def _initialize_added_modules(self) -> None:
        """Initialize the projection and Global classifier layers."""
        added_modules = (
            self.projection,
            self.global_layer_norm,
            self.global_mlp,
        )

        for module in added_modules:
            module.apply(self._init_weights)

    @staticmethod
    def _validate_input(batch: Mapping[str, Tensor], name: str) -> None:
        """Check that required token tensors exist and have matching shapes."""
        required_keys = {
            "input_ids",
            "attention_mask",
            "special_tokens_mask",
        }

        missing_keys = required_keys.difference(batch.keys())

        if missing_keys:
            raise KeyError(f"{name} missing {sorted(missing_keys)}")

        token_shape = batch["input_ids"].shape
        attention_shape = batch["attention_mask"].shape
        special_shape = batch["special_tokens_mask"].shape

        if token_shape != attention_shape or token_shape != special_shape:
            raise ValueError(f"{name} token/mask shapes differ")

    @staticmethod
    def residue_mask(batch: Mapping[str, Tensor]) -> Tensor:
        """Return residue positions [B, L], excluding padding and special tokens."""
        valid_tokens = batch["attention_mask"].bool()
        special_tokens = batch["special_tokens_mask"].bool()

        mask = valid_tokens & ~special_tokens

        has_residue = mask.any(dim=1)

        if not has_residue.all().item():
            raise ValueError(
                "every protein must contain at least one residue token"
            )

        return mask

    def encode(
        self,
        batch: Mapping[str, Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Return projected residue representations [B, L, D] and their mask."""
        self._validate_input(batch, "protein batch")

        encoder_output = self.encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        embeddings = encoder_output.last_hidden_state
        projected = self.projection(embeddings)
        projected = self.projection_dropout(projected)

        mask = self.residue_mask(batch)

        return projected, mask

    @staticmethod
    def mean_pool(
        projected: Tensor,
        residue_mask: Tensor,
    ) -> Tensor:
        """Average true residue positions into protein representations [B, D]."""
        if projected.ndim != 3:
            raise ValueError("projected embeddings must have shape [B, L, D]")

        if residue_mask.shape != projected.shape[:2]:
            raise ValueError("projected embeddings and residue mask shapes differ")

        # [B, L] -> [B, L, 1] applies one mask value to every residue dimension.
        weights = residue_mask.to(dtype=projected.dtype)
        weights = weights.unsqueeze(-1)

        residue_sum = (projected * weights).sum(dim=1)
        residue_count = weights.sum(dim=1).clamp_min(1)

        return residue_sum / residue_count

    def _ordered_logit(
        self,
        pooled_a: Tensor,
        pooled_b: Tensor,
    ) -> Tensor:
        """Compute the Global logit for one protein order."""
        difference = (pooled_a - pooled_b).abs()

        features = torch.cat(
            [pooled_a, pooled_b, difference],
            dim=-1,
        )

        features = self.global_layer_norm(features)
        logits = self.global_mlp(features)

        return logits.squeeze(-1)

    def score_projected(
        self,
        projected_a: Tensor,
        residue_mask_a: Tensor,
        projected_b: Tensor,
        residue_mask_b: Tensor,
        labels: Tensor | None = None,
        return_dict: bool = True,
    ) -> ESM2PPIOutput | tuple[Tensor, ...]:
        """Compute Global logits and optional loss from projected residues."""
        pooled_a = self.mean_pool(projected_a, residue_mask_a)
        pooled_b = self.mean_pool(projected_b, residue_mask_b)

        # Apply the shared classifier to both protein orders.
        logits_ab = self._ordered_logit(pooled_a, pooled_b)
        logits_ba = self._ordered_logit(pooled_b, pooled_a)

        logits = (logits_ab + logits_ba) * 0.5

        loss = None

        if labels is not None:
            targets = labels.to(
                device=logits.device,
                dtype=logits.dtype,
            )
            targets = targets.reshape_as(logits)

            pos_weight = self.pos_weight.to(
                device=logits.device,
                dtype=logits.dtype,
            )

            loss = F.binary_cross_entropy_with_logits(
                logits,
                targets,
                pos_weight=pos_weight,
            )

        if return_dict:
            return ESM2PPIOutput(
                loss=loss,
                logits=logits,
                logits_ab=logits_ab,
                logits_ba=logits_ba,
                pooled_a=pooled_a,
                pooled_b=pooled_b,
            )

        outputs = (
            logits,
            logits_ab,
            logits_ba,
            pooled_a,
            pooled_b,
        )

        if loss is not None:
            return (loss,) + outputs

        return outputs

    def forward(
        self,
        input_a: Mapping[str, Tensor],
        input_b: Mapping[str, Tensor],
        labels: Tensor | None = None,
        return_dict: bool | None = None,
    ) -> ESM2PPIOutput | tuple[Tensor, ...]:
        """Encode two proteins and compute their symmetric Global logit."""
        if return_dict is None:
            return_dict = self.config.use_return_dict

        projected_a, mask_a = self.encode(input_a)
        projected_b, mask_b = self.encode(input_b)

        return self.score_projected(
            projected_a=projected_a,
            residue_mask_a=mask_a,
            projected_b=projected_b,
            residue_mask_b=mask_b,
            labels=labels,
            return_dict=return_dict,
        )


class GlobalHead(nn.Module):
    """Compute Global logits from stored projected residue representations."""

    def __init__(
        self,
        dimension: int = 512,
        hidden_dimension: int = 512,
    ) -> None:
        super().__init__()

        pair_dimension = 3 * dimension

        self.layer_norm = nn.LayerNorm(pair_dimension)
        self.mlp = nn.Sequential(
            nn.Linear(pair_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_dimension, 1),
        )

    @classmethod
    def from_step1_state(
        cls,
        state: Mapping[str, Tensor],
    ) -> GlobalHead:
        """Load only the Global classifier from a Step 1 checkpoint."""
        model = cls()
        head_state = {}

        for name, weight in state.items():
            if name.startswith("global_layer_norm."):
                suffix = name.removeprefix("global_layer_norm.")
                head_state[f"layer_norm.{suffix}"] = weight

            elif name.startswith("global_mlp."):
                suffix = name.removeprefix("global_mlp.")
                head_state[f"mlp.{suffix}"] = weight

        model.load_state_dict(head_state, strict=True)

        model.eval()
        model.requires_grad_(False)

        return model

    @staticmethod
    def mean_pool(embeddings: Tensor, mask: Tensor) -> Tensor:
        """Average cached residue representations in FP32."""
        embeddings = embeddings.float()
        weights = mask.to(dtype=torch.float32)
        weights = weights.unsqueeze(-1)

        residue_sum = (embeddings * weights).sum(dim=1)
        residue_count = weights.sum(dim=1).clamp_min(1)

        return residue_sum / residue_count

    def ordered(
        self,
        pooled_a: Tensor,
        pooled_b: Tensor,
    ) -> Tensor:
        """Compute the Global logit for one protein order."""
        difference = (pooled_a - pooled_b).abs()

        features = torch.cat(
            [pooled_a, pooled_b, difference],
            dim=-1,
        )

        features = self.layer_norm(features)
        logits = self.mlp(features)

        return logits.squeeze(-1)

    def forward(
        self,
        a: Tensor,
        mask_a: Tensor,
        b: Tensor,
        mask_b: Tensor,
    ) -> Tensor:
        pooled_a = self.mean_pool(a, mask_a)
        pooled_b = self.mean_pool(b, mask_b)

        logits_ab = self.ordered(pooled_a, pooled_b)
        logits_ba = self.ordered(pooled_b, pooled_a)

        return (logits_ab + logits_ba) * 0.5
