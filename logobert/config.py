"""Configuration for the LoGoPPI model."""

from __future__ import annotations

from transformers import PretrainedConfig


class ESM2PPIConfig(PretrainedConfig):
    """Configuration for the LoGoPPI ESM-2 encoder and Global classifier."""

    model_type = "esm2_ppi"

    def __init__(
        self,
        # ESM-2 encoder
        base_model_name: str = "facebook/esm2_t33_650M_UR50D",
        base_model_revision: str | None = None,
        encoder_config: dict | None = None,
        # Projection and global classifier
        projection_dim: int = 512,
        global_hidden_dim: int = 512,
        projection_dropout: float = 0.1,
        classifier_dropout: float = 0.1,
        # Training and input settings
        pos_weight: float = 10.0,
        max_residues: int = 800,
        exclude_special_tokens: bool = True,
        # Global symmetry and loss
        global_symmetry: str = "mean_ordered_logits",
        loss_definition: str = "bce_on_mean_ordered_logits",
        **kwargs: object,
    ) -> None:
        kwargs.setdefault("architectures", ["ESM2ForPPI"])
        super().__init__(**kwargs)

        # Validate parameters
        if not base_model_name:
            raise ValueError("base_model_name must not be empty")

        if projection_dim <= 0:
            raise ValueError("projection_dim must be positive")

        if global_hidden_dim <= 0:
            raise ValueError("global_hidden_dim must be positive")

        if not (0 <= projection_dropout < 1):
            raise ValueError("projection_dropout must be in [0, 1)")

        if not (0 <= classifier_dropout < 1):
            raise ValueError("classifier_dropout must be in [0, 1)")

        if pos_weight <= 0:
            raise ValueError("pos_weight must be positive")

        if max_residues <= 0:
            raise ValueError("max_residues must be positive")

        if not exclude_special_tokens:
            raise ValueError("exclude_special_tokens must be True")

        if global_symmetry != "mean_ordered_logits":
            raise ValueError(
                "global_symmetry must be 'mean_ordered_logits'"
            )

        if loss_definition != "bce_on_mean_ordered_logits":
            raise ValueError(
                "loss_definition must be 'bce_on_mean_ordered_logits'"
            )

        # Encoder settings
        self.base_model_name = str(base_model_name)
        self.base_model_revision = base_model_revision
        self.encoder_config = encoder_config

        # Projection and classifier settings
        self.projection_dim = int(projection_dim)
        self.global_hidden_dim = int(global_hidden_dim)
        self.projection_dropout = float(projection_dropout)
        self.classifier_dropout = float(classifier_dropout)

        # Training and input data settings
        self.pos_weight = float(pos_weight)
        self.max_residues = int(max_residues)
        self.exclude_special_tokens = bool(exclude_special_tokens)

        # Global symmetry and loss definition
        self.global_symmetry = global_symmetry
        self.loss_definition = loss_definition
