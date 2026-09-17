"""Evaluate a released LoGoPPI model with the shared cached scorer."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from inference import read_inputs
from logobert.scoring import apply_calibration
from utils.prediction import (
    FinalPredictor,
    bootstrap_aupr,
    classification_metrics,
    place_embeddings,
)


def main(args: argparse.Namespace) -> None:
    """Evaluate one released model without fitting new calibration parameters."""
    # Resolve the prepared test pairs and the matching FASTA from one config.
    config = yaml.safe_load(args.config.read_text())
    paths = config["paths"]
    pair_csv = resolve_path(args.config, paths["test_csv"])
    fasta_path = resolve_path(args.config, paths["fasta"])
    frame = pd.read_csv(pair_csv)
    labels = frame["label"].to_numpy(dtype=np.int64)
    if not np.isin(labels, (0, 1)).all() or len(np.unique(labels)) != 2:
        raise ValueError("test data must contain both binary labels")

    # Reuse the public inference reader so test and prediction accept the same IDs.
    input_args = argparse.Namespace(
        pair_csv=pair_csv,
        fasta_path=fasta_path,
        sequence_a=None,
        sequence_b=None,
    )
    sequences, pairs = read_inputs(input_args)

    # Generate fresh embeddings and apply the released cached scorer.
    device = torch.device("cuda", 0)
    predictor = FinalPredictor(args.model_dir, device)
    if config["model_format"] != predictor.state["format"]:
        raise RuntimeError("config model_format does not match the model bundle")
    embeddings = predictor.encode(
        sequences,
        int(config["postprocess"]["embedding_batch_size"]),
        args.quiet,
    )
    embeddings, _ = place_embeddings(embeddings, "auto", device)
    arrays = predictor.score_arrays(
        pairs,
        embeddings,
        int(config["postprocess"]["maxsim_pair_batch_size"]),
        args.quiet,
        global_batch_size=int(config["postprocess"]["global_batch_size"]),
    )
    applied = apply_calibration(
        arrays["global_score"], arrays["maxsim_score"], predictor.state
    )

    # Keep input rows and every reported score in the same order.
    predictions = pd.DataFrame(pairs, columns=["query", "text"])
    predictions["label"] = labels
    predictions["global_score"] = arrays["global_score"]
    predictions["maxsim_score"] = arrays["maxsim_score"]
    score_columns = (
        "global_logit",
        "maxsim_logit",
        "final_logit",
        "p_global",
        "p_maxsim",
        "score",
    )
    for name in score_columns:
        predictions[name] = applied[name]

    # Labels are used only for evaluation; calibration is never refit on test data.
    metrics = classification_metrics(labels, applied["final_logit"])
    low, high = bootstrap_aupr(
        labels,
        applied["final_logit"],
        args.bootstrap_replicates,
        args.bootstrap_seed,
    )
    metrics.update(
        {
            "variant": "final",
            "aupr_ci_low": low,
            "aupr_ci_high": high,
            "n": len(labels),
        }
    )

    # Write predictions and their one-row metric summary.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
    metric_frame = pd.DataFrame([metrics])
    metric_frame.to_csv(args.output_dir / "test_metrics.csv", index=False)
    print(metric_frame.to_string(index=False))


def resolve_path(config_path: Path, value: str) -> Path:
    """Resolve a data path relative to its YAML configuration file."""
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a released LoGoPPI model without fitting."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bootstrap_replicates", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    main(parser.parse_args())
