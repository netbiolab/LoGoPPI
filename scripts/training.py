"""Train, calibrate, and export a LoGoPPI model.

The pipeline runs in this order:

1. Train the Global model and select a checkpoint by validation AUPR.
2. Encode each protein once and store its residue representations.
3. Calculate Global and Maxsim scores from the residue cache.
4. Calibrate both score branches and combine their logits with equal weights.
5. Export the model, tokenizer, calibration state, and loading code.

Cross-species and Bernett runs use this same pipeline. Their datasets and
training settings are selected through the YAML configuration file.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist
import yaml
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from logobert.config import ESM2PPIConfig
from logobert.esm2_global import ESM2ForPPI, GlobalHead
from logobert.maxsim import symmetric_maxsim
from logobert.scoring import _vector, apply_calibration, nll, validate_calibration
from utils.data import (
    CachedPairCollator,
    CachedPairDataset,
    DistributedEvalSampler,
    PairCollator,
    ProjectionCache,
    ProteinPair,
    ProteinPairDataset,
    TruncatedDistributedTrainSampler,
    read_fasta,
    write_cache_manifest,
)
from utils.execution import (
    accumulation_context,
    autocast_context,
    gather_variable,
    make_grad_scaler,
    move_batch,
    resolve_precision,
    seed_everything,
    setup_distributed,
    unwrap,
)
from utils.prediction import (
    probability_metrics,
)

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ValidationMetrics:
    """Metrics used to compare Global-model checkpoints."""

    aupr: float
    auroc: float
    weighted_bce: float
    unweighted_nll: float
    f1_0p5: float
    count: int


def main(args: argparse.Namespace) -> None:
    """Run training, cache creation, calibration, and model export."""
    # Internal workers run one distributed stage and then exit.
    if args._worker:
        config = load_config(args.config)
        worker_args = SimpleNamespace(
            config=args.config,
            run_name=args.output_dir.name,
            output_dir=args.output_dir,
            resume=args.resume,
            max_optimizer_steps=args.max_optimizer_steps,
        )
        if args._worker == "global":
            train_model(worker_args, config)
        else:
            score_and_calibrate(config, args.output_dir, args.config)
        return

    # The parent process owns configuration and launches all worker stages.
    if "LOCAL_RANK" in os.environ:
        raise RuntimeError(
            "Run python -m scripts.training; it starts its own GPU workers."
        )
    args.config = args.config.resolve()
    config = load_config(args.config)
    config.setdefault(
        "model_id", f"logoppi-{config['model_format']}-{uuid.uuid4().hex}"
    )
    config["paths"] = {
        name: str(resolve_path(args.config, value).resolve())
        for name, value in config["paths"].items()
    }
    args.output_dir = args.output_dir.resolve()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["runtime"]["visible_gpus"])

    # Create an immutable record of the settings used for this run.
    output = args.output_dir
    training_preflight(config, args.config, require_all_gpus=False)
    if output.exists() and any(output.iterdir()) and args.resume is None:
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    resolved = output / "resolved_config.yaml"
    resolved.write_text(yaml.safe_dump(config, sort_keys=False))

    # Stage 1: train the Global model and select the best checkpoint.
    launch_worker(args, "global", resolved, config["runtime"]["expected_world_size"])
    if args.max_optimizer_steps is not None:
        print("Stopped after the requested optimizer steps; resume with --resume.")
        return
    if not (output / "best.pt").is_file():
        raise FileNotFoundError(output / "best.pt")

    # Stages 2-4: cache embeddings, score pairs, and fit calibration.
    embedding_seconds = build_residue_cache(config, output, resolved, args.quiet)
    launch_worker(args, "score", resolved, config["runtime"]["expected_world_size"])

    metrics_path = output / "postprocess_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["runtime/embedding_seconds"] = embedding_seconds
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    # Stage 5: package and verify the files needed for public inference.
    bundle = export_bundle(config, output)
    print(f"Final model: {bundle}")


def train_model(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Train the Global model and save checkpoints selected by validation AUPR."""
    # 1. Set up distributed training, reproducibility, and numeric precision.
    device, rank, world = setup_distributed()
    if world != int(config["runtime"]["expected_world_size"]):
        raise RuntimeError("training worker count differs from the config")

    seed_everything(int(config["training"]["seed"]), rank)
    if rank == 0:
        training_preflight(config, args.config, require_all_gpus=True)
    if dist.is_initialized():
        dist.barrier()
    resolved = resolve_precision(
        config["runtime"]["requested_precision"],
        config["runtime"]["allow_fp16_fallback"],
    )
    config["runtime"]["resolved_precision"] = resolved
    config["runtime"]["grad_scaler_enabled"] = resolved == "fp16"

    run_dir = args.output_dir
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
    if dist.is_initialized():
        dist.barrier()

    # 2. Build the datasets, model, optimizer, and learning-rate scheduler.
    sequences, train_pairs, val_pairs = load_training_data(config, args.config)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["base_model_name"],
        revision=config["model"].get("base_model_revision"),
        trust_remote_code=bool(config["runtime"]["trust_remote_code"]),
        local_files_only=bool(config["runtime"]["local_files_only"]),
    )
    train_loader, val_loader, train_sampler = make_loaders(
        config, tokenizer, sequences, train_pairs, val_pairs, rank, world
    )

    model = make_model(config).to(device)
    if bool(config["training"].get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model = DDP(
        model,
        device_ids=[device.index],
        broadcast_buffers=False,
        find_unused_parameters=False,
    )

    training = config["training"]
    optimizer = build_optimizer(
        model, float(training["learning_rate"]), float(training["weight_decay"])
    )
    accumulation = int(training["gradient_accumulation_steps"])
    updates_per_epoch = len(train_loader) // accumulation
    max_updates = updates_per_epoch * int(training["max_epochs"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(max_updates * float(training["warmup_ratio"])), max_updates
    )
    scaler = make_grad_scaler(resolved)

    state = {
        "epoch": 0,
        "batch_offset": 0,
        "optimizer_attempts": 0,
        "successful_updates": 0,
        "best_key": None,
        "best_aupr": None,
        "best_epoch": None,
        "epochs_without_improvement": 0,
        "wandb_run_id": None,
    }
    # 3. Restore a previous run when requested and initialize optional logging.
    resume_payload = None
    if args.resume:
        resume_payload = torch.load(
            args.resume, map_location=device, weights_only=False
        )

    wandb_run = None
    if rank == 0 and config["wandb"]["enabled"]:
        import wandb

        wandb_run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", config["wandb"]["project"]),
            group=config["wandb"]["group"],
            job_type=config["wandb"]["job_type"],
            name=args.run_name,
            mode=config["wandb"]["mode"],
            config=config,
            id=(resume_payload or {}).get("state", {}).get("wandb_run_id"),
            resume="allow",
        )
        state["wandb_run_id"] = wandb_run.id

    if resume_payload is not None:
        payload = resume_payload
        if payload["resolved_precision"] != resolved:
            raise RuntimeError("resume precision differs from checkpoint")
        unwrap(model).load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        if scaler is not None:
            if payload["grad_scaler"] is None:
                raise RuntimeError("missing FP16 GradScaler state")
            scaler.load_state_dict(payload["grad_scaler"])
        state = payload["state"]
        states = payload.get("rng_by_rank")
        if not isinstance(states, list) or len(states) != world:
            raise RuntimeError("resume checkpoint lacks one RNG state per rank")
        restore_rng(states[rank])
        if wandb_run is not None:
            state["wandb_run_id"] = wandb_run.id

    metrics_path = run_dir / "metrics.csv"
    if rank == 0 and (not metrics_path.exists()):
        with metrics_path.open("w", newline="") as handle:
            csv.writer(handle).writerow(
                [
                    "epoch",
                    "global_step",
                    "train_weighted_bce",
                    "val_aupr",
                    "val_auroc",
                    "val_weighted_bce",
                    "val_unweighted_nll",
                ]
            )

    # 4. Train one epoch at a time, then evaluate and save its checkpoint.
    stop = False
    start_epoch = int(state["epoch"])
    log_every = int(config["wandb"].get("log_every_updates", 20))
    if log_every <= 0:
        raise ValueError("wandb.log_every_updates must be positive")

    for epoch in range(start_epoch, int(training["max_epochs"])):
        # Reset epoch-level counters and progress reporting.
        torch.cuda.reset_peak_memory_stats(device)
        train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros(4, dtype=torch.float64, device=device)
        log_window = torch.zeros(3, dtype=torch.float64, device=device)
        epoch_started = time.monotonic()
        window_started = epoch_started
        epoch_start_updates = int(state["successful_updates"])
        progress_bar = tqdm(
            train_loader,
            total=len(train_loader),
            disable=rank != 0 or bool(config.get("quiet", False)),
            desc=f"Epoch {epoch + 1:02d}/{int(training['max_epochs']):02d}",
            dynamic_ncols=True,
        )

        for batch_index, raw_batch in enumerate(progress_bar):
            # Accumulate gradients until the configured optimizer boundary.
            batch = move_batch(raw_batch, device)
            boundary = (batch_index + 1) % accumulation == 0
            with accumulation_context(model, boundary):
                with autocast_context(resolved):
                    output = model(batch["input_a"], batch["input_b"], batch["labels"])
                    scaled_loss = output.loss / accumulation
                if scaler is None:
                    scaled_loss.backward()
                else:
                    scaler.scale(scaled_loss).backward()

            # Track training loss and the two input-order predictions.
            count = batch["labels"].numel()
            loss_sum[0] += output.loss.detach().double() * count
            loss_sum[1] += count
            ordered_delta = (
                output.logits_ab.detach().float() - output.logits_ba.detach().float()
            ).abs()
            loss_sum[2] += ordered_delta.double().sum()
            loss_sum[3] = torch.maximum(loss_sum[3], ordered_delta.double().max())
            log_window[0] += output.loss.detach().double() * count
            log_window[1] += count
            log_window[2] += ordered_delta.double().sum()
            if not boundary:
                continue

            # Update model parameters, with GradScaler handling FP16 overflow.
            state["optimizer_attempts"] += 1
            if scaler is None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training["gradient_clip"])
                )
                optimizer.step()
                skipped = False
            else:
                before = scaler.get_scale()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training["gradient_clip"])
                )
                scaler.step(optimizer)
                scaler.update()
                skipped = scaler.get_scale() < before
            optimizer.zero_grad(set_to_none=True)

            # Advance the scheduler and periodically report speed and ETA.
            if not skipped:
                scheduler.step()
                state["successful_updates"] += 1
                updates_this_epoch = (
                    int(state["successful_updates"]) - epoch_start_updates
                )
                should_log = (
                    updates_this_epoch % log_every == 0
                    or updates_this_epoch == updates_per_epoch
                )
                if should_log:
                    reduced_window = log_window.clone()
                    if dist.is_initialized():
                        dist.all_reduce(reduced_window)
                    now = time.monotonic()
                    window_seconds = max(now - window_started, 1e-09)
                    epoch_seconds = max(now - epoch_started, 1e-09)
                    window_loss = float((reduced_window[0] / reduced_window[1]).item())
                    ordered_delta_window = float(
                        (reduced_window[2] / reduced_window[1]).item()
                    )
                    pairs_per_second = float(reduced_window[1].item() / window_seconds)
                    seconds_per_update = epoch_seconds / updates_this_epoch
                    epoch_remaining = max(updates_per_epoch - updates_this_epoch, 0)
                    total_remaining = max(
                        max_updates - int(state["successful_updates"]), 0
                    )
                    if rank == 0:
                        progress = 100.0 * updates_this_epoch / updates_per_epoch
                        message = (
                            f"epoch {epoch + 1}/{int(training['max_epochs'])} "
                            f"update {updates_this_epoch}/{updates_per_epoch} "
                            f"({progress:5.1f}%) "
                            f"global_step={state['successful_updates']} "
                            f"loss={window_loss:.6f} "
                            f"lr={optimizer.param_groups[0]['lr']:.3e} "
                            f"grad_norm={float(grad_norm):.4f} "
                            f"pairs/s={pairs_per_second:.1f} "
                            "epoch_eta="
                            f"{format_duration(epoch_remaining * seconds_per_update)} "
                            "total_eta="
                            f"{format_duration(total_remaining * seconds_per_update)}"
                        )
                        progress_bar.set_postfix_str(
                            f"loss {window_loss:.4f} | "
                            f"lr {optimizer.param_groups[0]['lr']:.2e} | "
                            f"grad {float(grad_norm):.2f} | "
                            f"{pairs_per_second:.1f} pairs/s | "
                            "ETA "
                            f"{format_duration(epoch_remaining * seconds_per_update)}",
                            refresh=True,
                        )
                        log_path = run_dir / "train.log"
                        with log_path.open("a", encoding="utf-8") as log_handle:
                            log_handle.write(message + "\n")
                        if wandb_run:
                            wandb_run.log(
                                {
                                    "train/weighted_bce": window_loss,
                                    "train/learning_rate": optimizer.param_groups[0][
                                        "lr"
                                    ],
                                    "train/grad_norm": float(grad_norm),
                                    "train/pairs_per_second": pairs_per_second,
                                },
                                step=int(state["successful_updates"]),
                            )
                    log_window.zero_()
                    window_started = now

            if (
                args.max_optimizer_steps
                and state["optimizer_attempts"] >= args.max_optimizer_steps
            ):
                stop = True
                break

        # Reduce epoch statistics and evaluate the current model.
        if dist.is_initialized():
            dist.all_reduce(loss_sum[:3])
            dist.all_reduce(loss_sum[3], op=dist.ReduceOp.MAX)
        train_loss = float((loss_sum[0] / loss_sum[1]).item())
        ordered_delta_mean = float((loss_sum[2] / loss_sum[1]).item())
        ordered_delta_max = float(loss_sum[3].item())
        metrics = evaluate_model(
            unwrap(model),
            val_loader,
            device,
            resolved,
            len(val_pairs),
            float(training["pos_weight"]),
            show_progress=rank == 0 and (not bool(config.get("quiet", False))),
        )

        # Save all RNG states so a resumed run follows the same random stream.
        state["epoch"] = epoch + 1
        state["batch_offset"] = 0
        local_rng = rng_state()
        if dist.is_initialized():
            gathered_rng: list[Any] = [None for _ in range(world)]
            dist.all_gather_object(gathered_rng, local_rng)
        else:
            gathered_rng = [local_rng]

        # Rank 0 selects and writes checkpoints and validation metrics.
        if rank == 0 and metrics is not None:
            state["validation_metrics"] = {
                "aupr": metrics.aupr,
                "auroc": metrics.auroc,
                "unweighted_nll": metrics.unweighted_nll,
                "f1_0p5": metrics.f1_0p5,
            }
            candidate_key = (-metrics.aupr, metrics.unweighted_nll, epoch)
            selected = state["best_key"] is None or tuple(candidate_key) < tuple(
                state["best_key"]
            )
            aupr_improved = state["best_aupr"] is None or metrics.aupr > float(
                state["best_aupr"]
            )
            if selected:
                state["best_key"] = candidate_key
                state["best_epoch"] = epoch
            if aupr_improved:
                state["best_aupr"] = metrics.aupr
                state["epochs_without_improvement"] = 0
            else:
                state["epochs_without_improvement"] += 1
            payload = checkpoint_payload(
                model,
                optimizer,
                scheduler,
                scaler,
                state,
                resolved,
                config,
                gathered_rng,
            )
            atomic_save(payload, run_dir / "last.pt")
            if selected:
                atomic_save(payload, run_dir / "best.pt")
            if bool(training.get("save_every_epoch", True)):
                checkpoints = run_dir / "checkpoints"
                checkpoints.mkdir(parents=True, exist_ok=True)
                atomic_save(payload, checkpoints / f"epoch_{epoch + 1:03d}.pt")
            with metrics_path.open("a", newline="") as handle:
                csv.writer(handle).writerow(
                    [
                        epoch,
                        state["successful_updates"],
                        train_loss,
                        metrics.aupr,
                        metrics.auroc,
                        metrics.weighted_bce,
                        metrics.unweighted_nll,
                    ]
                )
            with (run_dir / "train.log").open("a", encoding="utf-8") as log_handle:
                log_handle.write(
                    f"epoch={epoch + 1} "
                    f"train_weighted_bce={train_loss:.6f} "
                    f"validation_aupr={metrics.aupr:.6f} "
                    f"validation_auroc={metrics.auroc:.6f} "
                    f"validation_nll={metrics.unweighted_nll:.6f} "
                    f"validation_f1_0p5={metrics.f1_0p5:.6f} "
                    f"epoch_seconds={time.monotonic() - epoch_started:.1f}\n"
                )
            if wandb_run:
                wandb_run.log(
                    {
                        "validation/aupr": metrics.aupr,
                        "validation/auroc": metrics.auroc,
                        "validation/unweighted_nll": metrics.unweighted_nll,
                        "validation/f1_0p5": metrics.f1_0p5,
                        "runtime/epoch_seconds": time.monotonic() - epoch_started,
                        "runtime/gpu_peak_allocated_gib": torch.cuda.max_memory_allocated(
                            device
                        )
                        / 2**30,
                    },
                    step=int(state["successful_updates"]),
                )
            stop = stop or state["epochs_without_improvement"] >= int(
                training["early_stopping_patience"]
            )

        # All workers receive the same early-stop decision.
        stop_tensor = torch.tensor([int(stop)], device=device)
        if dist.is_initialized():
            dist.broadcast(stop_tensor, 0)
        if bool(stop_tensor.item()):
            break

    # Close the run only after every distributed worker has left the loop.
    if rank == 0:
        (run_dir / "run_status.json").write_text(
            json.dumps({"status": "complete", **state}, indent=2) + "\n"
        )
        if wandb_run:
            wandb_run.finish()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def training_preflight(
    config: Mapping[str, Any], config_path: Path, require_all_gpus: bool = True
) -> dict[str, Any]:
    """Validate public inputs, configured GPUs, and requested precision."""
    validate_training_inputs(config, config_path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    expected = int(config["runtime"]["expected_world_size"])
    visible = str(config["runtime"]["visible_gpus"]).split(",")
    if len(visible) != expected:
        raise ValueError("runtime.visible_gpus count differs from expected_world_size")
    if require_all_gpus and torch.cuda.device_count() != expected:
        raise RuntimeError(
            f"expected exactly {expected} visible GPUs, got {torch.cuda.device_count()}"
        )
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    expected_name = str(config["runtime"].get("expected_gpu_name", ""))
    if (
        require_all_gpus
        and expected_name
        and any((expected_name not in name for name in names))
    ):
        raise RuntimeError(f"visible GPUs are not all B200: {names}")
    precision = resolve_precision(
        str(config["runtime"]["requested_precision"]),
        bool(config["runtime"]["allow_fp16_fallback"]),
    )
    return {"resolved_precision": precision, "gpu_names": names}


def validate_training_inputs(config: Mapping[str, Any], config_path: Path) -> None:
    """Check pair-table structure, labels, and FASTA references."""
    paths = config["paths"]
    fasta_path = resolve_path(config_path, paths["fasta"])
    sequences = read_fasta(fasta_path)
    for role in (
        "train_csv",
        "validation_selection_csv",
        "validation_calibration_csv",
    ):
        pair_path = resolve_path(config_path, paths[role])
        pairs = read_pair_rows(pair_path)
        if any(pair.label not in {0, 1} for pair in pairs):
            raise ValueError(f"labels must be binary: {pair_path}")
        missing = {
            protein_id
            for pair in pairs
            for protein_id in (pair.query, pair.text)
            if protein_id not in sequences
        }
        if missing:
            example = sorted(missing)[0]
            raise KeyError(
                f"FASTA is missing {len(missing)} pair IDs, including {example}"
            )


def resolve_path(config_path: Path, value: str) -> Path:
    """Resolve a config path relative to the directory containing the YAML file."""
    path = Path(value)
    return path if path.is_absolute() else config_path.parent / path


def load_training_data(
    config: Mapping[str, Any], config_path: Path,
) -> tuple[dict[str, str], list[Any], list[Any]]:
    """Load protein sequences and the training and selection pair tables."""
    paths = config["paths"]
    sequences = read_fasta(resolve_path(config_path, paths["fasta"]))
    return (
        sequences,
        read_pair_rows(resolve_path(config_path, paths["train_csv"])),
        read_pair_rows(resolve_path(config_path, paths["validation_selection_csv"])),
    )


def make_loaders(
    config: Mapping[str, Any],
    tokenizer: Any,
    sequences: Mapping[str, str],
    train_pairs: list[Any],
    validation_pairs: list[Any],
    rank: int,
    world: int,
) -> tuple[Any, Any, Any]:
    """Create distributed training and validation data loaders."""
    training = config["training"]
    batch = int(training["batch_size_per_device"])
    accumulation = int(training["gradient_accumulation_steps"])
    collator = PairCollator(tokenizer, int(config["model"]["max_residues"]))
    train_dataset = ProteinPairDataset(train_pairs, sequences)
    val_dataset = ProteinPairDataset(validation_pairs, sequences)
    train_sampler = TruncatedDistributedTrainSampler(
        len(train_dataset),
        batch,
        accumulation,
        int(training["sampler_seed"]),
        world,
        rank,
    )
    val_sampler = DistributedEvalSampler(len(val_dataset), world, rank)
    workers = int(training["num_workers"])
    common = dict(
        batch_size=batch,
        collate_fn=collator,
        num_workers=workers,
        pin_memory=bool(training["pin_memory"]),
        persistent_workers=workers > 0,
    )
    if workers:
        common["prefetch_factor"] = int(training["prefetch_factor"])
    train_loader = DataLoader(
        train_dataset, sampler=train_sampler, drop_last=True, **common
    )
    val_loader = DataLoader(val_dataset, sampler=val_sampler, drop_last=False, **common)
    if len(train_loader) % accumulation:
        raise RuntimeError("train loader does not end at an accumulation boundary")
    return (train_loader, val_loader, train_sampler)


def make_model(config: Mapping[str, Any]) -> ESM2ForPPI:
    """Build the ESM-2 encoder and symmetric Global classification head."""
    model_cfg, training, runtime = (
        config["model"],
        config["training"],
        config["runtime"],
    )
    architecture = ESM2PPIConfig(
        base_model_name=model_cfg["base_model_name"],
        base_model_revision=model_cfg.get("base_model_revision"),
        projection_dim=model_cfg["projection_dim"],
        global_hidden_dim=model_cfg["global_hidden_dim"],
        projection_dropout=model_cfg["projection_dropout"],
        classifier_dropout=model_cfg["classifier_dropout"],
        pos_weight=training["pos_weight"],
        max_residues=model_cfg["max_residues"],
        exclude_special_tokens=model_cfg["exclude_special_tokens"],
        global_symmetry=model_cfg["global_symmetry"],
        loss_definition=model_cfg["loss_definition"],
        model_format=config["model_format"],
        model_id=config.get("model_id"),
        inference_precision=runtime.get(
            "resolved_precision", runtime["requested_precision"]
        ),
    )
    return ESM2ForPPI(architecture)


def build_optimizer(
    model: nn.Module, learning_rate: float, weight_decay: float
) -> AdamW:
    """Create AdamW without weight decay on bias and normalization parameters."""
    decay, no_decay = ([], [])
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lowered = name.lower()
        if (
            parameter.ndim == 1
            or "bias" in lowered
            or "layer_norm" in lowered
            or ("layernorm" in lowered)
        ):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )


def restore_rng(state: Mapping[str, Any]) -> None:
    """Restore Python, NumPy, CPU, and per-GPU random-number states."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    cpu_rng = state["torch"].detach().to(device="cpu", dtype=torch.uint8).contiguous()
    cuda_rng = [
        value.detach().to(device="cpu", dtype=torch.uint8).contiguous()
        for value in state["cuda"]
    ]
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state_all(cuda_rng)


def format_duration(seconds: float) -> str:
    """Format a non-negative duration as a compact, human-readable ETA."""
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    precision: str,
    expected_size: int,
    pos_weight: float,
    show_progress: bool = False,
) -> ValidationMetrics | None:
    """Gather distributed predictions and calculate validation metrics on rank 0."""
    model.eval()
    indices, labels, logits = ([], [], [])
    batches = loader
    if show_progress:
        from tqdm.auto import tqdm

        batches = tqdm(loader, desc="Validation", dynamic_ncols=True)
    for batch in batches:
        batch = move_batch(batch, device)
        with autocast_context(precision):
            output = model(batch["input_a"], batch["input_b"])
        indices.append(batch["dataset_indices"])
        labels.append(batch["labels"].float())
        logits.append(output.logits.float())
    local_indices = (
        torch.cat(indices)
        if indices
        else torch.empty(0, device=device, dtype=torch.long)
    )
    local_labels = torch.cat(labels) if labels else torch.empty(0, device=device)
    local_logits = torch.cat(logits) if logits else torch.empty(0, device=device)
    all_indices = gather_variable(local_indices).cpu().numpy()
    all_labels = gather_variable(local_labels).cpu().numpy()
    all_logits = gather_variable(local_logits).cpu().numpy()
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank != 0:
        return None
    if (
        len(all_indices) != expected_size
        or len(np.unique(all_indices)) != expected_size
        or (not np.array_equal(np.sort(all_indices), np.arange(expected_size)))
    ):
        raise RuntimeError("validation indices contain duplicates or omissions")
    order = np.argsort(all_indices, kind="stable")
    y = torch.from_numpy(all_labels[order]).float()
    z = torch.from_numpy(all_logits[order]).float()
    weighted = F.binary_cross_entropy_with_logits(
        z, y, pos_weight=torch.tensor([pos_weight])
    ).item()
    nll = F.binary_cross_entropy_with_logits(z, y).item()
    return ValidationMetrics(
        aupr=float(average_precision_score(y.numpy(), z.numpy())),
        auroc=float(roc_auc_score(y.numpy(), z.numpy())),
        weighted_bce=float(weighted),
        unweighted_nll=float(nll),
        f1_0p5=float(
            f1_score(y.numpy(), (torch.sigmoid(z) >= 0.5).numpy(), zero_division=0)
        ),
        count=len(y),
    )


def rng_state() -> dict[str, Any]:
    """Collect random-number states needed for an exact training resume."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def checkpoint_payload(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    state: Mapping[str, Any],
    precision: str,
    config: Mapping[str, Any],
    rank_rng_states: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Collect model, optimizer, scheduler, scaler, and run state in a checkpoint."""
    return {
        "format": "esm2_ppi_training_v1",
        "model": unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "grad_scaler": scaler.state_dict() if scaler is not None else None,
        "grad_scaler_enabled": scaler is not None,
        "resolved_precision": precision,
        "state": dict(state),
        "rng_by_rank": rank_rng_states,
        "config": dict(config),
    }


def atomic_save(payload: Any, path: Path) -> None:
    """Write a checkpoint to a temporary file before replacing its destination."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def score_and_calibrate(
    config: Mapping[str, Any], run_dir: Path, config_path: Path
) -> None:
    """Score cached pairs, fit both calibrators, and write postprocess metrics."""
    # Load the shared residue cache and the trained Global head.
    device, rank, world = setup_distributed(timeout_seconds=6 * 3600)
    started = time.monotonic()
    cache = ProjectionCache(run_dir / "cache")
    pooled = {
        protein_id: torch.from_numpy(cache.get(protein_id).astype(np.float32).mean(0))
        for protein_id in cache.entries
    }
    payload = torch.load(
        run_dir / "best.pt", map_location="cpu", weights_only=False, mmap=True
    )
    head = GlobalHead.from_step1_state(payload["model"]).to(device)

    # Read the three roles used for statistics, evaluation, and calibration.
    paths = config["paths"]
    roles = {
        "train": read_pair_rows(resolve_path(config_path, paths["train_csv"])),
        "selection": read_pair_rows(resolve_path(config_path, paths["validation_selection_csv"])),
        "calibration": read_pair_rows(
            resolve_path(config_path, paths["validation_calibration_csv"])
        ),
    }

    # Each worker scores its non-overlapping share of every role.
    results = {
        name: score_role(rows, cache, pooled, head, config, device, rank, world, name)
        for name, rows in roles.items()
    }

    if rank == 0:
        # Save the gathered scores before fitting any calibration parameters.
        out = run_dir / "maxsim"
        out.mkdir(parents=True, exist_ok=True)
        for name, arrays in results.items():
            assert arrays is not None
            target = out / f"{name}_scores.npz"
            temporary = target.with_suffix(target.suffix + ".tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **arrays)
            os.replace(temporary, target)

        # Measure uncalibrated Maxsim on the checkpoint-selection split.
        selection = results["selection"]
        assert selection is not None
        maxsim_metrics = {
            "aupr": float(
                average_precision_score(selection["labels"], selection["maxsim"])
            ),
            "auroc": float(roc_auc_score(selection["labels"], selection["maxsim"])),
        }
        (out / "selection_metrics.json").write_text(
            json.dumps(maxsim_metrics, indent=2) + "\n"
        )

        # Fit with train statistics and the designated calibration split only.
        train, calibration = results["train"], results["calibration"]
        assert train is not None and calibration is not None
        state, fit_diagnostics = fit_calibration(
            train["maxsim"],
            calibration["global"],
            calibration["maxsim"],
            calibration["labels"],
            str(config["model_format"]),
        )
        calibration_dir = run_dir / "calibration"
        calibration_dir.mkdir(parents=True, exist_ok=True)
        state_path = calibration_dir / "scoring_state.json"
        state_path.write_text(json.dumps(state, indent=2) + "\n")

        # Evaluate the fitted state on the independent selection split.
        applied = apply_calibration(selection["global"], selection["maxsim"], state)
        diagnostics = {
            "fit": fit_diagnostics,
            "global": probability_metrics(selection["labels"], applied["global_logit"]),
            "maxsim": probability_metrics(selection["labels"], applied["maxsim_logit"]),
            "final": probability_metrics(selection["labels"], applied["final_logit"]),
            "fit_counts": {
                "train": int(len(train["maxsim"])),
                "calibration": int(len(calibration["labels"])),
                "evaluation": int(len(selection["labels"])),
            },
        }
        (calibration_dir / "diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2) + "\n"
        )

        # Keep a compact summary for run tracking and release checks.
        summary = {
            "runtime/maxsim_scoring_seconds": time.monotonic() - started,
            "maxsim/selection_aupr": maxsim_metrics["aupr"],
            "maxsim/selection_auroc": maxsim_metrics["auroc"],
            "calibration/maxsim_nll": fit_diagnostics["maxsim"]["nll"],
            "calibration/global_nll": fit_diagnostics["global"]["nll"],
            "final/nll": diagnostics["final"]["nll"],
        }
        (run_dir / "postprocess_metrics.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def read_pair_rows(path: Path) -> list[ProteinPair]:
    """Read a non-empty pair CSV whose columns are exactly query,text,label."""
    result: list[ProteinPair] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["query", "text", "label"]:
            raise ValueError(f"pair columns must be query,text,label: {path}")
        for index, row in enumerate(reader):
            result.append(
                ProteinPair(
                    index,
                    row["query"],
                    row["text"],
                    int(row["label"]),
                )
            )
    if not result:
        raise ValueError(f"pair file is empty: {path}")
    return result


@torch.inference_mode()
def score_role(
    pairs: list[ProteinPair],
    cache: ProjectionCache,
    pooled: Mapping[str, torch.Tensor],
    head: GlobalHead,
    config: Mapping[str, Any],
    device: torch.device,
    rank: int,
    world: int,
    role: str,
) -> dict[str, np.ndarray] | None:
    """Calculate distributed Global and Maxsim scores for one data split."""
    # Assign each pair to exactly one distributed worker.
    sampler = DistributedEvalSampler(len(pairs), world, rank)
    local_indices = list(iter(sampler))
    precision = resolve_precision(
        config["runtime"]["requested_precision"],
        config["runtime"]["allow_fp16_fallback"],
    )

    # Global scoring uses pooled FP32 representations and its larger batch size.
    global_parts = []
    global_batch = int(config["postprocess"]["global_batch_size"])
    for start in range(0, len(local_indices), global_batch):
        indices = local_indices[start : start + global_batch]
        pooled_a = torch.stack([pooled[pairs[index].query] for index in indices]).to(
            device, non_blocking=True
        )
        pooled_b = torch.stack([pooled[pairs[index].text] for index in indices]).to(
            device, non_blocking=True
        )
        with autocast_context(precision):
            logits = 0.5 * (
                head.ordered(pooled_a, pooled_b) + head.ordered(pooled_b, pooled_a)
            )
        global_parts.append(logits.float())

    # Maxsim uses full residue representations and a separate, smaller batch size.
    loader = DataLoader(
        CachedPairDataset(pairs),
        batch_size=int(config["postprocess"]["maxsim_pair_batch_size"]),
        sampler=sampler,
        collate_fn=CachedPairCollator(cache),
        num_workers=int(config["training"]["num_workers"]),
        pin_memory=True,
        drop_last=False,
    )
    chunks: dict[str, list[torch.Tensor]] = {
        key: [] for key in ("indices", "labels", "maxsim")
    }
    bar = tqdm(
        loader,
        desc=f"Maxsim scoring ({role})",
        disable=rank != 0 or bool(config.get("quiet", False)),
        dynamic_ncols=True,
    )
    for batch in bar:
        a = batch["embeddings_a"].to(device, non_blocking=True)
        b = batch["embeddings_b"].to(device, non_blocking=True)
        ma = batch["mask_a"].to(device, non_blocking=True)
        mb = batch["mask_b"].to(device, non_blocking=True)
        maxsim_score, _, _ = symmetric_maxsim(a, ma, b, mb)
        chunks["indices"].append(batch["dataset_indices"].to(device))
        chunks["labels"].append(batch["labels"].to(device))
        chunks["maxsim"].append(maxsim_score)

    # Gather both branches and restore the exact input-row order on rank 0.
    chunks["global"] = global_parts
    gathered: dict[str, np.ndarray] = {}
    for key, parts in chunks.items():
        dtype = torch.long if key == "indices" else torch.float32
        local = (
            torch.cat(parts) if parts else torch.empty(0, device=device, dtype=dtype)
        )
        gathered[key] = gather_variable(local).cpu().numpy()
    if rank != 0:
        return None
    order = np.argsort(gathered["indices"], kind="stable")
    if not np.array_equal(gathered["indices"][order], np.arange(len(pairs))):
        raise RuntimeError(f"incomplete distributed score coverage: {role}")
    return {key: value[order] for key, value in gathered.items()}


# Fit only on the designated validation roles; never on test pairs.
def fit_calibration(
    train_maxsim: Any,
    calibration_global: Any,
    calibration_maxsim: Any,
    calibration_labels: Any,
    model_format: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit the two branch calibrators used by either released model."""
    # Validate one-dimensional inputs and derive Maxsim train statistics.
    train_scores = _vector(train_maxsim, "train_maxsim")
    global_scores = _vector(calibration_global, "calibration_global")
    maxsim_scores = _vector(calibration_maxsim, "calibration_maxsim")
    labels = _vector(calibration_labels, "calibration_labels")
    if not global_scores.size == maxsim_scores.size == labels.size:
        raise ValueError("calibration array lengths differ")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be binary")

    train_mean = float(train_scores.mean())
    train_std = float(train_scores.std(ddof=0))
    if train_std <= 0:
        raise ValueError("train Maxsim standard deviation must be positive")
    standardized_maxsim = (maxsim_scores - train_mean) / train_std

    # Fit temperature scaling and bias for the Global branch.
    global_result = minimize(
        lambda parameters: nll(
            labels,
            global_scores / math.exp(float(parameters[0])) + float(parameters[1]),
        ),
        np.asarray([0.0, 0.0]),
        method="L-BFGS-B",
        bounds=[(-10.0, 10.0), (None, None)],
    )
    log_temperature, global_bias = map(float, global_result.x)
    temperature = math.exp(log_temperature)

    # Fit scale and bias for standardized Maxsim scores.
    maxsim_result = minimize(
        lambda parameters: nll(
            labels,
            math.exp(float(parameters[0])) * standardized_maxsim
            + float(parameters[1]),
        ),
        np.asarray([0.0, 0.0]),
        method="L-BFGS-B",
        bounds=[(-10.0, 10.0), (None, None)],
    )
    log_scale, maxsim_bias = map(float, maxsim_result.x)
    scale = math.exp(log_scale)
    if not global_result.success or not maxsim_result.success:
        raise RuntimeError(
            "calibration failed: "
            f"global={global_result.message}; maxsim={maxsim_result.message}"
        )

    # Both model formats share this state schema and fixed 0.5:0.5 blend.
    state = {
        "format": model_format,
        "maxsim_train": {"mean": train_mean, "std": train_std},
        "global": {"temperature": temperature, "bias": global_bias},
        "maxsim": {"scale": scale, "bias": maxsim_bias},
        "blend": {"global_weight": 0.5, "maxsim_weight": 0.5},
    }
    diagnostics = {
        "global": {
            "nll": nll(labels, global_scores / temperature + global_bias),
            "success": True,
            "message": str(global_result.message),
        },
        "maxsim": {
            "nll": nll(labels, scale * standardized_maxsim + maxsim_bias),
            "success": True,
            "message": str(maxsim_result.message),
        },
    }
    validate_calibration(state)
    return state, diagnostics


def launch_worker(
    args: argparse.Namespace, worker: str, config_path: Path, world: int
) -> None:
    """Launch distributed training or cached scoring workers with torchrun."""
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={world}",
        "--module",
        "scripts.training",
        "--_worker",
        worker,
        "--config",
        str(config_path),
        "--output_dir",
        str(args.output_dir),
    ]
    if args.resume:
        command += ["--resume", str(args.resume)]
    if args.max_optimizer_steps:
        command += ["--max_optimizer_steps", str(args.max_optimizer_steps)]
    subprocess.run(command, check=True, cwd=ROOT)


def build_residue_cache(
    config: Mapping[str, Any], run_dir: Path, config_path: Path, quiet: bool = False
) -> float:
    """Create the exact Step-2 FP16 residue-only cache on one GPU."""
    # Load the selected model and all sequences on the first visible GPU.
    started = time.monotonic()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    model, payload = load_best(run_dir, device)
    tokenizer = AutoTokenizer.from_pretrained(
        model.config.base_model_name,
        revision=model.config.base_model_revision,
        local_files_only=bool(config["runtime"].get("local_files_only", False)),
    )
    fasta_path = resolve_path(config_path, config["paths"]["fasta"])
    sequences = read_fasta(fasta_path)

    # Allocate one contiguous FP16 file for every residue representation.
    output = run_dir / "cache"
    output.mkdir(parents=True, exist_ok=True)
    ordered = list(sequences.items())
    total = sum((min(len(sequence), 800) for _, sequence in ordered))
    values = np.memmap(
        output / "residue_projection.fp16.tmp",
        mode="w+",
        dtype=np.float16,
        shape=(total, 512),
    )
    precision = resolve_precision(
        config["runtime"]["requested_precision"],
        config["runtime"]["allow_fp16_fallback"],
    )
    batch_size = int(config["postprocess"]["embedding_batch_size"])
    offset = 0

    # Encode each protein once and record its offset in the contiguous file.
    with (output / "protein_index.csv.tmp").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["protein_id", "offset", "length"])
        writer.writeheader()
        bar = tqdm(
            range(0, len(ordered), batch_size),
            desc="Embedding proteins",
            disable=quiet,
            dynamic_ncols=True,
        )
        for start in bar:
            rows = ordered[start : start + batch_size]
            encoded = tokenizer(
                [sequence[:800] for _, sequence in rows],
                padding=True,
                truncation=True,
                max_length=802,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            batch = {
                key: encoded[key].to(device)
                for key in ("input_ids", "attention_mask", "special_tokens_mask")
            }
            with torch.inference_mode(), autocast_context(precision):
                projected, mask = model.encode(batch)
            for local, (protein_id, sequence) in enumerate(rows):
                residue = (
                    projected[local][mask[local]]
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np.float16)
                )
                length = min(len(sequence), 800)
                if residue.shape != (length, 512):
                    raise RuntimeError(f"residue shape mismatch: {protein_id}")
                values[offset : offset + length] = residue
                writer.writerow(
                    {"protein_id": protein_id, "offset": offset, "length": length}
                )
                offset += length
            bar.set_postfix_str(
                f"{min(start + len(rows), len(ordered))}/{len(ordered)} proteins"
            )

    # Atomically publish the completed cache and its layout metadata.
    values.flush()
    del values
    os.replace(
        output / "residue_projection.fp16.tmp", output / "residue_projection.fp16"
    )
    os.replace(output / "protein_index.csv.tmp", output / "protein_index.csv")
    write_cache_manifest(
        output,
        dimension=512,
        total_residues=total,
        protein_count=len(ordered),
    )
    return time.monotonic() - started


def load_best(
    run_dir: Path, device: torch.device
) -> tuple[ESM2ForPPI, dict[str, Any]]:
    """Load the selected checkpoint on the requested device in evaluation mode."""
    payload = torch.load(
        run_dir / "best.pt", map_location="cpu", weights_only=False, mmap=True
    )
    model = make_model(payload["config"])
    model.load_state_dict(payload["model"], strict=True)
    return (model.eval().requires_grad_(False).to(device), payload)


def export_bundle(config: Mapping[str, Any], run_dir: Path) -> Path:
    """Export a reloadable bundle and verify every saved model tensor."""
    # Build the release in a temporary directory so partial bundles are hidden.
    final_dir = run_dir / "final_model"
    temporary_dir = run_dir / "final_model.tmp"
    if final_dir.exists():
        raise FileExistsError(f"final model already exists: {final_dir}")
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir(parents=True)
    model, payload = load_best(run_dir, torch.device("cpu"))
    tokenizer = AutoTokenizer.from_pretrained(
        model.config.base_model_name,
        revision=model.config.base_model_revision,
        local_files_only=bool(config["runtime"].get("local_files_only", False)),
    )
    model.save_pretrained(temporary_dir, safe_serialization=True)
    tokenizer.save_pretrained(temporary_dir)

    # Add calibration and standalone model loading code.
    state_source = run_dir / "calibration" / "scoring_state.json"
    state_target = temporary_dir / "scoring_state.json"
    state_target.write_bytes(state_source.read_bytes())
    write_model_code(temporary_dir)

    # Reload the exported model and require exact tensor equality.
    reloaded = ESM2ForPPI.from_pretrained(temporary_dir).eval()
    for name, value in model.state_dict().items():
        if not torch.equal(value, reloaded.state_dict()[name]):
            raise RuntimeError(f"bundle tensor mismatch: {name}")
    os.replace(temporary_dir, final_dir)
    return final_dir


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML configuration and validate the locked public pipeline settings."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("model_format") not in {"x-species", "bernett"}:
        raise ValueError("model_format must be x-species or bernett")
    expected_paths = {
        "fasta",
        "train_csv",
        "validation_selection_csv",
        "validation_calibration_csv",
    }
    if set(config.get("paths", {})) != expected_paths:
        raise ValueError(f"paths must contain exactly: {sorted(expected_paths)}")
    batch = int(config["training"]["batch_size_per_device"])
    accumulation = int(config["training"]["gradient_accumulation_steps"])
    world = int(config["runtime"]["expected_world_size"])
    effective = batch * accumulation * world
    if effective != int(config["training"]["target_effective_batch"]):
        raise ValueError(
            f"effective batch {effective} does not equal configured target"
        )
    if int(config["model"]["max_residues"]) != 800:
        raise ValueError("locked profile requires max_residues=800")
    postprocess = config["postprocess"]
    if str(postprocess.get("cache_dtype")) != "float16":
        raise ValueError("cache_dtype must be float16")
    if int(postprocess["global_batch_size"]) != 4096:
        raise ValueError("global_batch_size must be 4096")
    config["training"]["save_every_epoch"] = True
    return config


def write_model_code(folder: Path) -> None:
    """Copy model code and add Hugging Face AutoClass loading metadata."""
    for name in ("config.py", "esm2_global.py"):
        shutil.copy2(ROOT / "logobert" / name, folder / name)
    config_path = folder / "config.json"
    config = json.loads(config_path.read_text())
    config["auto_map"] = {
        "AutoConfig": "config.ESM2PPIConfig",
        "AutoModelForSequenceClassification": "esm2_global.ESM2ForPPI",
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--max_optimizer_steps", type=int, help=argparse.SUPPRESS
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--_worker", choices=("global", "score"), help=argparse.SUPPRESS
    )
    main(parser.parse_args())
