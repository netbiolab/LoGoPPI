"""Small helpers shared by training and distributed evaluation.

These functions centralize GPU setup, numeric precision, gradient
accumulation, distributed gathering, and batch transfers. They do not contain
model-specific calculations.
"""

from __future__ import annotations

import contextlib
import datetime
import os
import random
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed(timeout_seconds: int = 3600) -> tuple[torch.device, int, int]:
    """Initialize NCCL when launched by torchrun and select the local GPU."""
    if "RANK" in os.environ and (not dist.is_initialized()):
        dist.init_process_group(
            "nccl", timeout=datetime.timedelta(seconds=timeout_seconds)
        )
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training")
    torch.cuda.set_device(local_rank)
    return (torch.device("cuda", local_rank), rank, world)


def seed_everything(seed: int, rank: int = 0) -> None:
    """Seed Python, NumPy, and PyTorch, using a distinct seed on each rank."""
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def resolve_precision(requested: str, allow_fp16_fallback: bool) -> str:
    """Choose BF16 or FP16 according to the config and GPU capability."""
    requested = requested.lower()
    if requested == "fp16":
        return "fp16"
    if requested != "bf16":
        raise ValueError("requested_precision must be bf16 or fp16")
    if torch.cuda.is_bf16_supported():
        return "bf16"
    if allow_fp16_fallback:
        return "fp16"
    raise RuntimeError("BF16 is unavailable and FP16 fallback is disabled")


def autocast_context(precision: str) -> Any:
    """Return the CUDA autocast context for the resolved precision."""
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def make_grad_scaler(precision: str) -> Any | None:
    """Create a GradScaler for FP16; BF16 training does not require one."""
    if precision == "bf16":
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()


def accumulation_context(model: nn.Module, boundary: bool) -> Any:
    """Skip DDP gradient synchronization before an optimizer boundary."""
    return (
        model.no_sync()
        if isinstance(model, DDP) and (not boundary)
        else contextlib.nullcontext()
    )


def gather_variable(tensor: Tensor) -> Tensor:
    """All-gather a first-dimension-variable tensor, including empty shards."""
    if not dist.is_initialized():
        return tensor
    local_length = torch.tensor(
        [tensor.shape[0]], device=tensor.device, dtype=torch.long
    )
    lengths = [torch.zeros_like(local_length) for _ in range(dist.get_world_size())]
    dist.all_gather(lengths, local_length)
    maximum = max((int(value.item()) for value in lengths))
    shape = (maximum, *tensor.shape[1:])
    padded = torch.zeros(shape, device=tensor.device, dtype=tensor.dtype)
    if tensor.shape[0]:
        padded[: tensor.shape[0]] = tensor
    gathered = [torch.zeros_like(padded) for _ in lengths]
    dist.all_gather(gathered, padded)
    return torch.cat(
        [value[: int(length.item())] for value, length in zip(gathered, lengths)], 0
    )


def unwrap(model: nn.Module) -> nn.Module:
    """Return the underlying model when it is wrapped by DDP."""
    return model.module if isinstance(model, DDP) else model


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensors, including nested tokenizer tensors, to one device."""
    result: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, Mapping):
            result[key] = {
                name: tensor.to(device, non_blocking=True)
                for name, tensor in value.items()
            }
        elif isinstance(value, Tensor):
            result[key] = value.to(device, non_blocking=True)
        else:
            result[key] = value
    return result
