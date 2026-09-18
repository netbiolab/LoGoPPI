"""Load a released model, embed proteins, and score protein pairs.

Inference has two main phases. ``FinalPredictor.encode`` embeds each unique
protein once. ``FinalPredictor.score_arrays`` then reuses those embeddings for
Global and Maxsim scoring before applying the saved calibration state.

The loops in this module are performance sensitive. Their operation order and
CPU/GPU transfers match the validated B200 inference path.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from logobert.esm2_global import ESM2ForPPI, GlobalHead
from logobert.maxsim import symmetric_maxsim
from logobert.scoring import _vector, apply_calibration, nll, validate_calibration
from utils.execution import autocast_context


def probability_metrics(labels: Any, logits: Any, bins: int = 15) -> dict[str, float]:
    """Calculate discrimination and calibration metrics from binary logits."""
    labels_array = _vector(labels, "labels")
    logits_array = _vector(logits, "logits")
    probabilities = expit(logits_array)
    predicted_labels = (probabilities >= 0.5).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_array.astype(np.int64),
        predicted_labels,
        average="binary",
        zero_division=0,
    )
    bin_edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for index in range(bins):
        in_bin = (probabilities >= bin_edges[index]) & (
            probabilities < bin_edges[index + 1]
            if index < bins - 1
            else probabilities <= 1
        )
        if in_bin.any():
            ece += float(in_bin.mean()) * abs(
                float(probabilities[in_bin].mean())
                - float(labels_array[in_bin].mean())
            )
    return {
        "aupr": float(average_precision_score(labels_array, logits_array)),
        "auroc": float(roc_auc_score(labels_array, logits_array)),
        "nll": nll(labels_array, logits_array),
        "brier": float(brier_score_loss(labels_array, probabilities)),
        "ece": ece,
        "precision_0p5": float(precision),
        "recall_0p5": float(recall),
        "f1_0p5": float(f1),
    }


def read_fasta_any(path: Path) -> dict[str, str]:
    """Read a FASTA file without assuming a species-specific ID format."""
    records: dict[str, str] = {}
    identifier: str | None = None
    chunks: list[str] = []

    def commit() -> None:
        if identifier is None:
            return
        sequence = "".join(chunks).replace(" ", "").upper()
        if not sequence or identifier in records:
            raise ValueError(f"empty or duplicate FASTA ID: {identifier}")
        records[identifier] = sequence

    with path.open(encoding="utf-8-sig") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                commit()
                identifier = line[1:].split(maxsplit=1)[0]
                chunks = []
            elif identifier is None:
                raise ValueError("FASTA sequence appears before a header")
            else:
                chunks.append(line)
    commit()
    if not records:
        raise ValueError("FASTA is empty")
    return records


class FinalPredictor:
    """Released LoGoPPI model with embedding, scoring, and calibration state."""

    def __init__(self, model_dir: Path, device: torch.device) -> None:
        self.model_dir = model_dir
        self.device = device
        self.model = (
            ESM2ForPPI.from_pretrained(model_dir)
            .eval()
            .requires_grad_(False)
            .to(device)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.head = GlobalHead.from_step1_state(self.model.state_dict()).to(device)
        self.state = json.loads((model_dir / "scoring_state.json").read_text())
        validate_calibration(self.state)
        self.model_format = self.model.config.model_format
        self.model_id = self.model.config.model_id
        self.precision = str(self.model.config.inference_precision)
        if self.model_format not in {"x-species", "bernett"}:
            raise ValueError("model config must identify x-species or bernett")
        if not self.model_id:
            raise ValueError("model config is missing model_id")
        if self.state["format"] != self.model_format:
            raise ValueError("model config and scoring state formats differ")

    @torch.inference_mode()
    def encode(
        self, sequences: Mapping[str, str], batch_size: int, quiet: bool = False
    ) -> dict[str, torch.Tensor]:
        """Encode each protein once and return residue-level FP16 CPU tensors."""
        # Similar lengths are encoded together to reduce padding work.
        ordered = sorted(
            sequences.items(), key=lambda item: (-min(len(item[1]), 800), item[0])
        )
        embeddings: dict[str, torch.Tensor] = {}
        bar = tqdm(
            range(0, len(ordered), batch_size),
            desc="Embedding proteins",
            disable=quiet,
            dynamic_ncols=True,
        )

        for start in bar:
            rows = ordered[start : start + batch_size]
            encoded = self.tokenizer(
                [seq[:800] for _, seq in rows],
                padding=True,
                truncation=True,
                max_length=802,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            batch = {
                key: encoded[key].to(self.device)
                for key in ("input_ids", "attention_mask", "special_tokens_mask")
            }
            with autocast_context(self.precision):
                projected, mask = self.model.encode(batch)

            # Special tokens are excluded by the mask before caching.
            for index, (protein_id, _) in enumerate(rows):
                embeddings[protein_id] = (
                    projected[index][mask[index]]
                    .float()
                    .to("cpu", torch.float16)
                    .contiguous()
                )
        return embeddings

    @torch.inference_mode()
    def score_arrays(
        self,
        pair_ids: list[tuple[str, str]],
        embeddings: Mapping[str, torch.Tensor],
        batch_size: int,
        quiet: bool = False,
        *,
        global_batch_size: int = 4096,
        length_bucketing: bool = True,
        timing: dict[str, float] | None = None,
    ) -> dict[str, np.ndarray]:
        """Score pairs with the B200 benchmark execution layout.

        Global pooling is performed once per protein, Global and Maxsim use
        independent batch sizes, and device results are copied to the CPU once.
        Internal length sorting never changes the returned input order.
        """
        if batch_size <= 0 or global_batch_size <= 0:
            raise ValueError("pair and global batch sizes must be positive")
        count = len(pair_ids)
        if not count:
            return {
                key: np.empty(0, dtype=np.float64)
                for key in (
                    "global_score",
                    "maxsim_score",
                    "global_logit",
                    "maxsim_logit",
                    "final_logit",
                    "score",
                )
            }

        # CUDA synchronization is used only when benchmark timing is requested.
        def sync() -> None:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

        def start_stage() -> float:
            sync()
            return time.perf_counter()

        def end_stage(name: str, started: float) -> None:
            sync()
            if timing is not None:
                timing[name] = timing.get(name, 0.0) + time.perf_counter() - started

        # Convert a mapping cache once, validate IDs, and pool every protein once.
        stage_started = start_stage()
        if isinstance(embeddings, ContiguousEmbeddingCache):
            cache = embeddings
        else:
            cache = ContiguousEmbeddingCache.from_mapping(
                embeddings, embeddings[next(iter(embeddings))].device
            )
        missing = {pid for pair in pair_ids for pid in pair}.difference(cache.entries)
        if missing:
            raise KeyError(f"embedding cache misses protein: {min(missing)}")
        pooled = cache.pooled(self.device, fp32=True)
        end_stage("pooling_sec", stage_started)

        # Translate string pair IDs into positions in the contiguous cache.
        stage_started = start_stage()
        left_indices = [cache.id_to_index[q] for q, _ in pair_ids]
        right_indices = [cache.id_to_index[t] for _, t in pair_ids]
        left = torch.tensor(left_indices, dtype=torch.long)
        right = torch.tensor(right_indices, dtype=torch.long)
        end_stage("pair_index_sec", stage_started)

        # Global scoring uses pooled representations and its own large batch size.
        global_logits = torch.empty(count, dtype=torch.float32, device=self.device)
        stage_started = start_stage()
        global_bar = tqdm(
            range(0, count, global_batch_size),
            desc="Global scoring",
            disable=quiet,
            dynamic_ncols=True,
        )
        for start in global_bar:
            stop = min(start + global_batch_size, count)
            li = left[start:stop].to(self.device, non_blocking=True)
            ri = right[start:stop].to(self.device, non_blocking=True)
            with autocast_context(self.precision):
                pa, pb = (pooled[li], pooled[ri])
                logits = 0.5 * (self.head.ordered(pa, pb) + self.head.ordered(pb, pa))
            global_logits[start:stop] = logits.float()
        end_stage("global_sec", stage_started)

        # Detect complete-graph order before deciding whether bucketing is useful.
        stage_started = start_stage()
        use_bucketing = False
        if length_bucketing and count > 1:
            sample = min(count - 1, 4096)
            stride = max(1, (count - 1) // sample)
            compared = range(0, count - 1, stride)
            same_query = sum(
                (pair_ids[index][0] == pair_ids[index + 1][0] for index in compared)
            )
            use_bucketing = same_query / max(1, len(compared)) < 0.25
        order: list[int] | range = range(count)
        if use_bucketing:

            def length_key(index: int) -> tuple[int, int, int, int]:
                lq = cache.lengths[left_indices[index]]
                lt = cache.lengths[right_indices[index]]
                short, long = sorted((lq, lt))
                return (short // 64, long // 64, short, long)

            order = sorted(order, key=length_key)
        if timing is not None:
            timing["length_bucketing_used"] = float(use_bucketing)

        # Maxsim uses full residue tensors and restores the original pair order.
        maxsim_scores = torch.empty(count, dtype=torch.float32, device=self.device)
        maxsim_bar = tqdm(
            range(0, count, batch_size),
            desc="Maxsim",
            disable=quiet,
            dynamic_ncols=True,
        )
        for start in maxsim_bar:
            stop = min(start + batch_size, count)
            positions = order[start:stop]
            a_indices = [left_indices[index] for index in positions]
            b_indices = [right_indices[index] for index in positions]
            a_rows = [cache.by_index(index) for index in a_indices]
            b_rows = [cache.by_index(index) for index in b_indices]
            a = pad_sequence(a_rows, batch_first=True).to(
                self.device, non_blocking=True
            )
            b = pad_sequence(b_rows, batch_first=True).to(
                self.device, non_blocking=True
            )
            la = torch.tensor(
                [cache.lengths[index] for index in a_indices], device=self.device
            )
            lb = torch.tensor(
                [cache.lengths[index] for index in b_indices], device=self.device
            )
            mask_grid = cache.mask_grid(self.device)
            ma = mask_grid[: a.shape[1]][None, :] < la[:, None]
            mb = mask_grid[: b.shape[1]][None, :] < lb[:, None]
            maxsim_score, _, _ = symmetric_maxsim(a, ma, b, mb)
            if use_bucketing:
                destination = torch.tensor(
                    positions, dtype=torch.long, device=self.device
                )
                maxsim_scores[destination] = maxsim_score
            else:
                maxsim_scores[start:stop] = maxsim_score
        end_stage("maxsim_sec", stage_started)

        # Copy each branch to CPU once, then apply the saved calibration state.
        stage_started = start_stage()
        global_np = global_logits.cpu().numpy()
        maxsim_np = maxsim_scores.cpu().numpy()
        applied = apply_calibration(global_np, maxsim_np, self.state)
        results = {
            "global_score": global_np,
            "maxsim_score": maxsim_np,
            "global_logit": applied["global_logit"],
            "maxsim_logit": applied["maxsim_logit"],
            "final_logit": applied["final_logit"],
            "score": applied["score"],
        }
        end_stage("calibration_sec", stage_started)
        return results

    def score(
        self,
        pair_ids: list[tuple[str, str]],
        embeddings: Mapping[str, torch.Tensor],
        batch_size: int,
        quiet: bool = False,
        *,
        global_batch_size: int = 4096,
        length_bucketing: bool = True,
        timing: dict[str, float] | None = None,
    ) -> list[dict[str, float]]:
        """Return the same scores as dictionaries, one dictionary per input pair."""
        arrays = self.score_arrays(
            pair_ids,
            embeddings,
            batch_size,
            quiet,
            global_batch_size=global_batch_size,
            length_bucketing=length_bucketing,
            timing=timing,
        )
        return [
            {key: float(value[index]) for key, value in arrays.items()}
            for index in range(len(pair_ids))
        ]


class ContiguousEmbeddingCache(Mapping[str, torch.Tensor]):
    """Store all residue embeddings in one compact FP16 tensor.

    ``entries`` maps each protein ID to an ``(offset, length)`` slice. Keeping
    one tensor avoids thousands of separate GPU allocations during inference.
    """

    def __init__(
        self, values: torch.Tensor, entries: Mapping[str, tuple[int, int]]
    ) -> None:
        if values.ndim != 2 or values.dtype != torch.float16:
            raise ValueError("contiguous embedding values must be a 2D FP16 tensor")
        self.values = values
        self.entries = dict(entries)
        self.protein_ids = sorted(self.entries)
        self.id_to_index = {
            protein_id: index for index, protein_id in enumerate(self.protein_ids)
        }
        self.offsets = [self.entries[protein_id][0] for protein_id in self.protein_ids]
        self.lengths = [self.entries[protein_id][1] for protein_id in self.protein_ids]
        self._pooled: dict[str, torch.Tensor] = {}
        self._mask_grids: dict[str, torch.Tensor] = {}

    @classmethod
    def from_mapping(
        cls, embeddings: Mapping[str, torch.Tensor], device: torch.device
    ) -> "ContiguousEmbeddingCache":
        """Pack individual protein tensors in sorted protein-ID order."""
        entries: dict[str, tuple[int, int]] = {}
        tensors: list[torch.Tensor] = []
        offset = 0
        for protein_id in sorted(embeddings):
            value = (
                embeddings[protein_id].detach().to("cpu", torch.float16).contiguous()
            )
            entries[protein_id] = (offset, len(value))
            offset += len(value)
            tensors.append(value)
        if not tensors:
            raise ValueError("embedding cache is empty")
        values = torch.cat(tensors).to(device)
        if device.type == "cpu":
            try:
                values = values.pin_memory()
            except RuntimeError:
                pass
        return cls(values, entries)

    def __getitem__(self, protein_id: str) -> torch.Tensor:
        offset, length = self.entries[protein_id]
        return self.values[offset : offset + length]

    def by_index(self, index: int) -> torch.Tensor:
        """Return one protein tensor using its integer cache index."""
        offset, length = (self.offsets[index], self.lengths[index])
        return self.values[offset : offset + length]

    def pooled(self, device: torch.device, *, fp32: bool = True) -> torch.Tensor:
        """Mean-pool each protein once and memoize the result on the target device."""
        key = f"{device}:{('fp32' if fp32 else 'fp16')}"
        if key not in self._pooled:
            lengths = torch.tensor(
                self.lengths, dtype=torch.long, device=self.values.device
            )
            source = self.values.float() if fp32 else self.values
            pooled = torch.segment_reduce(source, "mean", lengths=lengths)
            self._pooled[key] = pooled.to(device, non_blocking=True)
        return self._pooled[key]

    def mask_grid(self, device: torch.device) -> torch.Tensor:
        """Reuse a single position vector when constructing residue masks."""
        key = str(device)
        if key not in self._mask_grids:
            self._mask_grids[key] = torch.arange(max(self.lengths), device=device)
        return self._mask_grids[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)


def place_embeddings(
    embeddings: Mapping[str, torch.Tensor], mode: str, device: torch.device
) -> tuple[ContiguousEmbeddingCache, str]:
    """Place a packed embedding cache on GPU when requested and memory permits."""
    required = sum((t.numel() * t.element_size() for t in embeddings.values()))
    use_cuda = mode == "cuda"
    if mode == "auto" and device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        use_cuda = required < int(free * 0.7)
    if use_cuda:
        if device.type != "cuda":
            raise RuntimeError("--embedding_device cuda requires CUDA")
        return (ContiguousEmbeddingCache.from_mapping(embeddings, device), "cuda")
    if mode not in {"auto", "cpu"}:
        raise ValueError(f"invalid embedding device: {mode}")
    return (
        ContiguousEmbeddingCache.from_mapping(embeddings, torch.device("cpu")),
        "cpu",
    )


def binary_nll(labels: np.ndarray, logits: np.ndarray) -> float:
    """Calculate numerically stable binary negative log likelihood."""
    return float(np.mean(np.logaddexp(0.0, logits) - labels * logits))


def classification_metrics(
    labels: np.ndarray, logits: np.ndarray, bins: int = 15
) -> dict[str, float]:
    """Calculate the metrics reported by the public test command."""
    probabilities = expit(logits)
    labels_array = labels.astype(int)
    predicted_labels = (probabilities >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_array, predicted_labels, average="binary", zero_division=0
    )
    ece = 0.0
    bin_edges = np.linspace(0, 1, bins + 1)
    for index in range(bins):
        in_bin = (probabilities >= bin_edges[index]) & (
            probabilities < bin_edges[index + 1]
            if index < bins - 1
            else probabilities <= 1
        )
        if in_bin.any():
            ece += in_bin.mean() * abs(
                probabilities[in_bin].mean() - labels_array[in_bin].mean()
            )
    return {
        "aupr": float(average_precision_score(labels_array, probabilities)),
        "auroc": float(roc_auc_score(labels_array, probabilities)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "nll": binary_nll(labels_array, logits),
        "brier": float(brier_score_loss(labels_array, probabilities)),
        "ece": float(ece),
    }


def bootstrap_aupr(
    labels: np.ndarray, logits: np.ndarray, replicates: int, seed: int
) -> tuple[float, float]:
    """Return a paired percentile interval for AUPR."""
    rng = np.random.default_rng(seed)
    values = []
    sample_count = len(labels)
    probabilities = expit(logits)
    while len(values) < replicates:
        indices = rng.integers(0, sample_count, sample_count)
        if len(np.unique(labels[indices])) == 2:
            values.append(
                average_precision_score(labels[indices], probabilities[indices])
            )
    return tuple((float(x) for x in np.quantile(values, [0.025, 0.975])))
