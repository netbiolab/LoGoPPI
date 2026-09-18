"""Data structures, batches, and residue caches used by LoGoPPI.

The public training pipeline expects prepared ``query,text,label`` CSV files
whose protein IDs occur in the accompanying FASTA file. Dataset preparation
and research-specific split generation are intentionally kept outside this
runtime module.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler


@dataclass(frozen=True, slots=True)
class ProteinPair:
    """One labeled pair from a prepared training or evaluation CSV file."""

    source_row_id: int
    query: str
    text: str
    label: int


@dataclass(frozen=True, slots=True)
class PairExample:
    """A protein pair together with the two sequences used by the model."""

    dataset_index: int
    source_row_id: int
    query: str
    text: str
    query_sequence: str
    text_sequence: str
    label: int


def read_fasta(path: str | Path) -> dict[str, str]:
    """Read a FASTA file as ``protein ID -> uppercase sequence``."""
    records: dict[str, str] = {}
    identifier: str | None = None
    chunks: list[str] = []

    def commit() -> None:
        nonlocal identifier, chunks
        if identifier is None:
            return
        sequence = "".join(chunks).replace(" ", "").upper()
        if not sequence or identifier in records:
            raise ValueError(f"empty or duplicate FASTA record: {identifier}")
        records[identifier] = sequence

    with Path(path).open(encoding="utf-8-sig") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                commit()
                identifier, chunks = (line[1:].split(maxsplit=1)[0], [])
            else:
                if identifier is None:
                    raise ValueError("sequence before FASTA header")
                chunks.append(line)
    commit()
    if not records:
        raise ValueError("FASTA is empty")
    return records


class ProteinPairDataset(Dataset[PairExample]):
    """Attach FASTA sequences to the protein IDs in each labeled pair."""

    def __init__(
        self, pairs: Sequence[ProteinPair], sequences: Mapping[str, str]
    ) -> None:
        self.pairs, self.sequences = (tuple(pairs), sequences)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> PairExample:
        pair = self.pairs[index]
        return PairExample(
            index,
            pair.source_row_id,
            pair.query,
            pair.text,
            self.sequences[pair.query],
            self.sequences[pair.text],
            pair.label,
        )


class PairCollator:
    """Tokenize two protein columns and retain explicit special-token masks."""

    def __init__(self, tokenizer: Any, max_residues: int = 800) -> None:
        self.tokenizer, self.max_residues = (tokenizer, int(max_residues))

    def _tokenize(self, sequences: Sequence[str]) -> dict[str, Tensor]:
        encoded = self.tokenizer(
            [sequence[: self.max_residues] for sequence in sequences],
            padding=True,
            truncation=True,
            max_length=self.max_residues + 2,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        return {
            key: encoded[key]
            for key in ("input_ids", "attention_mask", "special_tokens_mask")
        }

    def __call__(self, examples: Sequence[PairExample]) -> dict[str, Any]:
        return {
            "input_a": self._tokenize([row.query_sequence for row in examples]),
            "input_b": self._tokenize([row.text_sequence for row in examples]),
            "labels": torch.tensor(
                [row.label for row in examples], dtype=torch.float32
            ),
            "dataset_indices": torch.tensor(
                [row.dataset_index for row in examples], dtype=torch.long
            ),
            "source_indices": torch.tensor(
                [row.source_row_id for row in examples], dtype=torch.long
            ),
        }


class TruncatedDistributedTrainSampler(Sampler[int]):
    """Shuffle without padding and end every epoch at an accumulation boundary."""

    def __init__(
        self,
        size: int,
        batch_size: int,
        accumulation: int,
        seed: int,
        replicas: int,
        rank: int,
    ) -> None:
        self.size, self.seed, self.epoch = (int(size), int(seed), 0)
        self.replicas, self.rank = (int(replicas), int(rank))
        global_update = int(batch_size) * self.replicas * int(accumulation)
        self.total_size = self.size // global_update * global_update
        self.num_samples = self.total_size // self.replicas
        if self.total_size == 0:
            raise ValueError("dataset is smaller than one effective batch")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.size, generator=generator).tolist()[
            : self.total_size
        ]
        return iter(indices[self.rank : self.total_size : self.replicas])

    def __len__(self) -> int:
        return self.num_samples


class DistributedEvalSampler(Sampler[int]):
    """Rank-strided validation sampler with no padding or duplicates."""

    def __init__(self, size: int, replicas: int, rank: int) -> None:
        self.size, self.replicas, self.rank = (int(size), int(replicas), int(rank))

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, self.size, self.replicas))

    def __len__(self) -> int:
        return len(range(self.rank, self.size, self.replicas))


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """Location of one protein in the contiguous residue cache."""

    protein_id: str
    offset: int
    length: int


class ProjectionCache:
    """Read-only memory map of concatenated ``[residue,dimension]`` FP16 rows."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.manifest = json.loads((self.cache_dir / "cache_manifest.json").read_text())
        if self.manifest.get("format") != "logoppi_residue_cache_v2":
            raise ValueError("unsupported projection cache format")
        if self.manifest.get("special_tokens") != "excluded":
            raise ValueError("Step 2 requires a residue-only cache")
        if self.manifest.get("dtype") != "float16":
            raise ValueError("projection cache dtype must be float16")
        self.dimension = int(self.manifest["projection_dim"])
        self.total_residues = int(self.manifest["total_residues"])
        data_path = self.cache_dir / "residue_projection.fp16"
        index_path = self.cache_dir / "protein_index.csv"
        expected_bytes = (
            self.total_residues * self.dimension * np.dtype(np.float16).itemsize
        )
        if data_path.stat().st_size != expected_bytes:
            raise RuntimeError("projection cache size does not match its metadata")
        self.values = np.memmap(
            data_path,
            mode="r",
            dtype=np.float16,
            shape=(self.total_residues, self.dimension),
        )
        self.entries: dict[str, CacheEntry] = {}
        with index_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["protein_id", "offset", "length"]:
                raise ValueError(
                    "projection index columns must be protein_id,offset,length"
                )
            for row in reader:
                entry = CacheEntry(
                    row["protein_id"], int(row["offset"]), int(row["length"])
                )
                if entry.protein_id in self.entries:
                    raise ValueError(
                        f"duplicate projection cache ID: {entry.protein_id}"
                    )
                if entry.offset < 0 or entry.length <= 0:
                    raise ValueError(
                        f"invalid projection cache row: {entry.protein_id}"
                    )
                if entry.offset + entry.length > self.total_residues:
                    raise ValueError(
                        f"projection cache row exceeds data: {entry.protein_id}"
                    )
                self.entries[entry.protein_id] = entry
        if len(self.entries) != int(self.manifest["protein_count"]):
            raise RuntimeError("projection cache protein count mismatch")

    def get(self, protein_id: str) -> np.ndarray:
        try:
            entry = self.entries[protein_id]
        except KeyError as exc:
            raise KeyError(
                f"protein absent from projection cache: {protein_id}"
            ) from exc
        return self.values[entry.offset : entry.offset + entry.length]


class CachedPairDataset(Dataset[tuple[int, ProteinPair]]):
    """Return pair rows with their original dataset positions."""

    def __init__(self, pairs: Sequence[ProteinPair]) -> None:
        self.pairs = tuple(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[int, ProteinPair]:
        return (index, self.pairs[index])


class CachedPairCollator:
    """Pad cached residue projections; masks contain residues only."""

    def __init__(self, cache: ProjectionCache) -> None:
        self.cache = cache

    @staticmethod
    def _pad(arrays: Sequence[np.ndarray], dimension: int) -> tuple[Tensor, Tensor]:
        maximum = max((array.shape[0] for array in arrays))
        values = torch.zeros(len(arrays), maximum, dimension, dtype=torch.float16)
        mask = torch.zeros(len(arrays), maximum, dtype=torch.bool)
        for index, array in enumerate(arrays):
            length = array.shape[0]
            values[index, :length].copy_(torch.from_numpy(np.array(array, copy=True)))
            mask[index, :length] = True
        return (values, mask)

    def __call__(self, rows: Sequence[tuple[int, ProteinPair]]) -> dict[str, Tensor]:
        indices, pairs = zip(*rows)
        embeddings_a, mask_a = self._pad(
            [self.cache.get(p.query) for p in pairs], self.cache.dimension
        )
        embeddings_b, mask_b = self._pad(
            [self.cache.get(p.text) for p in pairs], self.cache.dimension
        )
        return {
            "embeddings_a": embeddings_a,
            "mask_a": mask_a,
            "embeddings_b": embeddings_b,
            "mask_b": mask_b,
            "labels": torch.tensor([p.label for p in pairs], dtype=torch.float32),
            "dataset_indices": torch.tensor(indices, dtype=torch.long),
            "source_indices": torch.tensor(
                [p.source_row_id for p in pairs], dtype=torch.long
            ),
        }


def write_cache_manifest(
    cache_dir: Path,
    *,
    dimension: int,
    total_residues: int,
    protein_count: int,
) -> dict[str, Any]:
    """Record the residue-cache layout needed for safe memory mapping."""
    manifest: dict[str, Any] = {
        "format": "logoppi_residue_cache_v2",
        "projection_dim": dimension,
        "dtype": "float16",
        "max_residues": 800,
        "special_tokens": "excluded",
        "protein_count": protein_count,
        "total_residues": total_residues,
    }
    temporary = cache_dir / "cache_manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(cache_dir / "cache_manifest.json")
    return manifest
