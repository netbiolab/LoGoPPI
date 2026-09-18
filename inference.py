"""Run LoGoPPI inference for one pair or a CSV of protein pairs.

The command reads sequences, embeds each unique protein once, scores all pairs,
and writes calibrated probabilities. Embeddings may be saved and reused when
the cache was produced by the same released checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path

import torch
import torch.multiprocessing as mp

from utils.prediction import FinalPredictor, place_embeddings, read_fasta_any


def main(args) -> None:
    """Load inputs and a released model, then write pair predictions."""
    # Configure CPU and GPU visibility before creating CUDA objects.
    torch.set_num_threads(max(1, args.num_workers))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    if args.output_path.exists() and (not args.overwrite) and (not args.embed_only):
        raise FileExistsError(f"output exists; use --overwrite: {args.output_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the exported ESM-2 model")

    # Read pair IDs and collect each unique sequence required by those pairs.
    device = torch.device("cuda", 0)
    sequences, pairs = read_inputs(args)
    predictor = FinalPredictor(args.model_dir.resolve(), device)

    # Load a compatible cache or embed every requested protein once.
    if args.embeddings_path:
        embeddings = load_embedding_cache(args.embeddings_path, predictor.model_id)
    else:
        embeddings = predictor.encode(sequences, args.embedding_batch_size, args.quiet)
        if args.embedding_save_path:
            save_embedding_cache(
                args.embedding_save_path,
                embeddings,
                predictor.model_id,
                predictor.model.config.max_residues,
            )

    if set(sequences).difference(embeddings):
        raise KeyError("embedding cache does not cover all requested proteins")
    if args.embed_only:
        if not args.embedding_save_path and (not args.embeddings_path):
            raise ValueError(
                "--embed_only requires --embedding_save_path or --embeddings_path"
            )
        print(f"Saved {len(embeddings)} protein embeddings.")
        return

    # Score on one GPU or divide pairs among the requested GPUs.
    world = min(
        len(pairs), len([item for item in args.gpus.split(",") if item.strip()])
    )
    if world > 1:
        del predictor
        torch.cuda.empty_cache()
        scores = multi_gpu_score(args, pairs, embeddings)
        if not args.quiet:
            print(f"Pair scoring: {world} GPUs")
    else:
        embeddings, location = place_embeddings(
            embeddings, args.embedding_device, device
        )
        if not args.quiet:
            print(f"Embedding cache: {location} ({len(embeddings)} proteins)")
        scores = predictor.score(
            pairs,
            embeddings,
            args.pair_batch_size,
            args.quiet,
            global_batch_size=args.global_batch_size,
            length_bucketing=args.length_bucketing,
        )

    # Write through a temporary file so incomplete output is never published.
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["query", "text", "score"]
    if args.detailed:
        fields = [
            "query",
            "text",
            "global_score",
            "maxsim_score",
            "global_logit",
            "maxsim_logit",
            "final_logit",
            "score",
        ]
    temporary = args.output_path.with_suffix(args.output_path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (query, text), score in zip(pairs, scores):
            row = {"query": query, "text": text, **score}
            writer.writerow({key: row[key] for key in fields})
    os.replace(temporary, args.output_path)
    print(f"Saved {len(scores)} predictions to {args.output_path}")


def load_embedding_cache(path: Path, model_id: str) -> dict[str, torch.Tensor]:
    """Load embeddings only when they were created by the selected model."""
    cached = torch.load(path, map_location="cpu", weights_only=True)
    if cached.get("format") != "logoppi_residue_embeddings_v2":
        raise ValueError(
            "unsupported or legacy embedding cache; regenerate it with this "
            "version of inference.py"
        )
    if cached.get("model_id") != model_id:
        raise RuntimeError("embedding cache belongs to another model checkpoint")
    return cached["embeddings"]


def save_embedding_cache(
    path: Path,
    embeddings: dict[str, torch.Tensor],
    model_id: str,
    max_residues: int,
) -> None:
    """Save reusable residue embeddings with their public model identity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format": "logoppi_residue_embeddings_v2",
            "model_id": model_id,
            "max_residues": int(max_residues),
            "embeddings": embeddings,
        },
        temporary,
    )
    os.replace(temporary, path)


def read_inputs(
    args: argparse.Namespace,
) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Read one pair, FASTA IDs, or sequences stored directly in a CSV file."""
    # Single-pair and file inputs are intentionally mutually exclusive.
    single = args.sequence_a is not None or args.sequence_b is not None
    files = args.fasta_path is not None or args.pair_csv is not None
    if single and files:
        raise ValueError("use either sequence_a/sequence_b or file input, not both")
    if single:
        if not args.sequence_a or not args.sequence_b:
            raise ValueError("both --sequence_a and --sequence_b are required")
        return (
            {
                "sequence_a": args.sequence_a.upper(),
                "sequence_b": args.sequence_b.upper(),
            },
            [("sequence_a", "sequence_b")],
        )
    if not args.pair_csv:
        raise ValueError("provide a single pair or --pair_csv")
    sequences = read_fasta_any(args.fasta_path) if args.fasta_path else {}
    pairs: list[tuple[str, str]] = []
    with args.pair_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if args.fasta_path:
            if not {"query", "text"}.issubset(fields):
                raise ValueError("FASTA pair CSV requires query,text columns")
            for row in reader:
                pairs.append((row["query"].strip(), row["text"].strip()))
        else:
            if not {"query_sequence", "text_sequence"}.issubset(fields):
                raise ValueError(
                    "sequence CSV requires query_sequence,text_sequence columns"
                )
            for index, row in enumerate(reader):
                query = row.get("query", "").strip() or f"row_{index}_query"
                text = row.get("text", "").strip() or f"row_{index}_text"
                if query in sequences or text in sequences:
                    raise ValueError("direct sequence CSV IDs must be unique")
                sequences[query] = row["query_sequence"].strip().upper()
                sequences[text] = row["text_sequence"].strip().upper()
                pairs.append((query, text))
    if not pairs:
        raise ValueError("pair CSV is empty")
    missing = sorted({pid for pair in pairs for pid in pair}.difference(sequences))
    if missing:
        raise KeyError(
            f"{len(missing)} pair IDs are absent from FASTA; first: {missing[0]}"
        )
    return ({pid: sequences[pid] for pair in pairs for pid in pair}, pairs)


def multi_gpu_score(
    args: argparse.Namespace,
    pairs: list[tuple[str, str]],
    embeddings: dict[str, torch.Tensor],
) -> list[dict[str, float]]:
    """Score rank-strided pair shards and restore their original row order."""
    world = min(
        len(pairs), len([item for item in args.gpus.split(",") if item.strip()])
    )
    with tempfile.TemporaryDirectory(prefix="logobert_inference_") as temporary:
        temp = Path(temporary)
        cache = temp / "embeddings.pt"
        torch.save(embeddings, cache)
        mp.spawn(
            _score_worker,
            args=(
                world,
                str(args.model_dir.resolve()),
                str(cache),
                pairs,
                args.pair_batch_size,
                args.global_batch_size,
                args.length_bucketing,
                args.embedding_device,
                args.quiet,
                str(temp),
            ),
            nprocs=world,
            join=True,
        )
        ordered: list[dict[str, float] | None] = [None] * len(pairs)
        for rank in range(world):
            shard = torch.load(
                temp / f"rank_{rank}.pt", map_location="cpu", weights_only=False
            )
            for index, score in zip(shard["indices"], shard["scores"]):
                ordered[index] = score
        if any((score is None for score in ordered)):
            raise RuntimeError("multi-GPU score coverage is incomplete")
        return [score for score in ordered if score is not None]


def _score_worker(
    rank: int,
    world: int,
    model_dir: str,
    cache_path: str,
    pairs: list[tuple[str, str]],
    batch_size: int,
    global_batch_size: int,
    length_bucketing: bool,
    embedding_device: str,
    quiet: bool,
    output_dir: str,
) -> None:
    """Score the pair shard assigned to one local GPU process."""
    # Each spawned process sees the selected GPUs as local ranks 0..world-1.
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    local_indices = list(range(rank, len(pairs), world))
    local_pairs = [pairs[index] for index in local_indices]
    all_embeddings = torch.load(cache_path, map_location="cpu", weights_only=True)
    required = {pid for pair in local_pairs for pid in pair}
    embeddings = {pid: all_embeddings[pid] for pid in required}
    del all_embeddings

    # Keep only this shard's embeddings before choosing CPU or GPU placement.
    embeddings, _ = place_embeddings(embeddings, embedding_device, device)
    predictor = FinalPredictor(Path(model_dir), device)
    scores = predictor.score(
        local_pairs,
        embeddings,
        batch_size,
        quiet or rank != 0,
        global_batch_size=global_batch_size,
        length_bucketing=length_bucketing,
    )

    # Preserve global row indices so the parent can restore input order.
    target = Path(output_dir) / f"rank_{rank}.pt"
    temporary = target.with_suffix(".pt.tmp")
    torch.save({"indices": local_indices, "scores": scores}, temporary)
    os.replace(temporary, target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--fasta_path", type=Path)
    parser.add_argument("--pair_csv", type=Path)
    parser.add_argument("--sequence_a")
    parser.add_argument("--sequence_b")
    parser.add_argument("--output_path", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--detailed", action="store_true")
    parser.add_argument("--embedding_batch_size", type=int, default=32)
    parser.add_argument("--pair_batch_size", type=int, default=128)
    parser.add_argument("--global_batch_size", type=int, default=4096)
    parser.add_argument(
        "--length_bucketing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--embedding_device", choices=("auto", "cuda", "cpu"), default="auto"
    )
    parser.add_argument("--embedding_save_path", type=Path)
    parser.add_argument("--embeddings_path", type=Path)
    parser.add_argument("--embed_only", action="store_true")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    main(parser.parse_args())
