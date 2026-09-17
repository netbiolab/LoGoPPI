# Small input examples

These files demonstrate the public input formats. The sequences and labels are
synthetic and are intended only for checking that commands run successfully.
They do not provide meaningful biological results or model performance.

## Inference

Score the IDs in `pairs.csv` using sequences from a FASTA file:

```bash
python inference.py --model_dir models/cross_species \
  --fasta_path examples/inference/proteins.fasta \
  --pair_csv examples/inference/pairs.csv \
  --output_path runs/example_inference.csv --gpus 0
```

The same command also accepts sequences directly from a CSV:

```bash
python inference.py --model_dir models/cross_species \
  --pair_csv examples/inference/sequence_pairs.csv \
  --output_path runs/example_sequence_inference.csv --gpus 0
```

## Test

The test CSV adds a binary `label` column. Testing applies the calibration
already stored in the model bundle; it does not refit calibration.

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.test \
  --config examples/test/config.yaml \
  --model_dir models/cross_species \
  --output_dir runs/example_test --bootstrap_replicates 20 --quiet
```

## Training

The training example contains all required roles and a matching hash manifest.
Its config uses one GPU and one epoch so that the full workflow can be checked
with a small input. ESM-2 is still a large model, so a CUDA GPU is required.

```bash
python -m scripts.training \
  --config examples/training/config.yaml \
  --output_dir runs/example_training
```
