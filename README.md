# LoGoPPI

LoGoPPI is a protein–protein interaction (PPI) prediction model built on an
ESM-2 protein language model. It combines a sequence-level Global classifier
with residue-level Maxsim scoring. Each protein is embedded once, so its
embedding can be reused when scoring many protein pairs.

This repository contains the training, testing, and inference code for two
separately trained models:

- [LoGoPPI-Cross-species](https://huggingface.co/netbiolab/LoGoPPI-Cross-species)
  was trained on human PPIs derived from the
  [D-SCRIPT dataset](https://d-script.readthedocs.io/en/stable/data.html) and
  evaluated across eukaryotic and prokaryotic species.
- [LoGoPPI-Bernett](https://huggingface.co/netbiolab/LoGoPPI-Bernett) was
  trained on the
  [Bernett human PPI benchmark](https://doi.org/10.6084/m9.figshare.21591618.v3).


The model weights and exact datasets used in the study are hosted with their
respective models on Hugging Face.

## Installation

```bash
git clone https://github.com/netbiolab/LoGoPPI.git
cd LoGoPPI
conda env create -f environment.yml
conda activate logoppi
```

The provided environment targets Linux with an NVIDIA GPU and CUDA-enabled
PyTorch. An NVIDIA driver must already be installed on the system. Run the
commands below after activating the `logoppi` environment.

## Model and data download

For inference, download a model without its training and evaluation data:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    "netbiolab/LoGoPPI-Cross-species",
    revision="v2.0.0",
    ignore_patterns=["data/*"],
    local_dir="models/cross_species",
)
PY
```

Use `netbiolab/LoGoPPI-Bernett` and `models/bernett` for the Bernett model.

To run training or testing, download the matching dataset into the repository
root. Choose one of the following calls:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    "netbiolab/LoGoPPI-Cross-species",
    revision="v2.0.0",
    allow_patterns=["data/*"],
    local_dir=".",
)

# Bernett alternative
snapshot_download(
    "netbiolab/LoGoPPI-Bernett",
    revision="v2.0.0",
    allow_patterns=["data/*"],
    local_dir=".",
)
PY
```

## Input format

FASTA headers must match the protein IDs in the pair CSV:

```text
>P12345
MKT...
>Q99999
MSE...
```

Inference CSV files use `query,text`:

```csv
query,text
P12345,Q99999
P12345,Q88888
```

Training and test CSV files add a binary `label`:

```csv
query,text,label
P12345,Q99999,1
P12345,Q88888,0
```

`query` and `text` contain FASTA IDs rather than protein sequences.

## Inference

Score pairs using a FASTA file:

```bash
python inference.py \
  --model_dir models/cross_species \
  --fasta_path examples/inference/proteins.fasta \
  --pair_csv examples/inference/pairs.csv \
  --output_path predictions.csv \
  --gpus 0
```

The default output contains `query,text,score`. A CSV may instead provide
`query,text,query_sequence,text_sequence`, with no `--fasta_path`:

```bash
python inference.py \
  --model_dir models/cross_species \
  --pair_csv examples/inference/sequence_pairs.csv \
  --output_path predictions.csv \
  --gpus 0
```

Embeddings can be generated once and reused:

```bash
python inference.py \
  --model_dir models/cross_species \
  --fasta_path examples/inference/proteins.fasta \
  --pair_csv examples/inference/pairs.csv \
  --embed_only --embedding_save_path embeddings.pt --gpus 0

python inference.py \
  --model_dir models/cross_species \
  --fasta_path examples/inference/proteins.fasta \
  --pair_csv examples/inference/pairs.csv \
  --embeddings_path embeddings.pt \
  --output_path predictions.csv --gpus 0
```

## Training

Cross-species and Bernett use the same training script. 
Dataset paths and training parameters, including batch size, learning rate, number of epochs, random seed, GPU settings, and optional W&B logging, 
can be configured separately in configs/cross_species.yaml and configs/bernett.yaml.

```bash
python -m scripts.training \
  --config configs/cross_species.yaml \
  --output_dir runs/cross_species

python -m scripts.training \
  --config configs/bernett.yaml \
  --output_dir runs/bernett
```

Each run trains the Global model, selects a checkpoint on validation data,
creates the embedding cache, fits calibration on a separate validation split,
and exports `final_model/`. Resume an interrupted run with its last checkpoint:

```bash
python -m scripts.training \
  --config configs/cross_species.yaml \
  --output_dir runs/cross_species \
  --resume runs/cross_species/last.pt
```

## Test

Evaluate a released model using the test data selected in its config:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.test \
  --config configs/cross_species.yaml \
  --model_dir models/cross_species \
  --output_dir runs/test_cross_species

CUDA_VISIBLE_DEVICES=0 python -m scripts.test \
  --config configs/bernett.yaml \
  --model_dir models/bernett \
  --output_dir runs/test_bernett
```

Testing uses the calibration stored with the released model. Test labels are
used only to calculate evaluation metrics and are never used to refit
calibration.

## Examples

Small synthetic inputs and commands for inference, testing, and the complete
training workflow are provided in [`examples/`](examples/README.md). They are
intended to verify the input format and execution flow, not biological model
performance.

## Acknowledgements

Parts of the batching utilities were adapted from PLM-interact (MIT License,
Dan Liu, 2024). See [NOTICE](NOTICE) for details.

## License

LoGoPPI is released under the [Apache License 2.0](LICENSE).
