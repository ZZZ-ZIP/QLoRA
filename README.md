# Qwen3-VL-2B QLoRA for Multimodal Emotion Recognition

This repository contains code and configuration for fine-tuning
**Qwen3-VL-2B** with **QLoRA** on a multimodal emotion recognition task.
The training data is built from four filtered datasets used in the
**EmoBench-M** paper, and can be downloaded from Hugging Face.

The goal is to predict the speaker's emotion from video frames and transcript
text. The model is prompted to return one JSON object:

```json
{"emotion":"positive"}
```

Allowed labels:

- `positive`
- `negative`
- `neutral`

## Overview

This project focuses on instruction-style multimodal emotion recognition:

- **Base model:** Qwen3-VL-2B / Qwen3-VL-2B-Instruct
- **Fine-tuning method:** QLoRA
- **Input modalities:** sampled video frames and utterance transcript
- **Output format:** JSON emotion label
- **Task type:** three-class sentiment/emotion recognition
- **Datasets:** four filtered datasets from EmoBench-M

The repository is organized for reproducible experiments. Raw datasets,
extracted frames, checkpoints, model weights, cache files, and private logs are
not included.

## Data

The data comes from four filtered multimodal datasets used in the EmoBench-M
paper:

- CMU-MOSI
- CMU-MOSEI
- CH-SIMSv2
- SIMS

Please download the EmoBench-M data from Hugging Face:

- [GMLHUHE/Emobench-M](https://huggingface.co/datasets/GMLHUHE/Emobench-M)

This project uses the four filtered subsets corresponding to `MOSEI`, `MOSI`,
`SIMS`, and `ch-simsv2s`. After downloading and preprocessing the required
subsets, place the processed split files under:

```text
data_four_datasets/
  train.jsonl
  val.jsonl
  test.jsonl
```

If your data is stored somewhere else, edit `data.output_dir` in the YAML
configuration files. The Hugging Face dataset card reports the EmoBench-M
dataset under the Apache-2.0 license and links the associated arXiv paper
`2502.04424`; please follow the dataset card and original dataset licenses when
using or redistributing data.

### JSONL Format

Each line in the split files should be one JSON object:

```json
{
  "sample_id": "mosei_000001",
  "dataset": "MOSEI",
  "frame_paths": [
    "frames/mosei_000001/000001.jpg",
    "frames/mosei_000001/000002.jpg"
  ],
  "transcript": "I really enjoyed this experience.",
  "label": "positive",
  "split": "train",
  "subject_id": "speaker_001"
}
```

Required fields:

- `frame_paths`: list of sampled video-frame image paths
- `transcript`: utterance transcript text
- `label`: one of `positive`, `negative`, or `neutral`

Recommended fields:

- `sample_id`: unique sample identifier
- `dataset`: source dataset name
- `split`: `train`, `val`, or `test`
- `subject_id`: speaker or video-level grouping identifier, if available

See [data/README.md](data/README.md) for additional data notes.

## Repository Layout

```text
.
|-- configs/       # YAML configs for QLoRA, rank sweep, ablations, and runs
|-- data/          # Data format instructions only
|-- docs/          # Reproducibility notes and experiment command index
|-- figures/       # Public figures or placeholders
|-- results/       # Lightweight result placeholders
|-- scripts/       # Data preparation, training, prediction, and evaluation
|-- src/           # Reserved for reusable package modules
|-- README.md
|-- requirements.txt
`-- .gitignore
```

## Installation

Create a Python environment with a CUDA-enabled PyTorch build suitable for your
GPU, then install the project dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies include:

- `torch`
- `transformers`
- `accelerate`
- `peft`
- `bitsandbytes`
- `scikit-learn`
- `Pillow`
- `PyYAML`

## Model Setup

By default, the configuration expects the base model at:

```text
../models/Qwen3-VL-2B-Instruct
```

If you download the model directly from Hugging Face, update the model section
in the config file:

```yaml
model:
  base_model: Qwen/Qwen3-VL-2B-Instruct
  local_files_only: false
  trust_remote_code: true
```

The default four-dataset experiment config is:

```text
configs/emotion_qlora_four_datasets.yaml
```

## Check Environment

Before training, verify that the model path, Python environment, and data files
are ready:

```bash
python scripts/check_environment.py --config configs/emotion_qlora_four_datasets.yaml
```

## Training

Train a QLoRA adapter on the four-dataset emotion recognition setup:

```bash
python scripts/train_qlora_emotion.py --config configs/emotion_qlora_four_datasets.yaml
```

The default QLoRA settings include:

- 4-bit loading
- NF4 quantization
- bfloat16 computation
- double quantization
- gradient checkpointing
- LoRA adapters on attention and MLP projection modules

The default target modules are:

```yaml
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
```

## Rank Sweep

Run the rank sweep from `r=2` to `r=32`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_rank_2_32_four_datasets.ps1 -IncludeBase -SkipExisting
```

This script can train adapters, run prediction, evaluate metrics, and generate
rank-sweep outputs under ignored experiment folders.

## Inference

Run base-model prediction:

```bash
python scripts/predict_emotion.py \
  --config configs/emotion_qlora_four_datasets.yaml \
  --split test \
  --model_kind base \
  --output_file outputs/base_test_predictions.jsonl
```

Run adapter prediction:

```bash
python scripts/predict_emotion.py \
  --config configs/rank_sweep_four_datasets/emotion_qlora_r32.yaml \
  --split test \
  --model_kind adapter \
  --output_file outputs/adapter_r32_test_predictions.jsonl
```

## Evaluation

Evaluate model predictions:

```bash
python scripts/evaluate_emotion.py \
  --config configs/emotion_qlora_four_datasets.yaml \
  --predictions outputs/adapter_r32_test_predictions.jsonl \
  --output_file outputs/adapter_r32_test_metrics.json \
  --output_dir outputs/eval_r32
```

Reported metrics include:

- Accuracy
- Precision
- Recall
- Macro-F1
- Weighted-F1
- per-class metrics
- confusion matrix
- invalid-output count

## Experiment Commands

| Stage | Description | Example command |
|---|---|---|
| Rank sweep | Compare base model and LoRA ranks from `r=2` to `r=32` | `scripts/run_rank_2_32_four_datasets.ps1` |
| Main comparison | Compare selected ranks such as `r8`, `r16`, `r20`, `r28`, and `r32` | `python scripts/run_four_dataset_experiment_stage.py --stage stage2_main_table` |
| Modality ablation | Compare video+text, text-only, vision-only, and random-frame settings | `python scripts/run_four_dataset_experiment_stage.py --stage modality_ablation --rank 32 --skip_existing` |
| Cross-dataset generalization | Leave-one-dataset-out evaluation | `python scripts/run_four_dataset_experiment_stage.py --stage cross_dataset --rank 32 --skip_existing` |
| Seed stability | Run multiple random seeds | `python scripts/run_four_dataset_experiment_stage.py --stage seed_stability --rank_best 32 --rank_stable 28 --seeds 42 123 2024 --skip_existing` |
| LoRA module ablation | Compare attention-only, MLP-only, and attention+MLP LoRA modules | `python scripts/run_four_dataset_experiment_stage.py --stage lora_module_ablation --rank 32 --skip_existing` |
| Alpha scaling | Compare rank-alpha combinations | `python scripts/run_four_dataset_experiment_stage.py --stage alpha_scaling --skip_existing` |
| Frame ablation | Study different numbers of sampled video frames | `python scripts/run_frame_ablation.py --rank 28 --skip_existing` |

More commands are listed in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Results

Fill this table with metrics generated by `scripts/evaluate_emotion.py` or the
experiment stage launcher.

| Setting | Accuracy | Macro-F1 | Weighted-F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Base Qwen3-VL-2B | TBD | TBD | TBD | TBD | TBD |
| QLoRA r8 | TBD | TBD | TBD | TBD | TBD |
| QLoRA r16 | TBD | TBD | TBD | TBD | TBD |
| QLoRA r20 | TBD | TBD | TBD | TBD | TBD |
| QLoRA r28 | TBD | TBD | TBD | TBD | TBD |
| QLoRA r32 | TBD | TBD | TBD | TBD | TBD |

## Notes

- Do not commit raw videos, extracted frames, private data files, model
  checkpoints, adapter weights, `.safetensors`, `.bin`, `.pt`, `.pth`, wandb
  logs, `.env`, or API keys.
- Paths inside JSONL files may be absolute or relative. Make sure they resolve
  correctly on the machine used for training and inference.
- QLoRA with Qwen3-VL requires a GPU environment compatible with 4-bit
  `bitsandbytes` loading.
- Dataset use must follow the licenses and terms of the original datasets and
  the Hugging Face dataset card.

## Citation

If you use this repository, please cite the original datasets, the EmoBench-M
paper, and Qwen3-VL.

```bibtex
@misc{emobenchm,
  title = {EmoBench-M: Benchmarking Emotional Intelligence for Multimodal Large Language Models},
  year = {2025},
  eprint = {2502.04424},
  archivePrefix = {arXiv},
  url = {https://huggingface.co/datasets/GMLHUHE/Emobench-M}
}
```

## License

This repository is released under the MIT License unless otherwise specified.
Dataset licenses and model licenses are governed by their original providers.
