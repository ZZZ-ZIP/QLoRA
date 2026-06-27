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

## Task Definition

This repository treats multimodal emotion recognition as an instruction
following problem rather than as a conventional classification-head problem.
For each utterance-level sample, the model receives:

- one or more sampled video frames
- the corresponding transcript text
- an instruction describing the allowed emotion labels

The model is trained to generate a compact JSON answer:

```json
{"emotion":"positive"}
```

This format makes the same model interface usable for base-model prompting,
QLoRA fine-tuning, prediction parsing, and evaluation. During evaluation,
malformed generations or labels outside the allowed set are counted as invalid
outputs, which makes instruction-following failures visible instead of silently
mapping them to a class.

Conceptually, each training example is converted into a chat-style sequence:

```text
system: You are a multimodal emotion recognition model...
user:   [video frames] The person in the video says: ...
        Determine the emotion conveyed. Allowed labels: positive, negative, neutral.
assistant: {"emotion":"<gold_label>"}
```

Only the assistant answer tokens are used as supervised targets. The prompt
tokens are masked out in the loss, so the model is optimized to produce the
emotion answer rather than to reproduce the full input prompt.

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

## Fine-Tuning Method

The fine-tuning pipeline is implemented in
[`scripts/train_qlora_emotion.py`](scripts/train_qlora_emotion.py). It uses
the Hugging Face `Trainer` together with PEFT LoRA adapters and 4-bit
quantization.

The main steps are:

1. Load the JSONL train and validation splits.
2. Load the Qwen3-VL processor and base image-text-to-text model.
3. Load the base model in 4-bit mode with `bitsandbytes`.
4. Enable gradient checkpointing to reduce memory usage.
5. Prepare the model for k-bit training with PEFT.
6. Freeze the vision backbone.
7. Insert LoRA adapters into selected language-model projection modules.
8. Train only the adapter parameters.
9. Save the final adapter and processor files.

The default QLoRA configuration uses:

```yaml
qlora:
  load_in_4bit: true
  bnb_4bit_quant_type: nf4
  bnb_4bit_compute_dtype: bfloat16
  bnb_4bit_use_double_quant: true
  lora_r: 20
  lora_alpha: 40
  lora_dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
```

The default training configuration uses small per-device batches with gradient
accumulation:

```yaml
training:
  num_train_epochs: 3
  learning_rate: 0.0001
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  optim: paged_adamw_8bit
  bf16: true
  gradient_checkpointing: true
```

This setup is designed for memory-efficient adaptation of a multimodal large
language model on limited GPU resources.

## Why This Design

Qwen3-VL already has strong image-text instruction-following capability. For
this task, the goal is not to train a new vision encoder from scratch, but to
adapt the model's multimodal reasoning and response behavior to the emotion
recognition label space used by EmoBench-M.

QLoRA is used for three reasons:

- **Memory efficiency:** 4-bit loading, NF4 quantization, double quantization,
  and 8-bit optimizer states reduce GPU memory requirements.
- **Parameter efficiency:** only LoRA adapter weights are trained, while most
  pretrained parameters remain frozen.
- **Reproducible ablations:** adapter rank, target modules, modality inputs,
  frame counts, and cross-dataset splits can be changed through YAML configs
  without rewriting the training loop.

The vision backbone is frozen because the four filtered datasets are relatively
small compared with the scale of the pretrained vision-language model. Freezing
visual parameters reduces overfitting risk and keeps training focused on how
the language model combines visual cues, transcript content, and the emotion
instruction.

The JSON output constraint is used because it gives a clear, machine-readable
prediction target. It also allows the evaluation code to separate classification
errors from formatting or instruction-following errors.

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

By default, the script trains on `data_four_datasets/train.jsonl`, evaluates on
`data_four_datasets/val.jsonl`, and writes the final adapter to the configured
`training.output_dir`.

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
