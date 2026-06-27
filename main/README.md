# QLoRA Multimodal Emotion Recognition with Qwen3-VL

This repository contains a reproducible experiment framework for multimodal three-class emotion recognition based on **Qwen3-VL-2B-Instruct** and **QLoRA**. The model takes sampled video frames, transcript text, and an instruction prompt as input, and is constrained to output a JSON object:

```json
{"emotion":"positive/negative/neutral"}
```

The project is organized for GitHub release. Raw datasets, extracted frames, checkpoints, model weights, cache files, and private logs are intentionally excluded.

## Task Definition

Given a multimodal sample

```json
{
  "sample_id": "mosei_000001",
  "dataset": "MOSEI",
  "frame_paths": ["frames/mosei_000001/000001.jpg", "frames/mosei_000001/000002.jpg"],
  "transcript": "I really enjoyed this experience.",
  "label": "positive"
}
```

the system predicts one label from:

- `positive`
- `negative`
- `neutral`

The prediction parser accepts the expected JSON format and maps invalid or malformed outputs to `invalid` for evaluation diagnostics.

## Model and QLoRA Setting

The main model is `Qwen3-VL-2B-Instruct`. Fine-tuning uses QLoRA with:

- 4-bit loading
- NF4 quantization
- bfloat16 computation
- double quantization
- `prepare_model_for_kbit_training`
- frozen vision backbone
- LoRA adapters injected mainly into language-model attention modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`

Some mechanism ablations also include MLP modules such as `gate_proj`, `up_proj`, and `down_proj`.

The default four-dataset configuration is:

```bash
configs/emotion_qlora_four_datasets.yaml
```

By default, configs expect the base model at:

```bash
../models/Qwen3-VL-2B-Instruct
```

Edit `model.base_model` in the YAML files if your model is stored elsewhere.

## Datasets

The experiment uses four multimodal sentiment/emotion datasets:

- CMU-MOSI
- CMU-MOSEI
- CH-SIMSv2
- SIMS

Raw videos and extracted image frames are not distributed in this repository. Prepare local JSONL files under `data_four_datasets/` or change `data.output_dir` in the configs.

Expected split files:

```text
data_four_datasets/
  train.jsonl
  val.jsonl
  test.jsonl
```

See [data/README.md](data/README.md) for the exact JSONL schema.

## Repository Layout

```text
.
├── configs/      # YAML configs for QLoRA, rank sweep, ablations, and cross-dataset runs
├── data/         # Dataset instructions only; raw data is ignored
├── docs/         # Reproduction notes and experiment command index
├── figures/      # Public figures only; generated figures are ignored by default
├── results/      # Lightweight result placeholders only
├── scripts/      # Training, prediction, evaluation, data preparation, and ablation launchers
├── src/          # Reserved for reusable package modules
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

Create an environment with a CUDA-enabled PyTorch build suitable for your GPU, then install dependencies:

```bash
pip install -r requirements.txt
```

Check the environment, model path, and prepared split files:

```bash
python scripts/check_environment.py --config configs/emotion_qlora_four_datasets.yaml
```

## Training

Train a single QLoRA adapter:

```bash
python scripts/train_qlora_emotion.py --config configs/emotion_qlora_four_datasets.yaml
```

Train a specific rank from the generated rank-sweep configs:

```bash
python scripts/train_qlora_emotion.py --config configs/rank_sweep_four_datasets/emotion_qlora_r32.yaml
```

Run the full rank sweep from `r=2` to `r=32` on Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_rank_2_32_four_datasets.ps1 -Ranks 2..32 -IncludeBase -SkipExisting
```

The script default is already `r=2..32`, so the simpler equivalent command is:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_rank_2_32_four_datasets.ps1 -IncludeBase -SkipExisting
```

The script trains adapters, predicts the test split, evaluates metrics, and writes rank-sweep plots/tables under ignored output folders.

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

Evaluate predictions:

```bash
python scripts/evaluate_emotion.py \
  --config configs/emotion_qlora_four_datasets.yaml \
  --predictions outputs/adapter_r32_test_predictions.jsonl \
  --output_file outputs/adapter_r32_test_metrics.json \
  --output_dir outputs/eval_r32
```

Metrics include:

- Accuracy
- Precision
- Recall
- Macro-F1
- Weighted-F1
- per-class metrics
- confusion matrix
- invalid-output count

## Experiment Stages

The completed experiment design includes:

| Stage | Description | Example command |
|---|---|---|
| Rank sweep | `base` and `r=2..32` | `scripts/run_rank_2_32_four_datasets.ps1` |
| Main comparison | `base / r8 / r16 / r20 / r28 / r32` | `python scripts/run_four_dataset_experiment_stage.py --stage stage2_main_table` |
| Modality ablation | `video_text`, `text_only`, `vision_only`, `random_frame_text` | `python scripts/run_four_dataset_experiment_stage.py --stage modality_ablation --rank 32 --skip_existing` |
| Cross-dataset generalization | leave-one-dataset-out | `python scripts/run_four_dataset_experiment_stage.py --stage cross_dataset --rank 32 --skip_existing` |
| Seed stability | seeds `42 / 123 / 2024` | `python scripts/run_four_dataset_experiment_stage.py --stage seed_stability --rank_best 32 --rank_stable 28 --seeds 42 123 2024 --skip_existing` |
| LoRA module ablation | `attention_only`, `mlp_only`, `attention_mlp` | `python scripts/run_four_dataset_experiment_stage.py --stage lora_module_ablation --rank 32 --skip_existing` |
| Alpha scaling | rank-alpha combinations | `python scripts/run_four_dataset_experiment_stage.py --stage alpha_scaling --skip_existing` |
| Neutral error analysis | neutral absorbed into positive/negative | `python scripts/run_four_dataset_experiment_stage.py --stage neutral_error_analysis --rank 32` |
| Frame ablation | supplementary frame-count study | `python scripts/run_frame_ablation.py --rank 28 --skip_existing` |

More commands are listed in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Result Table Template

Fill the table with metrics generated by `scripts/evaluate_emotion.py` or the stage launcher.

| Setting | Accuracy | Macro-F1 | Weighted-F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| base | TBD | TBD | TBD | TBD | TBD |
| QLoRA r8 | TBD | TBD | TBD | TBD | TBD |
| QLoRA r16 | TBD | TBD | TBD | TBD | TBD |
| QLoRA r20 | TBD | TBD | TBD | TBD | TBD |
| QLoRA r28 | TBD | TBD | TBD | TBD | TBD |
| QLoRA r32 | TBD | TBD | TBD | TBD | TBD |

## Notes

- Do not commit raw videos, extracted frames, model checkpoints, adapter weights, `.safetensors`, `.bin`, `.pt`, `.pth`, wandb logs, `.env`, or API keys.
- Paths inside JSONL files may be absolute or relative. Ensure they resolve correctly on the machine used for training/inference.
- QLoRA with Qwen3-VL requires a GPU environment compatible with 4-bit bitsandbytes loading.
- If the base model is downloaded from Hugging Face instead of a local directory, set `local_files_only: false` and update `model.base_model`.
- The repository contains code and configuration for reproducibility. Dataset access must follow the licenses and terms of the original datasets.

## License

This repository is released under the MIT License unless otherwise specified. Dataset licenses and model licenses are governed by their original providers.
