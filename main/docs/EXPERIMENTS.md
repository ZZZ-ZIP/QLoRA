# Experiment Commands

Run commands from the repository root.

## Environment Check

```bash
python scripts/check_environment.py --config configs/emotion_qlora_four_datasets.yaml
```

## Rank Sweep r=2..32

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_rank_2_32_four_datasets.ps1 -Ranks 2..32 -IncludeBase -SkipExisting
```

Because the script defaults to `r=2..32`, this shorter command is usually enough:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_rank_2_32_four_datasets.ps1 -IncludeBase -SkipExisting
```

Manual single-rank run:

```bash
python scripts/train_qlora_emotion.py --config configs/rank_sweep_four_datasets/emotion_qlora_r32.yaml
python scripts/predict_emotion.py --config configs/rank_sweep_four_datasets/emotion_qlora_r32.yaml --split test --model_kind adapter --output_file outputs/adapter_r32_test_predictions.jsonl
python scripts/evaluate_emotion.py --config configs/rank_sweep_four_datasets/emotion_qlora_r32.yaml --predictions outputs/adapter_r32_test_predictions.jsonl --output_file outputs/adapter_r32_test_metrics.json --output_dir outputs/eval_r32
```

## Stage 2: Main Comparison

```bash
python scripts/run_four_dataset_experiment_stage.py --stage stage2_main_table --skip_existing
```

Main comparison includes:

- base
- r8
- r16
- r20
- r28
- r32

## Stage 3: Modality Ablation

```bash
python scripts/run_four_dataset_experiment_stage.py --stage modality_ablation --rank 32 --skip_existing
python scripts/run_four_dataset_experiment_stage.py --stage modality_ablation --rank 28 --skip_existing
```

Input modes:

- `video_text`
- `text_only`
- `vision_only`
- `random_frame_text`

## Stage 5: Cross-Dataset Generalization

```bash
python scripts/run_four_dataset_experiment_stage.py --stage cross_dataset --rank 32 --skip_existing
```

Leave-one-dataset-out settings:

- hold out MOSEI
- hold out MOSI
- hold out CH-SIMSv2
- hold out SIMS

## Stage 6: Multi-Seed Stability

```bash
python scripts/run_four_dataset_experiment_stage.py --stage seed_stability --rank_best 32 --rank_stable 28 --seeds 42 123 2024 --skip_existing
```

## Stage 7: LoRA Module Ablation

```bash
python scripts/run_four_dataset_experiment_stage.py --stage lora_module_ablation --rank 32 --skip_existing
```

Module settings:

- `attention_only`
- `mlp_only`
- `attention_mlp`

## Stage 7: Alpha Scaling

```bash
python scripts/run_four_dataset_experiment_stage.py --stage alpha_scaling --skip_existing
```

Example combinations:

- r16 alpha16/32/64
- r32 alpha32/64/128

## Stage 8: Neutral Error Analysis

```bash
python scripts/run_four_dataset_experiment_stage.py --stage neutral_error_analysis --rank 32
python scripts/run_four_dataset_experiment_stage.py --stage neutral_error_analysis --rank 28
```

This analyzes whether neutral samples are absorbed into positive or negative predictions.

## Supplementary Frame Ablation

```bash
python scripts/run_frame_ablation.py --rank 28 --skip_existing
```

Frame ablation is treated as supplementary unless complete metrics are available.

## Text and Prompt Baselines

Text baselines:

```bash
python scripts/run_text_baselines.py
python scripts/train_xlmr_text_baseline.py
```

Qwen3-VL prompt baselines:

```bash
python scripts/run_qwen3vl_prompt_baselines.py --config configs/emotion_qlora_four_datasets.yaml
```

## GitHub Upload

After reviewing files:

```bash
git init
git add .
git commit -m "Initial release: QLoRA multimodal emotion recognition experiments"
git branch -M main
git remote add origin <repo_url>
git push -u origin main
```
