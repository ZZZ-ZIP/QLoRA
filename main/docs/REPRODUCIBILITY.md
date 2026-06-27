# Reproducibility Notes

## What Is Included

- Training, prediction, evaluation, and analysis scripts.
- YAML configs for four-dataset QLoRA experiments.
- Rank sweep, modality ablation, cross-dataset, seed-stability, LoRA-module, alpha-scaling, neutral-error, and frame-ablation launchers.
- Documentation of expected data schema and command lines.

## What Is Not Included

- Raw videos.
- Extracted frames.
- Prepared private JSONL splits.
- Qwen3-VL base model weights.
- QLoRA adapter checkpoints.
- `.safetensors`, `.bin`, `.pt`, `.pth`, and optimizer state files.
- wandb logs, local caches, `.env`, API keys, and generated heavy outputs.

## Selection Protocol

Use the validation split to choose hyperparameters such as:

- LoRA rank.
- LoRA alpha.
- frame count.
- neutral oversampling ratio.

The test split should be used only for final reporting.

## Output Constraint

All model outputs should follow:

```json
{"emotion":"<label>"}
```

where `<label>` is one of `positive`, `negative`, or `neutral`.
