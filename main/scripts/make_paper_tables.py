from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import pandas as pd

from paper_extension_utils import (
    EXT_ROOT,
    PAPER_TABLE_DIR,
    dataframe_to_markdown,
    ensure_extension_dirs,
    existing_metric_rows,
    rank_metric_rows,
    read_json,
    round_float,
    write_text,
)


MAIN_COLUMNS = [
    "model",
    "accuracy",
    "macro_f1",
    "positive_f1",
    "negative_f1",
    "neutral_f1",
    "weighted_f1",
    "valid_json_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create paper-ready tables and figures from existing metrics.")
    parser.add_argument("--output_dir", default=str(PAPER_TABLE_DIR))
    return parser.parse_args()


def _round_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in out.columns:
        if pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].round(4)
    return out


def write_rank_curve(rank_rows: List[Dict[str, Any]], output_path: Path) -> None:
    df = pd.DataFrame(rank_rows)
    if df.empty:
        return
    df = df.sort_values("rank")
    plt.figure(figsize=(8.2, 5.0), dpi=180)
    plt.plot(df["rank"], df["macro_f1"], marker="o", label="Macro F1", linewidth=2.2)
    plt.plot(df["rank"], df["neutral_f1"], marker="s", label="Neutral F1", linewidth=2.2)
    plt.xticks(df["rank"], ["base" if int(rank) == 0 else f"r{int(rank)}" for rank in df["rank"]])
    plt.xlabel("Model / LoRA rank")
    plt.ylabel("Score")
    plt.title("Rank Sensitivity on Emotion Recognition")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def write_placeholder(path: Path, title: str, stage: str) -> None:
    text = (
        f"# {title}\n\n"
        "This table is reserved for a training/evaluation stage that has not been run in this workspace yet.\n\n"
        f"Run `python run_paper_extension.py --stage {stage}` after confirming compute resources.\n"
    )
    write_text(path, text)


def adapter_size_mb(adapter_dir: Path) -> float | None:
    model_file = adapter_dir / "adapter_model.safetensors"
    if not model_file.exists():
        return None
    return model_file.stat().st_size / (1024 * 1024)


def efficiency_rows(metric_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    root = Path(__file__).resolve().parents[1]
    run_dirs = {
        "r8": root / "runs" / "qwen3vl_2b_emotion_qlora_r8" / "final_adapter",
        "r16": root / "runs" / "rank_sweep" / "qwen3vl_2b_emotion_qlora_r16" / "final_adapter",
        "r20": root / "runs" / "rank_sweep" / "qwen3vl_2b_emotion_qlora_r20" / "final_adapter",
        "r16_neutral_x2": root / "runs" / "qwen3vl_2b_emotion_qlora_r16_neutral_x2" / "final_adapter",
    }
    rows: List[Dict[str, Any]] = []
    for row in metric_rows:
        if row["model"] == "base":
            continue
        cfg_path = Path(str(row["metrics_file"])).with_name("missing")
        adapter_dir = run_dirs.get(row["model"])
        alpha = None
        target_modules = "NA"
        if adapter_dir:
            config_path = adapter_dir.parent / "experiment_config.json"
            if config_path.exists():
                cfg = read_json(config_path)
                alpha = cfg.get("qlora", {}).get("lora_alpha")
                target_modules = ",".join(cfg.get("qlora", {}).get("target_modules", []))
                cfg_path = config_path
        rows.append(
            {
                "model": row["model"],
                "rank": row.get("rank"),
                "alpha": alpha,
                "target_modules": target_modules,
                "trainable_params": "NA",
                "trainable_ratio": "NA",
                "peak_gpu_memory_gb": "NA",
                "train_time_minutes": "NA",
                "adapter_size_mb": round_float(adapter_size_mb(adapter_dir), 2) if adapter_dir else None,
                "accuracy": row["accuracy"],
                "macro_f1": row["macro_f1"],
                "neutral_f1": row["neutral_f1"],
                "config_file": str(cfg_path) if cfg_path.exists() else "NA",
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    ensure_extension_dirs()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    main_rows = existing_metric_rows()
    rank_rows = rank_metric_rows()

    main_df = dataframe_to_markdown(
        output_dir / "main_results_table.md",
        "Main Results: Base vs QLoRA",
        main_rows,
        MAIN_COLUMNS,
    )
    rank_df = dataframe_to_markdown(
        output_dir / "rank_sweep_table.md",
        "Rank Sweep and Fine Sweep",
        rank_rows,
        ["model", "rank", "accuracy", "macro_f1", "positive_f1", "negative_f1", "neutral_f1", "weighted_f1", "valid_json_rate"],
    )
    write_rank_curve(rank_rows, output_dir / "rank_sweep_curve.png")
    if (Path(__file__).resolve().parents[1] / "data_four_datasets" / "test.jsonl").exists():
        note = (
            "# Four-Dataset Result Status\n\n"
            "The active experiment data now includes CMU-MOSEI, CMU-MOSI, CH-SIMSv2, and SIMS in `data_four_datasets/`.\n\n"
            "The existing metric files in `outputs/` were produced before SIMS was added and evaluate the earlier three-dataset split. "
            "Do not cite `main_results_table.md` as four-dataset performance until base and adapter predictions are regenerated with "
            "`configs/emotion_qlora_four_datasets.yaml`.\n"
        )
        write_text(output_dir / "four_dataset_result_status.md", note)

    for filename, title, stage in [
        ("modality_ablation_table.md", "Modality Ablation", "modality_ablation"),
        ("frame_ablation_table.md", "Frame Number Ablation", "frame_ablation"),
        ("cross_dataset_table.md", "Cross-Dataset Generalization", "cross_dataset"),
        ("seed_stability_table.md", "Seed Stability", "seed_stability"),
        ("lora_module_ablation_table.md", "LoRA Module Ablation", "lora_module_ablation"),
        ("alpha_scaling_table.md", "Alpha Scaling", "alpha_scaling"),
    ]:
        if not (output_dir / filename).exists():
            write_placeholder(output_dir / filename, title, stage)

    eff_rows = efficiency_rows(main_rows)
    dataframe_to_markdown(
        output_dir / "efficiency_table.md",
        "Efficiency Summary",
        eff_rows,
        [
            "model",
            "rank",
            "alpha",
            "target_modules",
            "trainable_params",
            "trainable_ratio",
            "peak_gpu_memory_gb",
            "train_time_minutes",
            "adapter_size_mb",
            "accuracy",
            "macro_f1",
            "neutral_f1",
        ],
    )
    write_text(EXT_ROOT / "paper_tables" / "efficiency_summary.md", (output_dir / "efficiency_table.md").read_text(encoding="utf-8"))

    _round_df(main_df).to_csv(output_dir / "main_results_table.csv", index=False, encoding="utf-8-sig")
    _round_df(rank_df).to_csv(output_dir / "rank_sweep_table.csv", index=False, encoding="utf-8-sig")

    index_lines = [
        "# Paper Table Index",
        "",
        "- `main_results_table.md`: base, QLoRA ranks, and neutral oversampling results available now.",
        "- `rank_sweep_table.md` and `rank_sweep_curve.png`: all completed rank sweep metrics.",
        "- `four_dataset_result_status.md`: warns whether the completed metrics match the new four-dataset setup.",
        "- Ablation tables are generated as placeholders until their stages are run.",
        "- `efficiency_table.md`: adapter size is filled from disk; runtime/GPU fields remain `NA` unless logged by future training runs.",
    ]
    write_text(output_dir / "README.md", "\n".join(index_lines) + "\n")
    print(f"Wrote paper tables to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
