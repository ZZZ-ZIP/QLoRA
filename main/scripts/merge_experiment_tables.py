from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from common import ROOT, read_jsonl


LABELS = ["positive", "negative", "neutral"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge completed metrics into paper tables.")
    parser.add_argument("--project_root", default=str(ROOT))
    parser.add_argument("--config", default="configs/emotion_qlora_four_datasets.yaml")
    parser.add_argument("--output_tables_dir", default="outputs/tables")
    parser.add_argument("--output_figures_dir", default="outputs/figures")
    return parser.parse_args()


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def load_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def metric_row(method: str, modality: str, fine_tuning: str, trainable_params: str, metric_path: Path, valid_json_rate: str | float = "") -> Dict[str, Any]:
    metrics = load_json(metric_path)
    row: Dict[str, Any] = {
        "method": method,
        "modality": modality,
        "fine_tuning": fine_tuning,
        "trainable_params": trainable_params,
        "accuracy": "",
        "macro_f1": "",
        "weighted_f1": "",
        "positive_f1": "",
        "negative_f1": "",
        "neutral_f1": "",
        "valid_json_rate": valid_json_rate,
        "status": "not_completed",
        "metrics_file": str(metric_path),
    }
    if not metrics:
        return row
    report = metrics.get("classification_report", {})
    row.update(
        {
            "accuracy": metrics.get("accuracy", ""),
            "macro_f1": metrics.get("macro_f1", ""),
            "weighted_f1": metrics.get("weighted_f1", ""),
            "positive_f1": metrics.get("positive_f1", report.get("positive", {}).get("f1-score", "")),
            "negative_f1": metrics.get("negative_f1", report.get("negative", {}).get("f1-score", "")),
            "neutral_f1": metrics.get("neutral_f1", report.get("neutral", {}).get("f1-score", "")),
            "valid_json_rate": metrics.get("valid_json_rate", valid_json_rate),
            "status": "completed",
        }
    )
    return row


def split_count(path: Path) -> Dict[str, Counter]:
    rows = read_jsonl(path) if path.exists() else []
    return {
        "dataset": Counter(str(r.get("dataset")) for r in rows),
        "label": Counter(str(r.get("label")) for r in rows),
    }


def write_dataset_label_mapping(root: Path, out: Path) -> None:
    train = read_jsonl(root / "data_four_datasets" / "train.jsonl")
    val = read_jsonl(root / "data_four_datasets" / "val.jsonl")
    test = read_jsonl(root / "data_four_datasets" / "test.jsonl")
    rows = []
    datasets = sorted({str(r.get("dataset")) for r in train + val + test})
    for dataset in datasets:
        rows.append(
            {
                "dataset": dataset,
                "original_label_type": "preprocessed sentiment/emotion label",
                "positive_mapping_rule": "统一为 positive",
                "negative_mapping_rule": "统一为 negative",
                "neutral_mapping_rule": "统一为 neutral",
                "train_count": sum(1 for r in train if str(r.get("dataset")) == dataset),
                "val_count": sum(1 for r in val if str(r.get("dataset")) == dataset),
                "test_count": sum(1 for r in test if str(r.get("dataset")) == dataset),
            }
        )
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")


def write_hyperparameters(root: Path, config_path: Path, out: Path) -> None:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    train = cfg["training"]
    qlora = cfg["qlora"]
    gen = cfg["generation"]
    base = {
        "base_model": cfg["model"]["base_model"],
        "quantization": f"4-bit {qlora.get('bnb_4bit_quant_type', 'nf4')}, double_quant={qlora.get('bnb_4bit_use_double_quant', True)}",
        "compute_dtype": qlora.get("bnb_4bit_compute_dtype", "bfloat16"),
        "target_modules": ",".join(qlora.get("target_modules", [])),
        "batch_size": train.get("per_device_train_batch_size", 1),
        "gradient_accumulation": train.get("gradient_accumulation_steps", ""),
        "learning_rate": train.get("learning_rate", ""),
        "epochs": train.get("num_train_epochs", ""),
        "scheduler": train.get("lr_scheduler_type", "not recorded"),
        "warmup_ratio": train.get("warmup_ratio", ""),
        "max_new_tokens": gen.get("max_new_tokens", ""),
        "decoding_strategy": "greedy" if not gen.get("do_sample", False) else "sampling",
        "gpu": "RTX 5060 Ti 16GB",
        "trainable_parameters": "根据训练日志填写",
        "trainable_ratio": "根据训练日志填写",
    }
    rows = []
    for rank, alpha in [(28, 64), (32, 64)]:
        row = dict(base)
        row["lora_rank"] = rank
        row["lora_alpha"] = alpha
        rows.append(row)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")


def placeholder_figure(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=180)
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=13, weight="bold")
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=10, wrap=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_summary_metric(csv_path: Path, fig_path: Path, title: str, metric: str) -> None:
    if not csv_path.exists():
        placeholder_figure(fig_path, title, "Not completed: no metrics are available for plotting.")
        return
    df = pd.read_csv(csv_path)
    if metric not in df.columns or df[metric].dropna().empty:
        placeholder_figure(fig_path, title, "Not completed or metric missing. Mark this item as future work.")
        return
    ax = df.plot(x=df.columns[0], y=metric, kind="bar", legend=False, figsize=(6.4, 3.8))
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()


def write_report(root: Path, tables_dir: Path, figures_dir: Path) -> None:
    main_table = pd.read_csv(tables_dir / "main_comparison_table.csv")
    completed = main_table[main_table["status"] == "completed"]
    pending = main_table[main_table["status"] != "completed"]
    lines = [
        "# Experiment Update Report",
        "",
        "## 已生成的新增材料",
        "",
        "- `outputs/analysis/`: 已有预测的统一指标、bootstrap 置信区间、配对显著性检验、数据集/语言分组统计、neutral 错误案例。",
        "- `outputs/baselines/`: TF-IDF + LR/SVM 轻量文本基线的 metrics 与 predictions（若已运行）。",
        "- `outputs/tables/`: 论文实验表格 CSV，包括主对照、超参数、数据集标签映射和显著性表。",
        "- `outputs/figures/`: 混淆矩阵、数据集/语言分组图，以及未完成实验的明确占位图。",
        "",
        "## 已完成与未完成",
        "",
        f"- 已完成并可写入论文主表的方法数：{len(completed)}。",
        f"- 尚未完成或需要后续运行的方法数：{len(pending)}。",
        "",
        "未完成项不得在论文中写成有效结论，只能作为局限性或未来工作。",
        "",
        "## 可写入论文的实验分析摘要",
        "",
        "已有结果支持将 QLoRA-r28/r32 作为四数据集多模态情绪识别的主要对照，并进一步从模态消融、跨数据集泛化、多 seed 稳定性、LoRA 模块消融、alpha scaling 和 neutral 错误分析几个角度组织实验章节。对于未实际运行的 XLM-R、Qwen3-VL zero/few-shot、帧数消融和 neutral oversampling 结果，论文中应明确标注为未完成或未来工作，不能推断其优劣。",
        "",
        "从已有 QLoRA 结果看，neutral 类仍是主要瓶颈，应结合混淆矩阵和 `neutral_error_cases.csv` 讨论其被吸收到 positive/negative 的模式。video+text 相比 text-only 的收益需要以 `significance_tests.csv` 中的配对检验为准；若 p 值不显著，应使用“观察到提升趋势”而非“显著提升”。",
        "",
        "## 关键文件",
        "",
        f"- 主对照表：`{tables_dir / 'main_comparison_table.csv'}`",
        f"- 超参数表：`{tables_dir / 'training_hyperparameters.csv'}`",
        f"- 数据映射表：`{tables_dir / 'dataset_label_mapping.csv'}`",
        f"- 显著性表：`{tables_dir / 'significance_table.csv'}`",
        f"- 图目录：`{figures_dir}`",
    ]
    (root / "experiment_update_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = Path(args.project_root).resolve()
    tables_dir = resolve(root, args.output_tables_dir)
    figures_dir = resolve(root, args.output_figures_dir)
    config_path = resolve(root, args.config)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rank_root = root / "results_paper_extension" / "rank_2_32_four_datasets"
    baselines = root / "outputs" / "baselines"
    rows = [
        metric_row("TF-IDF + Logistic Regression", "text", "none", "0", baselines / "tfidf_lr_metrics.json", 1.0),
        metric_row("TF-IDF + Linear SVM", "text", "none", "0", baselines / "tfidf_svm_metrics.json", 1.0),
        metric_row("XLM-R-base", "text", "full text encoder fine-tuning", "待训练后填写", baselines / "xlmr_text_metrics.json", 1.0),
        metric_row("Qwen3-VL zero-shot text-only", "text", "none", "0", baselines / "qwen3vl_zeroshot_textonly_metrics.json", ""),
        metric_row("Qwen3-VL zero-shot video+text", "video+text", "none", "0", baselines / "qwen3vl_zeroshot_videotext_metrics.json", ""),
        metric_row("Qwen3-VL few-shot video+text", "video+text", "none", "0", baselines / "qwen3vl_fewshot_videotext_metrics.json", ""),
        metric_row("Qwen3-VL QLoRA-r28", "video+text", "QLoRA", "根据训练日志填写", rank_root / "adapter_r28_test_metrics.json", ""),
        metric_row("Qwen3-VL QLoRA-r32", "video+text", "QLoRA", "根据训练日志填写", rank_root / "adapter_r32_test_metrics.json", ""),
    ]
    main_df = pd.DataFrame(rows)
    for col in ["accuracy", "macro_f1", "weighted_f1", "positive_f1", "negative_f1", "neutral_f1", "valid_json_rate"]:
        converted = pd.to_numeric(main_df[col], errors="coerce")
        main_df[col] = converted.where(converted.notna(), main_df[col])
    main_df.to_csv(tables_dir / "main_comparison_table.csv", index=False, encoding="utf-8-sig")

    write_dataset_label_mapping(root, tables_dir / "dataset_label_mapping.csv")
    write_hyperparameters(root, config_path, tables_dir / "training_hyperparameters.csv")

    sig_src = root / "outputs" / "analysis" / "significance_tests.csv"
    sig_dst = tables_dir / "significance_table.csv"
    if sig_src.exists():
        pd.read_csv(sig_src).to_csv(sig_dst, index=False, encoding="utf-8-sig")
    elif not sig_dst.exists():
        pd.DataFrame([{"comparison": "not_available", "metric": "", "method": "", "p_value": "", "significant": "", "note": "先运行 analyze_existing_predictions.py"}]).to_csv(sig_dst, index=False, encoding="utf-8-sig")

    frame_summary = root / "outputs" / "frame_ablation" / "frame_ablation_summary.csv"
    neutral_summary = root / "outputs" / "neutral_oversampling" / "neutral_oversampling_summary.csv"
    if not frame_summary.exists():
        pd.DataFrame(
            [
                {"mode": "frame1", "status": "not_completed"},
                {"mode": "frame2", "status": "not_completed"},
                {"mode": "frame4", "status": "not_completed"},
                {"mode": "random_frame2", "status": "not_completed"},
            ]
        ).to_csv(frame_summary, index=False, encoding="utf-8-sig")
    if not neutral_summary.exists():
        neutral_summary.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"setting": "neutral_x2", "status": "not_completed"}, {"setting": "neutral_x3", "status": "not_completed"}]).to_csv(neutral_summary, index=False, encoding="utf-8-sig")

    plot_summary_metric(frame_summary, figures_dir / "frame_ablation_macro_f1.png", "Frame Ablation Macro-F1", "macro_f1")
    plot_summary_metric(frame_summary, figures_dir / "frame_ablation_neutral_f1.png", "Frame Ablation Neutral-F1", "neutral_f1")
    plot_summary_metric(neutral_summary, figures_dir / "neutral_oversampling_comparison.png", "Neutral Oversampling Comparison", "neutral_f1")

    write_report(root, tables_dir, figures_dir)

    print(f"输出主表: {tables_dir / 'main_comparison_table.csv'}")
    print(f"输出超参数表: {tables_dir / 'training_hyperparameters.csv'}")
    print(f"输出数据映射表: {tables_dir / 'dataset_label_mapping.csv'}")
    print(f"输出显著性表: {tables_dir / 'significance_table.csv'}")
    print(f"输出报告: {root / 'experiment_update_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
