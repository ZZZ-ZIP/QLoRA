from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from common import ROOT, parse_emotion_from_text, read_jsonl


LABELS = ["positive", "negative", "neutral"]
ENGLISH_DATASETS = {"MOSEI", "MOSI"}
CHINESE_DATASETS = {"CHSIMSV2", "CH-SIMSV2", "CH-SIMSV2S", "CH_SIMSV2", "CH_SIMSV2S", "SIMS"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze existing QLoRA prediction files.")
    parser.add_argument("--project_root", default=str(ROOT), help="项目根目录。")
    parser.add_argument("--results_root", default="results_paper_extension", help="已有实验结果目录。")
    parser.add_argument("--test_file", default="data_four_datasets/test.jsonl", help="四数据集 test split。")
    parser.add_argument("--output_analysis_dir", default="outputs/analysis", help="统计表输出目录。")
    parser.add_argument("--output_figures_dir", default="outputs/figures", help="图片输出目录。")
    parser.add_argument("--bootstrap_n", type=int, default=1000, help="bootstrap 重采样次数。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    parser.add_argument("--include_all_predictions", action="store_true", help="是否额外分析 results_root 下扫描到的全部 predictions。")
    return parser.parse_args()


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def normalize_dataset(value: Any) -> str:
    text = str(value or "").strip()
    upper = text.upper().replace("_", "-")
    if upper in {"CH-SIMSV2S", "CHSIMSV2S", "CH-SIMSV2", "CHSIMSV2"}:
        return "CHSIMSv2"
    if upper == "MOSEI":
        return "MOSEI"
    if upper == "MOSI":
        return "MOSI"
    if upper == "SIMS":
        return "SIMS"
    return text or "unknown"


def language_group(dataset: str) -> str:
    upper = normalize_dataset(dataset).upper()
    if upper in ENGLISH_DATASETS:
        return "English"
    if upper in CHINESE_DATASETS:
        return "Chinese"
    return "Unknown"


def normalize_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in LABELS or text == "invalid":
        return text
    return parse_emotion_from_text(text, LABELS, {})


def load_reference_records(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        print(f"[WARN] test_file not found: {path}")
        return {}
    records = read_jsonl(path)
    return {str(r.get("sample_id")): r for r in records if r.get("sample_id")}


def load_predictions(path: Path, refs: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = read_jsonl(path)
    normalized: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        sample_id = str(row.get("sample_id", row.get("id", idx)))
        ref = refs.get(sample_id, {})
        gold = normalize_label(row.get("gold", row.get("label", ref.get("label", ""))))
        pred = normalize_label(row.get("pred", row.get("prediction", row.get("raw_output", ""))))
        if pred == "invalid" and row.get("raw_output"):
            pred = normalize_label(row.get("raw_output"))
        dataset = normalize_dataset(row.get("dataset", ref.get("dataset", "")))
        normalized.append(
            {
                "sample_id": sample_id,
                "dataset": dataset,
                "language_group": language_group(dataset),
                "transcript": row.get("transcript", ref.get("transcript", "")),
                "gold": gold,
                "pred": pred,
                "raw_output": row.get("raw_output", ""),
                "source_file": str(path),
            }
        )
    return normalized


def discover_prediction_files(results_root: Path, include_all: bool) -> Dict[str, Path]:
    rank_root = results_root / "rank_2_32_four_datasets"
    modality_root = results_root / "modality_ablation"
    files: Dict[str, Path] = {}
    candidates = {
        "qwen3vl_qlora_r28": rank_root / "adapter_r28_test_predictions.jsonl",
        "qwen3vl_qlora_r32": rank_root / "adapter_r32_test_predictions.jsonl",
        "r28_video_text": modality_root / "r28_video_text" / "predictions.jsonl",
        "r32_video_text": modality_root / "r32_video_text" / "predictions.jsonl",
        "r28_text_only": modality_root / "r28_text_only" / "predictions.jsonl",
        "r32_text_only": modality_root / "r32_text_only" / "predictions.jsonl",
        "r28_random_frame_text": modality_root / "r28_random_frame_text" / "predictions.jsonl",
        "r32_random_frame_text": modality_root / "r32_random_frame_text" / "predictions.jsonl",
        "r28_vision_only": modality_root / "r28_vision_only" / "predictions.jsonl",
        "r32_vision_only": modality_root / "r32_vision_only" / "predictions.jsonl",
    }
    for name, path in candidates.items():
        if path.exists():
            files[name] = path

    if include_all:
        # 额外扫描所有 predictions，便于发现路径不一致的旧结果。
        for path in results_root.rglob("*predictions*.jsonl"):
            key = path.relative_to(results_root).with_suffix("").as_posix().replace("/", "__")
            files.setdefault(key, path)
    return files


def metric_dict(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    fast = fast_scores(records)
    y_true = [r["gold"] for r in records]
    y_pred = [r["pred"] for r in records]
    return {
        "count": len(records),
        "accuracy": fast["accuracy"],
        "macro_precision": fast["macro_precision"],
        "macro_recall": fast["macro_recall"],
        "macro_f1": fast["macro_f1"],
        "weighted_f1": fast["weighted_f1"],
        "positive_f1": fast["positive_f1"],
        "negative_f1": fast["negative_f1"],
        "neutral_f1": fast["neutral_f1"],
        "valid_json_rate": sum(1 for p in y_pred if p in LABELS) / len(y_pred) if y_pred else 0.0,
        "gold_distribution": json.dumps(dict(Counter(y_true)), ensure_ascii=False),
        "pred_distribution": json.dumps(dict(Counter(y_pred)), ensure_ascii=False),
    }


def fast_scores(records: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not records:
        return {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "positive_f1": 0.0,
            "negative_f1": 0.0,
            "neutral_f1": 0.0,
        }
    total = len(records)
    correct = sum(1 for r in records if r["gold"] == r["pred"])
    per_label: Dict[str, Dict[str, float]] = {}
    for label in LABELS:
        tp = sum(1 for r in records if r["gold"] == label and r["pred"] == label)
        fp = sum(1 for r in records if r["gold"] != label and r["pred"] == label)
        fn = sum(1 for r in records if r["gold"] == label and r["pred"] != label)
        support = sum(1 for r in records if r["gold"] == label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    macro_precision = sum(per_label[label]["precision"] for label in LABELS) / len(LABELS)
    macro_recall = sum(per_label[label]["recall"] for label in LABELS) / len(LABELS)
    macro_f1 = sum(per_label[label]["f1"] for label in LABELS) / len(LABELS)
    support_sum = sum(per_label[label]["support"] for label in LABELS)
    weighted_f1 = sum(per_label[label]["f1"] * per_label[label]["support"] for label in LABELS) / support_sum if support_sum else 0.0
    return {
        "accuracy": correct / total,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "positive_f1": per_label["positive"]["f1"],
        "negative_f1": per_label["negative"]["f1"],
        "neutral_f1": per_label["neutral"]["f1"],
    }


def fast_metric(records: Sequence[Dict[str, Any]], metric_name: str) -> float:
    return fast_scores(records)[metric_name]


def bootstrap_ci(records: Sequence[Dict[str, Any]], metric_fn: Callable[[Sequence[Dict[str, Any]]], float], n: int, seed: int) -> Tuple[float, float]:
    if not records:
        return 0.0, 0.0
    rng = random.Random(seed)
    values: List[float] = []
    size = len(records)
    for _ in range(n):
        sample = [records[rng.randrange(size)] for _ in range(size)]
        values.append(float(metric_fn(sample)))
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def paired_records(a: Sequence[Dict[str, Any]], b: Sequence[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    b_map = {r["sample_id"]: r for r in b}
    pairs = []
    for row in a:
        other = b_map.get(row["sample_id"])
        if other and row["gold"] == other["gold"]:
            pairs.append((row, other))
    return pairs


def paired_bootstrap_diff(
    pairs: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]],
    metric_fn: Callable[[Sequence[Dict[str, Any]]], float],
    n: int,
    seed: int,
) -> Tuple[float, float, float]:
    if not pairs:
        return 0.0, 0.0, 1.0
    rng = random.Random(seed)
    observed = metric_fn([p[0] for p in pairs]) - metric_fn([p[1] for p in pairs])
    diffs: List[float] = []
    size = len(pairs)
    for _ in range(n):
        sample = [pairs[rng.randrange(size)] for _ in range(size)]
        diffs.append(metric_fn([p[0] for p in sample]) - metric_fn([p[1] for p in sample]))
    low, high = np.percentile(diffs, [2.5, 97.5])
    if observed >= 0:
        p_value = 2 * min(sum(1 for d in diffs if d <= 0), sum(1 for d in diffs if d >= 0)) / len(diffs)
    else:
        p_value = 2 * min(sum(1 for d in diffs if d >= 0), sum(1 for d in diffs if d <= 0)) / len(diffs)
    return float(low), float(high), min(float(p_value), 1.0)


def mcnemar_exact(pairs: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]]) -> Tuple[int, int, float]:
    b = 0
    c = 0
    for left, right in pairs:
        left_ok = left["pred"] == left["gold"]
        right_ok = right["pred"] == right["gold"]
        if left_ok and not right_ok:
            b += 1
        elif right_ok and not left_ok:
            c += 1
    n = b + c
    if n == 0:
        return b, c, 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2**n)
    return b, c, min(1.0, 2 * tail)


def write_confusion(records: Sequence[Dict[str, Any]], csv_path: Path, png_path: Path, title: str) -> None:
    y_true = [r["gold"] for r in records]
    y_pred = [r["pred"] for r in records]
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(csv_path, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(5.4, 4.6), dpi=180)
    image = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=25, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Gold label")
    for i, row in enumerate(cm):
        for j, value in enumerate(row):
            ax.text(j, i, str(value), ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)


def grouped_rows(predictions: Dict[str, List[Dict[str, Any]]], group_key: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for setting, records in predictions.items():
        groups = sorted({r[group_key] for r in records})
        for group in groups:
            subset = [r for r in records if r[group_key] == group]
            metrics = metric_dict(subset)
            rows.append({"setting": setting, group_key: group, **metrics})
    return rows


def plot_group_bar(df: pd.DataFrame, group_col: str, output: Path, title: str) -> None:
    focus = df[df["setting"].isin(["qwen3vl_qlora_r28", "qwen3vl_qlora_r32", "r28_text_only", "r32_text_only"])]
    if focus.empty:
        focus = df.copy()
    pivot = focus.pivot_table(index=group_col, columns="setting", values="macro_f1", aggfunc="first")
    ax = pivot.plot(kind="bar", figsize=(7.6, 4.4), width=0.78)
    ax.set_title(title)
    ax.set_ylabel("Macro-F1")
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    ax.legend(loc="best", fontsize=8)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def write_neutral_errors(records_by_setting: Dict[str, List[Dict[str, Any]]], output: Path) -> None:
    rows: List[Dict[str, Any]] = []
    for setting in ["qwen3vl_qlora_r28", "qwen3vl_qlora_r32"]:
        for row in records_by_setting.get(setting, []):
            if row["gold"] != row["pred"] and ("neutral" in {row["gold"], row["pred"]}):
                rows.append(
                    {
                        "setting": setting,
                        "sample_id": row["sample_id"],
                        "dataset": row["dataset"],
                        "language_group": row["language_group"],
                        "transcript": row["transcript"],
                        "gold_label": row["gold"],
                        "predicted_label": row["pred"],
                        "error_type": f"{row['gold']}->{row['pred']}",
                    }
                )
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)


def main() -> int:
    args = parse_args()
    root = Path(args.project_root).resolve()
    results_root = resolve(root, args.results_root)
    test_file = resolve(root, args.test_file)
    analysis_dir = resolve(root, args.output_analysis_dir)
    figures_dir = resolve(root, args.output_figures_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    refs = load_reference_records(test_file)
    files = discover_prediction_files(results_root, args.include_all_predictions)
    predictions: Dict[str, List[Dict[str, Any]]] = {}
    for setting, path in sorted(files.items()):
        try:
            rows = load_predictions(path, refs)
            if rows:
                predictions[setting] = rows
        except Exception as exc:
            print(f"[WARN] skip {path}: {exc}")

    metric_rows: List[Dict[str, Any]] = []
    ci_focus = {"qwen3vl_qlora_r28", "qwen3vl_qlora_r32", "r28_text_only", "r32_text_only", "r28_video_text", "r32_video_text"}
    for setting, records in predictions.items():
        metrics = metric_dict(records)
        row = {"setting": setting, "prediction_file": records[0]["source_file"], **metrics}
        if setting in ci_focus:
            for metric_name, fn in [
                ("accuracy", lambda rs: fast_metric(rs, "accuracy")),
                ("macro_f1", lambda rs: fast_metric(rs, "macro_f1")),
                ("neutral_f1", lambda rs: fast_metric(rs, "neutral_f1")),
            ]:
                low, high = bootstrap_ci(records, fn, args.bootstrap_n, args.seed + len(setting) + len(metric_name))
                row[f"{metric_name}_ci_low"] = low
                row[f"{metric_name}_ci_high"] = high
        metric_rows.append(row)

    pd.DataFrame(metric_rows).sort_values("setting").to_csv(analysis_dir / "main_metrics_with_ci.csv", index=False, encoding="utf-8-sig")

    dataset_df = pd.DataFrame(grouped_rows(predictions, "dataset"))
    lang_df = pd.DataFrame(grouped_rows(predictions, "language_group"))
    dataset_df.to_csv(analysis_dir / "dataset_wise_results.csv", index=False, encoding="utf-8-sig")
    lang_df.to_csv(analysis_dir / "language_group_results.csv", index=False, encoding="utf-8-sig")

    if "qwen3vl_qlora_r28" in predictions:
        write_confusion(predictions["qwen3vl_qlora_r28"], analysis_dir / "confusion_matrix_r28.csv", figures_dir / "confusion_matrix_r28.png", "QLoRA r28")
    if "qwen3vl_qlora_r32" in predictions:
        write_confusion(predictions["qwen3vl_qlora_r32"], analysis_dir / "confusion_matrix_r32.csv", figures_dir / "confusion_matrix_r32.png", "QLoRA r32")

    if not dataset_df.empty:
        plot_group_bar(dataset_df, "dataset", figures_dir / "dataset_wise_macro_f1.png", "Dataset-wise Macro-F1")
    if not lang_df.empty:
        plot_group_bar(lang_df, "language_group", figures_dir / "language_group_macro_f1.png", "Language-group Macro-F1")

    comparisons = [
        ("r28_vs_r32", "qwen3vl_qlora_r28", "qwen3vl_qlora_r32"),
        ("r28_video_text_vs_text_only", "r28_video_text", "r28_text_only"),
        ("r32_video_text_vs_text_only", "r32_video_text", "r32_text_only"),
    ]
    sig_rows: List[Dict[str, Any]] = []
    for name, left_name, right_name in comparisons:
        if left_name not in predictions or right_name not in predictions:
            sig_rows.append({"comparison": name, "metric": "macro_f1", "method": "paired_bootstrap", "p_value": "", "significant": "", "note": "missing prediction file"})
            continue
        pairs = paired_records(predictions[left_name], predictions[right_name])
        for metric_name, fn in [
            ("accuracy", lambda rs: fast_metric(rs, "accuracy")),
            ("macro_f1", lambda rs: fast_metric(rs, "macro_f1")),
            ("neutral_f1", lambda rs: fast_metric(rs, "neutral_f1")),
        ]:
            low, high, p_value = paired_bootstrap_diff(pairs, fn, args.bootstrap_n, args.seed + len(name) + len(metric_name))
            observed = fn([p[0] for p in pairs]) - fn([p[1] for p in pairs]) if pairs else 0.0
            sig_rows.append(
                {
                    "comparison": name,
                    "metric": metric_name,
                    "method": "paired_bootstrap",
                    "effect_left_minus_right": observed,
                    "ci_low": low,
                    "ci_high": high,
                    "p_value": p_value,
                    "significant": p_value < 0.05,
                    "note": f"{left_name} - {right_name}; paired_n={len(pairs)}",
                }
            )
        b, c, p_value = mcnemar_exact(pairs)
        sig_rows.append(
            {
                "comparison": name,
                "metric": "accuracy",
                "method": "mcnemar_exact_binomial",
                "effect_left_minus_right": "",
                "ci_low": "",
                "ci_high": "",
                "p_value": p_value,
                "significant": p_value < 0.05,
                "note": f"left_correct_only={b}; right_correct_only={c}; paired_n={len(pairs)}",
            }
        )
    sig_df = pd.DataFrame(sig_rows)
    sig_df.to_csv(analysis_dir / "significance_tests.csv", index=False, encoding="utf-8-sig")
    tables_dir = root / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    sig_df.to_csv(tables_dir / "significance_table.csv", index=False, encoding="utf-8-sig")

    write_neutral_errors(predictions, analysis_dir / "neutral_error_cases.csv")

    print(f"输入结果目录: {results_root}")
    print(f"测试集文件: {test_file}")
    print(f"输出统计目录: {analysis_dir}")
    print(f"输出图片目录: {figures_dir}")
    print(f"已分析预测文件数: {len(predictions)}")
    if "qwen3vl_qlora_r28" in predictions:
        print("r28 主要指标:", json.dumps(metric_dict(predictions["qwen3vl_qlora_r28"]), ensure_ascii=False, indent=2)[:800])
    if "qwen3vl_qlora_r32" in predictions:
        print("r32 主要指标:", json.dumps(metric_dict(predictions["qwen3vl_qlora_r32"]), ensure_ascii=False, indent=2)[:800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
