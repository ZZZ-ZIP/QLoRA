from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from common import ROOT, load_config, parse_emotion_from_text, read_jsonl, resolve_path, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate emotion predictions.")
    parser.add_argument("--config", default="configs/emotion_qlora.yaml")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output_file", default="")
    parser.add_argument("--output_dir", default="", help="Optional directory for metrics.json/md, report, and confusion matrix.")
    return parser.parse_args()


def _prediction_value(record: Dict[str, Any], labels: List[str], aliases: Dict[str, str]) -> str:
    value = str(record.get("pred", "")).strip()
    if value in labels or value == "invalid":
        return value
    return parse_emotion_from_text(str(record.get("raw_output", value)), labels, aliases)


def _write_confusion_png(matrix: List[List[int]], labels: List[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.4), dpi=160)
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Gold label")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, str(value), ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _write_markdown_outputs(output_dir: Path, metrics: Dict[str, Any], eval_labels: List[str]) -> None:
    report = metrics["classification_report"]
    rows: List[Dict[str, Any]] = []
    for label in eval_labels:
        item = report.get(label, {})
        rows.append(
            {
                "label": label,
                "precision": float(item.get("precision", 0.0)),
                "recall": float(item.get("recall", 0.0)),
                "f1": float(item.get("f1-score", 0.0)),
                "support": int(item.get("support", 0)),
            }
        )
    report_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame(
        [
            {
                "count": metrics["count"],
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "weighted_precision": metrics["weighted_precision"],
                "weighted_recall": metrics["weighted_recall"],
                "weighted_f1": metrics["weighted_f1"],
                "valid_json_rate": metrics["valid_json_rate"],
                "invalid_output_count": metrics["invalid_output_count"],
            }
        ]
    )

    for df in (summary_df, report_df):
        for column in df.columns:
            if pd.api.types.is_float_dtype(df[column]):
                df[column] = df[column].round(4)

    lines = [
        "# Evaluation Metrics",
        "",
        "## Summary",
        "",
        summary_df.to_markdown(index=False),
        "",
        "## Per-Class Report",
        "",
        report_df.to_markdown(index=False),
        "",
        "## Distributions",
        "",
        f"- Gold distribution: `{metrics['gold_distribution']}`",
        f"- Prediction distribution: `{metrics['pred_distribution']}`",
    ]
    (output_dir / "metrics.md").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "classification_report.md").write_text(
        "# Classification Report\n\n" + report_df.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    labels: List[str] = cfg["task"]["labels"]
    aliases: Dict[str, str] = cfg["task"].get("label_aliases", {})
    pred_path = resolve_path(args.predictions, base=ROOT)
    records = read_jsonl(pred_path)
    if not records:
        raise RuntimeError(f"No prediction records in {pred_path}")

    y_true = [r["gold"] for r in records]
    y_pred = [_prediction_value(r, labels, aliases) for r in records]
    eval_labels = labels + ["invalid"]
    cm = confusion_matrix(y_true, y_pred, labels=eval_labels)

    metrics: Dict[str, Any] = {
        "prediction_file": str(pred_path),
        "count": len(records),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "weighted_precision": precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "weighted_recall": recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "parse_rate": sum(1 for value in y_pred if value != "invalid") / len(y_pred),
        "valid_json_rate": sum(1 for value in y_pred if value != "invalid") / len(y_pred),
        "invalid_output_count": sum(1 for value in y_pred if value == "invalid"),
        "gold_distribution": dict(Counter(y_true)),
        "pred_distribution": dict(Counter(y_pred)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=eval_labels,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": {
            "labels": eval_labels,
            "matrix": cm.tolist(),
        },
    }

    if args.output_file:
        output_file = resolve_path(args.output_file, base=ROOT)
    elif args.output_dir:
        output_file = resolve_path(args.output_dir, base=ROOT) / "metrics.json"
    else:
        output_file = pred_path.with_name(pred_path.stem.replace("_predictions", "") + "_metrics.json")
    write_json(output_file, metrics)

    if args.output_dir:
        output_dir = resolve_path(args.output_dir, base=ROOT)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "metrics.json", metrics)
        pd.DataFrame(cm, index=eval_labels, columns=eval_labels).to_csv(
            output_dir / "confusion_matrix.csv", encoding="utf-8-sig"
        )
        _write_confusion_png(cm.tolist(), eval_labels, output_dir / "confusion_matrix.png")
        _write_markdown_outputs(output_dir, metrics, eval_labels)

    print(
        json.dumps(
            {
                k: metrics[k]
                for k in [
                    "count",
                    "accuracy",
                    "balanced_accuracy",
                    "macro_precision",
                    "macro_recall",
                    "macro_f1",
                    "weighted_f1",
                    "valid_json_rate",
                    "invalid_output_count",
                ]
            },
            indent=2,
        )
    )
    print(f"Wrote metrics to {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
