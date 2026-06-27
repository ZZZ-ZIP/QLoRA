from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from common import ROOT, read_jsonl, write_json, write_jsonl


LABELS = ["positive", "negative", "neutral"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TF-IDF text-only baselines.")
    parser.add_argument("--train_file", default="data_four_datasets/train.jsonl")
    parser.add_argument("--val_file", default="data_four_datasets/val.jsonl")
    parser.add_argument("--test_file", default="data_four_datasets/test.jsonl")
    parser.add_argument("--output_dir", default="outputs/baselines")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_word_features", type=int, default=30000)
    parser.add_argument("--max_char_features", type=int, default=50000)
    parser.add_argument("--svm_c", type=float, default=1.0)
    parser.add_argument("--lr_c", type=float, default=2.0)
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def load_split(path: Path) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    rows = read_jsonl(path)
    texts = [str(r.get("transcript", "")).strip() for r in rows]
    labels = [str(r.get("label", r.get("gold", ""))).strip() for r in rows]
    return texts, labels, rows


def build_features(max_word_features: int, max_char_features: int) -> FeatureUnion:
    # 中文和英文混合语料同时使用词 n-gram 与字符 n-gram，避免依赖额外分词器。
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    lowercase=True,
                    min_df=1,
                    max_features=max_word_features,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 5),
                    lowercase=True,
                    min_df=1,
                    max_features=max_char_features,
                ),
            ),
        ]
    )


def compute_metrics(y_true: List[str], y_pred: List[str], prediction_file: Path) -> Dict[str, Any]:
    return {
        "prediction_file": str(prediction_file),
        "count": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0),
        "positive_f1": f1_score(y_true, y_pred, labels=["positive"], average="macro", zero_division=0),
        "negative_f1": f1_score(y_true, y_pred, labels=["negative"], average="macro", zero_division=0),
        "neutral_f1": f1_score(y_true, y_pred, labels=["neutral"], average="macro", zero_division=0),
        "valid_json_rate": 1.0,
        "gold_distribution": dict(Counter(y_true)),
        "pred_distribution": dict(Counter(y_pred)),
        "classification_report": classification_report(y_true, y_pred, labels=LABELS, zero_division=0, output_dict=True),
    }


def save_predictions(path: Path, rows: List[Dict[str, Any]], preds: List[str], model_name: str) -> None:
    output = []
    for row, pred in zip(rows, preds):
        output.append(
            {
                "sample_id": row.get("sample_id"),
                "dataset": row.get("dataset"),
                "transcript": row.get("transcript", ""),
                "gold": row.get("label", row.get("gold")),
                "pred": pred,
                "raw_output": json.dumps({"emotion": pred}, ensure_ascii=False),
                "model_kind": model_name,
                "split": row.get("split", "test"),
            }
        )
    write_jsonl(path, output)


def run_model(
    name: str,
    classifier: Any,
    train_x: List[str],
    train_y: List[str],
    test_x: List[str],
    test_y: List[str],
    test_rows: List[Dict[str, Any]],
    out_dir: Path,
    max_word_features: int,
    max_char_features: int,
) -> Dict[str, Any]:
    pipeline = Pipeline(
        [
            ("tfidf", build_features(max_word_features, max_char_features)),
            ("clf", classifier),
        ]
    )
    pipeline.fit(train_x, train_y)
    preds = list(pipeline.predict(test_x))
    pred_path = out_dir / f"{name}_predictions.jsonl"
    metric_path = out_dir / f"{name}_metrics.json"
    save_predictions(pred_path, test_rows, preds, name)
    metrics = compute_metrics(test_y, preds, pred_path)
    write_json(metric_path, metrics)
    return metrics


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_file = resolve(args.train_file)
    val_file = resolve(args.val_file)
    test_file = resolve(args.test_file)
    out_dir = resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_x, train_y, train_rows = load_split(train_file)
    val_x, val_y, _ = load_split(val_file)
    test_x, test_y, test_rows = load_split(test_file)

    # 轻量基线不调参，val 只用于记录同一划分确实存在；训练仍只使用 train。
    print(f"输入 train: {train_file}")
    print(f"输入 val: {val_file} ({len(val_x)} samples)")
    print(f"输入 test: {test_file}")
    print(f"输出目录: {out_dir}")
    print(f"训练样本数: {len(train_rows)}; 测试样本数: {len(test_rows)}")
    print(f"训练标签分布: {dict(Counter(train_y))}")
    print(f"测试标签分布: {dict(Counter(test_y))}")

    lr_metrics = run_model(
        "tfidf_lr",
        LogisticRegression(C=args.lr_c, max_iter=2000, class_weight="balanced", random_state=args.seed, solver="liblinear"),
        train_x,
        train_y,
        test_x,
        test_y,
        test_rows,
        out_dir,
        args.max_word_features,
        args.max_char_features,
    )
    svm_metrics = run_model(
        "tfidf_svm",
        LinearSVC(C=args.svm_c, class_weight="balanced", random_state=args.seed),
        train_x,
        train_y,
        test_x,
        test_y,
        test_rows,
        out_dir,
        args.max_word_features,
        args.max_char_features,
    )

    print("TF-IDF + LR 主要指标:", json.dumps({k: lr_metrics[k] for k in ["accuracy", "macro_f1", "weighted_f1", "neutral_f1"]}, indent=2))
    print("TF-IDF + SVM 主要指标:", json.dumps({k: svm_metrics[k] for k in ["accuracy", "macro_f1", "weighted_f1", "neutral_f1"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
