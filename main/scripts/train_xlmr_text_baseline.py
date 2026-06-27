from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from common import ROOT, read_jsonl, write_json, write_jsonl


LABELS = ["positive", "negative", "neutral"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}


class TextDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], tokenizer: Any, max_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        enc = self.tokenizer(
            str(row.get("transcript", "")),
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(LABEL_TO_ID[str(row.get("label", row.get("gold")))], dtype=torch.long)
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune XLM-R-base text-only baseline.")
    parser.add_argument("--model_name", default="xlm-roberta-base")
    parser.add_argument("--train_file", default="data_four_datasets/train.jsonl")
    parser.add_argument("--val_file", default="data_four_datasets/val.jsonl")
    parser.add_argument("--test_file", default="data_four_datasets/test.jsonl")
    parser.add_argument("--output_dir", default="outputs/baselines/xlmr_text")
    parser.add_argument("--metrics_file", default="outputs/baselines/xlmr_text_metrics.json")
    parser.add_argument("--predictions_file", default="outputs/baselines/xlmr_text_predictions.jsonl")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics_from_preds(y_true: List[str], y_pred: List[str], pred_file: Path) -> Dict[str, Any]:
    return {
        "prediction_file": str(pred_file),
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


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    train_file = resolve(args.train_file)
    val_file = resolve(args.val_file)
    test_file = resolve(args.test_file)
    output_dir = resolve(args.output_dir)
    metrics_file = resolve(args.metrics_file)
    predictions_file = resolve(args.predictions_file)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_jsonl(train_file)
    val_rows = read_jsonl(val_file)
    test_rows = read_jsonl(test_file)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    train_dataset = TextDataset(train_rows, tokenizer, args.max_length)
    val_dataset = TextDataset(val_rows, tokenizer, args.max_length)
    test_dataset = TextDataset(test_rows, tokenizer, args.max_length)

    def compute_metrics(eval_pred: Any) -> Dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        y_true = [ID_TO_LABEL[int(x)] for x in labels]
        y_pred = [ID_TO_LABEL[int(x)] for x in preds]
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0),
        }

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=20,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        fp16=args.fp16,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset, compute_metrics=compute_metrics)
    trainer.train()

    pred_output = trainer.predict(test_dataset)
    pred_ids = np.argmax(pred_output.predictions, axis=-1)
    y_pred = [ID_TO_LABEL[int(x)] for x in pred_ids]
    y_true = [str(r.get("label")) for r in test_rows]
    pred_rows = []
    for row, pred in zip(test_rows, y_pred):
        pred_rows.append(
            {
                "sample_id": row.get("sample_id"),
                "dataset": row.get("dataset"),
                "transcript": row.get("transcript", ""),
                "gold": row.get("label"),
                "pred": pred,
                "raw_output": json.dumps({"emotion": pred}, ensure_ascii=False),
                "model_kind": "xlmr_text",
                "split": "test",
            }
        )
    write_jsonl(predictions_file, pred_rows)
    metrics = metrics_from_preds(y_true, y_pred, predictions_file)
    write_json(metrics_file, metrics)

    print(f"输入 train: {train_file}")
    print(f"输入 val: {val_file}")
    print(f"输入 test: {test_file}")
    print(f"输出 metrics: {metrics_file}")
    print(f"输出 predictions: {predictions_file}")
    print(f"测试样本数: {len(test_rows)}")
    print(f"测试标签分布: {dict(Counter(y_true))}")
    print("主要指标:", json.dumps({k: metrics[k] for k in ["accuracy", "macro_f1", "weighted_f1", "neutral_f1"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
