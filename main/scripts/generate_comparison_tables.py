from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from common import ROOT, load_config, read_jsonl, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final comparison tables for base and QLoRA adapter.")
    parser.add_argument("--config", default="configs/emotion_qlora.yaml")
    parser.add_argument("--base_predictions", default="outputs/base_test_predictions.jsonl")
    parser.add_argument("--adapter_predictions", default="outputs/adapter_test_predictions.jsonl")
    parser.add_argument("--output_dir", default="outputs/tables")
    return parser.parse_args()


def safe_metric(fn, y_true: List[str], y_pred: List[str], default: float = 0.0) -> float:
    if not y_true:
        return default
    return float(fn(y_true, y_pred))


def metric_row(model_name: str, scope: str, records: List[Dict[str, Any]], labels: List[str]) -> Dict[str, Any]:
    y_true = [record["gold"] for record in records]
    y_pred = [record["pred"] for record in records]
    row: Dict[str, Any] = {
        "model": model_name,
        "scope": scope,
        "count": len(records),
        "accuracy": safe_metric(accuracy_score, y_true, y_pred),
        "balanced_accuracy": safe_metric(balanced_accuracy_score, y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0) if records else 0.0,
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0) if records else 0.0,
        "parse_rate": sum(1 for value in y_pred if value != "invalid") / len(y_pred) if y_pred else 0.0,
    }
    for label in labels:
        row[f"f1_{label}"] = f1_score(y_true, y_pred, labels=[label], average="macro", zero_division=0) if records else 0.0
    return row


def build_metric_table(
    base_records: List[Dict[str, Any]],
    adapter_records: List[Dict[str, Any]],
    labels: List[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for model_name, records in [("base", base_records), ("adapter", adapter_records)]:
        rows.append(metric_row(model_name, "all", records, labels))
        datasets = sorted({str(record.get("dataset", "")) for record in records if record.get("dataset", "")})
        for dataset in datasets:
            subset = [record for record in records if record.get("dataset", "") == dataset]
            rows.append(metric_row(model_name, dataset, subset, labels))
    return pd.DataFrame(rows)


def build_delta_table(metric_df: pd.DataFrame, labels: List[str]) -> pd.DataFrame:
    metric_columns = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "parse_rate"]
    metric_columns.extend([f"f1_{label}" for label in labels])

    rows: List[Dict[str, Any]] = []
    for scope in sorted(metric_df["scope"].unique(), key=lambda value: (value != "all", value)):
        base = metric_df[(metric_df["model"] == "base") & (metric_df["scope"] == scope)]
        adapter = metric_df[(metric_df["model"] == "adapter") & (metric_df["scope"] == scope)]
        if base.empty or adapter.empty:
            continue
        row: Dict[str, Any] = {"scope": scope, "count": int(adapter.iloc[0]["count"])}
        for column in metric_columns:
            row[f"delta_{column}"] = float(adapter.iloc[0][column] - base.iloc[0][column])
        rows.append(row)
    return pd.DataFrame(rows)


def build_repair_table(
    base_records: List[Dict[str, Any]],
    adapter_records: List[Dict[str, Any]],
) -> pd.DataFrame:
    base_by_id = {record["sample_id"]: record for record in base_records}
    adapter_by_id = {record["sample_id"]: record for record in adapter_records}
    common_ids = sorted(set(base_by_id) & set(adapter_by_id))

    counters: Dict[str, Counter] = defaultdict(Counter)
    for sample_id in common_ids:
        base = base_by_id[sample_id]
        adapter = adapter_by_id[sample_id]
        gold = base["gold"]
        scope_values = ["all"]
        if base.get("dataset"):
            scope_values.append(str(base["dataset"]))
        base_ok = base["pred"] == gold
        adapter_ok = adapter["pred"] == gold
        if base_ok and adapter_ok:
            bucket = "both_correct"
        elif (not base_ok) and adapter_ok:
            bucket = "repaired"
        elif base_ok and (not adapter_ok):
            bucket = "regressed"
        else:
            bucket = "both_wrong"
        for scope in scope_values:
            counters[scope][bucket] += 1
            counters[scope]["count"] += 1

    rows: List[Dict[str, Any]] = []
    for scope in sorted(counters, key=lambda value: (value != "all", value)):
        counter = counters[scope]
        repaired = int(counter["repaired"])
        regressed = int(counter["regressed"])
        rows.append(
            {
                "scope": scope,
                "count": int(counter["count"]),
                "both_correct": int(counter["both_correct"]),
                "repaired": repaired,
                "regressed": regressed,
                "both_wrong": int(counter["both_wrong"]),
                "repair_minus_regression": repaired - regressed,
                "repair_to_regression_ratio": repaired / regressed if regressed else None,
            }
        )
    return pd.DataFrame(rows)


def round_numeric(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in result.columns:
        if pd.api.types.is_float_dtype(result[column]):
            result[column] = result[column].round(4)
    return result


def write_markdown(path: Path, title: str, tables: Iterable[tuple[str, pd.DataFrame]]) -> None:
    lines = [f"# {title}", ""]
    for section, df in tables:
        lines.extend([f"## {section}", ""])
        lines.append(round_numeric(df).to_markdown(index=False))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    labels = cfg["task"]["labels"]
    output_dir = resolve_path(args.output_dir, base=ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_records = read_jsonl(resolve_path(args.base_predictions, base=ROOT))
    adapter_records = read_jsonl(resolve_path(args.adapter_predictions, base=ROOT))
    if not base_records or not adapter_records:
        raise RuntimeError("Both base and adapter prediction files must be non-empty.")

    metric_df = build_metric_table(base_records, adapter_records, labels)
    delta_df = build_delta_table(metric_df, labels)
    repair_df = build_repair_table(base_records, adapter_records)

    metric_out = output_dir / "metric_comparison.csv"
    delta_out = output_dir / "metric_deltas.csv"
    repair_out = output_dir / "repair_regression.csv"
    markdown_out = output_dir / "final_comparison_tables.md"

    round_numeric(metric_df).to_csv(metric_out, index=False, encoding="utf-8-sig")
    round_numeric(delta_df).to_csv(delta_out, index=False, encoding="utf-8-sig")
    round_numeric(repair_df).to_csv(repair_out, index=False, encoding="utf-8-sig")
    write_markdown(
        markdown_out,
        "Qwen3-VL-2B Emotion QLoRA Comparison",
        [
            ("Metrics", metric_df),
            ("Adapter minus Base", delta_df),
            ("Repair and Regression", repair_df),
        ],
    )

    print(f"Wrote {metric_out}")
    print(f"Wrote {delta_out}")
    print(f"Wrote {repair_out}")
    print(f"Wrote {markdown_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

