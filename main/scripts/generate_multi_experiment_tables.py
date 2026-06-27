from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from common import ROOT, load_config, read_jsonl, resolve_path


DEFAULT_PREDICTIONS = [
    "base=outputs/base_test_predictions.jsonl",
    "r8=outputs/adapter_test_predictions.jsonl",
    "r16=outputs/adapter_r16_test_predictions.jsonl",
    "r8_neutral_x2=outputs/adapter_r8_neutral_x2_test_predictions.jsonl",
    "r16_neutral_x2=outputs/adapter_r16_neutral_x2_test_predictions.jsonl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multi-experiment comparison tables.")
    parser.add_argument("--config", default="configs/emotion_qlora.yaml")
    parser.add_argument("--prediction", action="append", default=[], help="Format: name=path/to/predictions.jsonl")
    parser.add_argument("--reference", default="r8", help="Reference model for delta columns.")
    parser.add_argument("--output_dir", default="outputs/tables_multi")
    return parser.parse_args()


def parse_prediction_specs(specs: List[str]) -> List[Tuple[str, Path]]:
    parsed: List[Tuple[str, Path]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Prediction spec must be name=path, got: {spec}")
        name, path = spec.split("=", 1)
        parsed.append((name.strip(), resolve_path(path.strip(), base=ROOT)))
    return parsed


def metric_row(model_name: str, scope: str, records: List[Dict[str, Any]], labels: List[str]) -> Dict[str, Any]:
    y_true = [record["gold"] for record in records]
    y_pred = [record["pred"] for record in records]
    row: Dict[str, Any] = {
        "model": model_name,
        "scope": scope,
        "count": len(records),
        "accuracy": accuracy_score(y_true, y_pred) if records else 0.0,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred) if records else 0.0,
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0) if records else 0.0,
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0) if records else 0.0,
        "parse_rate": sum(1 for value in y_pred if value != "invalid") / len(y_pred) if y_pred else 0.0,
    }
    for label in labels:
        row[f"f1_{label}"] = f1_score(y_true, y_pred, labels=[label], average="macro", zero_division=0) if records else 0.0
    return row


def build_metrics(all_records: Dict[str, List[Dict[str, Any]]], labels: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for name, records in all_records.items():
        rows.append(metric_row(name, "all", records, labels))
        datasets = sorted({str(record.get("dataset", "")) for record in records if record.get("dataset")})
        for dataset in datasets:
            subset = [record for record in records if record.get("dataset") == dataset]
            rows.append(metric_row(name, dataset, subset, labels))
    return pd.DataFrame(rows)


def build_deltas(metrics: pd.DataFrame, reference: str, labels: List[str]) -> pd.DataFrame:
    metric_columns = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "parse_rate"]
    metric_columns.extend([f"f1_{label}" for label in labels])

    rows: List[Dict[str, Any]] = []
    for scope in sorted(metrics["scope"].unique(), key=lambda value: (value != "all", value)):
        ref = metrics[(metrics["model"] == reference) & (metrics["scope"] == scope)]
        if ref.empty:
            continue
        for model in sorted(metrics["model"].unique()):
            if model == reference:
                continue
            current = metrics[(metrics["model"] == model) & (metrics["scope"] == scope)]
            if current.empty:
                continue
            row: Dict[str, Any] = {"model": model, "reference": reference, "scope": scope}
            for column in metric_columns:
                row[f"delta_{column}"] = float(current.iloc[0][column] - ref.iloc[0][column])
            rows.append(row)
    return pd.DataFrame(rows)


def build_repair_against_reference(all_records: Dict[str, List[Dict[str, Any]]], reference: str) -> pd.DataFrame:
    if reference not in all_records:
        return pd.DataFrame()
    ref_by_id = {record["sample_id"]: record for record in all_records[reference]}
    rows: List[Dict[str, Any]] = []

    for model, records in all_records.items():
        if model == reference:
            continue
        current_by_id = {record["sample_id"]: record for record in records}
        common_ids = sorted(set(ref_by_id) & set(current_by_id))
        counters: Dict[str, Counter] = defaultdict(Counter)
        for sample_id in common_ids:
            ref = ref_by_id[sample_id]
            current = current_by_id[sample_id]
            gold = ref["gold"]
            ref_ok = ref["pred"] == gold
            current_ok = current["pred"] == gold
            if ref_ok and current_ok:
                bucket = "both_correct"
            elif (not ref_ok) and current_ok:
                bucket = "repaired"
            elif ref_ok and (not current_ok):
                bucket = "regressed"
            else:
                bucket = "both_wrong"
            scopes = ["all"]
            if ref.get("dataset"):
                scopes.append(str(ref["dataset"]))
            for scope in scopes:
                counters[scope][bucket] += 1
                counters[scope]["count"] += 1

        for scope, counter in counters.items():
            repaired = int(counter["repaired"])
            regressed = int(counter["regressed"])
            rows.append(
                {
                    "model": model,
                    "reference": reference,
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
    return pd.DataFrame(rows).sort_values(["scope", "model"]) if rows else pd.DataFrame()


def round_numeric(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in result.columns:
        if pd.api.types.is_float_dtype(result[column]):
            result[column] = result[column].round(4)
    return result


def write_markdown(path: Path, tables: List[Tuple[str, pd.DataFrame]]) -> None:
    lines = ["# Qwen3-VL-2B Neutral and r16 Ablation Tables", ""]
    for title, df in tables:
        lines.extend([f"## {title}", ""])
        if df.empty:
            lines.append("_No rows._")
        else:
            lines.append(round_numeric(df).to_markdown(index=False))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    labels = cfg["task"]["labels"]
    specs = args.prediction if args.prediction else DEFAULT_PREDICTIONS
    output_dir = resolve_path(args.output_dir, base=ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records: Dict[str, List[Dict[str, Any]]] = {}
    missing: List[str] = []
    for name, path in parse_prediction_specs(specs):
        if not path.exists():
            missing.append(f"{name}={path}")
            continue
        all_records[name] = read_jsonl(path)

    if missing:
        print("Skipped missing prediction files:")
        for item in missing:
            print(f"  {item}")
    if len(all_records) < 2:
        raise RuntimeError("Need at least two existing prediction files for comparison.")

    metrics = build_metrics(all_records, labels)
    deltas = build_deltas(metrics, args.reference, labels)
    repairs = build_repair_against_reference(all_records, args.reference)

    round_numeric(metrics).to_csv(output_dir / "multi_metric_comparison.csv", index=False, encoding="utf-8-sig")
    round_numeric(deltas).to_csv(output_dir / "multi_metric_deltas.csv", index=False, encoding="utf-8-sig")
    round_numeric(repairs).to_csv(output_dir / "multi_repair_regression.csv", index=False, encoding="utf-8-sig")
    write_markdown(
        output_dir / "multi_final_tables.md",
        [
            ("Metrics", metrics),
            (f"Deltas vs {args.reference}", deltas),
            (f"Repair and Regression vs {args.reference}", repairs),
        ],
    )
    print(f"Wrote tables under {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

