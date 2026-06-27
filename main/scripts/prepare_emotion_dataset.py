from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from sklearn.model_selection import train_test_split

from common import ROOT, load_config, normalize_label, resolve_path, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare emotion image manifest for Qwen3-VL QLoRA.")
    parser.add_argument("--config", default="configs/emotion_qlora.yaml")
    parser.add_argument("--input_csv", default="", help="CSV/TSV file containing image_path and label columns.")
    parser.add_argument("--input_json", default="", help="JSON or JSONL manifest.")
    parser.add_argument("--delimiter", default="", help="Optional CSV delimiter. Defaults to auto detection.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_records(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.input_csv:
        path = resolve_path(args.input_csv, base=ROOT)
        sep = args.delimiter or None
        df = pd.read_csv(path, sep=sep, engine="python")
        return df.to_dict("records")

    if args.input_json:
        path = resolve_path(args.input_json, base=ROOT)
        text = path.read_text(encoding="utf-8").strip()
        if path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "data" in data:
            return data["data"]
        raise ValueError("JSON input must be a list or an object with a 'data' list.")

    raise ValueError("Provide --input_csv or --input_json.")


def normalize_records(raw_records: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    data_cfg = cfg["data"]
    labels = cfg["task"]["labels"]
    aliases = cfg["task"].get("label_aliases", {})
    image_col = data_cfg.get("image_column", "image_path")
    label_col = data_cfg.get("label_column", "label")
    id_col = data_cfg.get("id_column", "sample_id")
    split_col = data_cfg.get("split_column", "split")
    subject_col = data_cfg.get("subject_column", "subject_id")

    records: List[Dict[str, Any]] = []
    skipped = 0
    for index, row in enumerate(raw_records):
        image_raw = row.get(image_col, row.get("image", row.get("path", "")))
        label_raw = row.get(label_col, row.get("emotion", row.get("label", "")))
        label = normalize_label(label_raw, labels, aliases)
        if not image_raw or not label:
            skipped += 1
            continue

        image_path = resolve_path(str(image_raw), base=ROOT)
        if not image_path.exists():
            skipped += 1
            continue

        records.append(
            {
                "sample_id": str(row.get(id_col, f"sample_{index:06d}")),
                "image_path": str(image_path),
                "label": label,
                "split": str(row.get(split_col, "")).strip().lower(),
                "source": str(row.get("source", "")),
                "subject_id": str(row.get(subject_col, "")),
            }
        )

    if not records:
        raise RuntimeError(f"No valid records found. Skipped {skipped} rows.")

    return records


def split_records(records: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    predefined = {name: [r for r in records if r.get("split") == name] for name in ["train", "val", "test"]}
    if all(predefined.values()):
        return predefined

    seed = int(cfg.get("seed", 42))
    val_ratio = float(cfg["data"].get("val_ratio", 0.1))
    test_ratio = float(cfg["data"].get("test_ratio", 0.1))

    labels = [r["label"] for r in records]
    stratify = labels if min(pd.Series(labels).value_counts()) >= 2 else None
    train_val, test = train_test_split(records, test_size=test_ratio, random_state=seed, stratify=stratify)

    train_val_labels = [r["label"] for r in train_val]
    val_size = val_ratio / max(1e-8, 1.0 - test_ratio)
    stratify_tv = train_val_labels if min(pd.Series(train_val_labels).value_counts()) >= 2 else None
    train, val = train_test_split(train_val, test_size=val_size, random_state=seed, stratify=stratify_tv)

    for split_name, split_records_ in [("train", train), ("val", val), ("test", test)]:
        for record in split_records_:
            record["split"] = split_name

    return {"train": train, "val": val, "test": test}


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = resolve_path(cfg["data"]["output_dir"], base=ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_records = load_records(args)
    records = normalize_records(raw_records, cfg)
    splits = split_records(records, cfg)

    summary = {
        "total": sum(len(v) for v in splits.values()),
        "splits": {name: len(items) for name, items in splits.items()},
        "labels": {
            name: pd.Series([item["label"] for item in items]).value_counts().to_dict()
            for name, items in splits.items()
        },
    }

    for name, items in splits.items():
        path = output_dir / f"{name}.jsonl"
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists. Use --overwrite to replace it.")
        write_jsonl(path, items)

    write_json(output_dir / "dataset_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

