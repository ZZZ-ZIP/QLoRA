from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from sklearn.model_selection import train_test_split

from common import REPO_ROOT, ROOT, load_config, normalize_label, resolve_path, write_json, write_jsonl


DEFAULT_DATASETS = {
    "MOSEI": REPO_ROOT / "preprocessed" / "MOSEI" / "manifest.csv",
    "MOSI": REPO_ROOT / "preprocessed" / "MOSI" / "manifest.csv",
    "ch-simsv2s": REPO_ROOT / "preprocessed" / "ch-simsv2s" / "manifest.csv",
    "SIMS": REPO_ROOT / "preprocessed" / "SIMS" / "manifest.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare MOSEI, MOSI, CH-SIMSv2, and SIMS preprocessed manifests for Qwen3-VL QLoRA."
    )
    parser.add_argument("--config", default="configs/emotion_qlora.yaml")
    parser.add_argument("--datasets", nargs="+", default=["MOSEI", "MOSI", "ch-simsv2s", "SIMS"])
    parser.add_argument("--max_frames", type=int, default=4, help="Uniformly sample up to this many frames per sample.")
    parser.add_argument("--val_ratio", type=float, default=-1.0)
    parser.add_argument("--test_ratio", type=float, default=-1.0)
    parser.add_argument("--limit_per_dataset", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sample_frames(paths: List[Path], max_frames: int) -> List[Path]:
    if max_frames <= 0 or len(paths) <= max_frames:
        return paths
    if max_frames == 1:
        return [paths[len(paths) // 2]]
    indexes = [round(i * (len(paths) - 1) / (max_frames - 1)) for i in range(max_frames)]
    return [paths[index] for index in indexes]


def resolve_frame_paths(row: Dict[str, Any], manifest_dir: Path, max_frames: int) -> List[str]:
    raw_paths = [item.strip() for item in str(row.get("frame_paths", "")).split(";") if item.strip()]
    paths: List[Path] = []

    for raw in raw_paths:
        path = Path(raw)
        if not path.is_absolute():
            path = manifest_dir / path
        if path.exists():
            paths.append(path.resolve())

    if not paths:
        frame_dir_raw = str(row.get("frame_dir", "")).strip()
        if frame_dir_raw:
            frame_dir = Path(frame_dir_raw)
            if not frame_dir.is_absolute():
                frame_dir = manifest_dir / frame_dir
            if frame_dir.exists():
                paths = [path.resolve() for path in sorted(frame_dir.glob("frame_*")) if path.is_file()]

    return [str(path) for path in sample_frames(paths, max_frames)]


def load_dataset_manifest(dataset: str, manifest_path: Path, cfg: Dict[str, Any], max_frames: int) -> List[Dict[str, Any]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found for {dataset}: {manifest_path}")

    labels = cfg["task"]["labels"]
    aliases = cfg["task"].get("label_aliases", {})
    records: List[Dict[str, Any]] = []

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = normalize_label(row.get("expected_value", row.get("label", "")), labels, aliases)
            frame_paths = resolve_frame_paths(row, manifest_path.parent, max_frames)
            if not label or not frame_paths:
                continue
            sample_id = str(row.get("sample_id", len(records))).strip()
            records.append(
                {
                    "sample_id": f"{dataset}_{sample_id}",
                    "dataset": dataset,
                    "frame_paths": frame_paths,
                    "label": label,
                    "transcript": str(row.get("transcript", "")).strip(),
                    "prompt_text": str(row.get("prompt_text", "")).strip(),
                    "video_original": str(row.get("video_original", "")).strip(),
                    "source_manifest": str(manifest_path),
                }
            )

    return records


def stratified_split(records: List[Dict[str, Any]], cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, List[Dict[str, Any]]]:
    seed = int(cfg.get("seed", 42))
    val_ratio = args.val_ratio if args.val_ratio >= 0 else float(cfg["data"].get("val_ratio", 0.1))
    test_ratio = args.test_ratio if args.test_ratio >= 0 else float(cfg["data"].get("test_ratio", 0.1))

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["dataset"]), []).append(record)

    result = {"train": [], "val": [], "test": []}
    for dataset, items in grouped.items():
        labels = [item["label"] for item in items]
        counts = pd.Series(labels).value_counts()
        stratify = labels if len(counts) > 1 and counts.min() >= 2 else None
        train_val, test = train_test_split(items, test_size=test_ratio, random_state=seed, stratify=stratify)

        train_val_labels = [item["label"] for item in train_val]
        train_val_counts = pd.Series(train_val_labels).value_counts()
        val_size = val_ratio / max(1e-8, 1.0 - test_ratio)
        stratify_tv = train_val_labels if len(train_val_counts) > 1 and train_val_counts.min() >= 2 else None
        train, val = train_test_split(train_val, test_size=val_size, random_state=seed, stratify=stratify_tv)

        for split_name, split_items in [("train", train), ("val", val), ("test", test)]:
            for item in split_items:
                item["split"] = split_name
            result[split_name].extend(split_items)

    return result


def summarize(splits: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "total": sum(len(items) for items in splits.values()),
        "splits": {},
    }
    for split, items in splits.items():
        summary["splits"][split] = {
            "count": len(items),
            "by_dataset": pd.Series([item["dataset"] for item in items]).value_counts().to_dict() if items else {},
            "by_label": pd.Series([item["label"] for item in items]).value_counts().to_dict() if items else {},
        }
    return summary


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = resolve_path(cfg["data"]["output_dir"], base=ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    for dataset in args.datasets:
        if dataset not in DEFAULT_DATASETS:
            raise ValueError(f"Unsupported dataset '{dataset}'. Supported: {', '.join(DEFAULT_DATASETS)}")
        dataset_records = load_dataset_manifest(dataset, DEFAULT_DATASETS[dataset], cfg, args.max_frames)
        if args.limit_per_dataset:
            dataset_records = dataset_records[: args.limit_per_dataset]
        records.extend(dataset_records)

    if not records:
        raise RuntimeError("No records prepared.")

    splits = stratified_split(records, cfg, args)
    for split, items in splits.items():
        path = output_dir / f"{split}.jsonl"
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists. Use --overwrite to replace it.")
        write_jsonl(path, items)

    summary = summarize(splits)
    write_json(output_dir / "mosei_mosi_chsimsv2_sims_summary.json", summary)
    # Keep the legacy summary filename for scripts that still expect it.
    write_json(output_dir / "mosei_mosi_chsimsv2_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
