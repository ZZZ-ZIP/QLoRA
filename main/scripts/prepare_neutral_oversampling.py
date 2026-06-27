from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from common import ROOT, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare neutral oversampling training splits.")
    parser.add_argument("--train_file", default="data_four_datasets/train.jsonl")
    parser.add_argument("--val_file", default="data_four_datasets/val.jsonl")
    parser.add_argument("--test_file", default="data_four_datasets/test.jsonl")
    parser.add_argument("--output_dir", default="outputs/neutral_oversampling/data")
    parser.add_argument("--multipliers", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def oversample_neutral(rows: List[Dict[str, Any]], multiplier: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    neutral = [r for r in rows if str(r.get("label")) == "neutral"]
    others = [r for r in rows if str(r.get("label")) != "neutral"]
    expanded = others + neutral * multiplier
    rng.shuffle(expanded)
    return expanded


def main() -> int:
    args = parse_args()
    train_file = resolve(args.train_file)
    val_file = resolve(args.val_file)
    test_file = resolve(args.test_file)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_jsonl(train_file)
    val_rows = read_jsonl(val_file)
    test_rows = read_jsonl(test_file)
    summary = []
    for multiplier in args.multipliers:
        mode_dir = output_dir / f"neutral_x{multiplier}"
        mode_dir.mkdir(parents=True, exist_ok=True)
        sampled = oversample_neutral(train_rows, multiplier, args.seed + multiplier)
        write_jsonl(mode_dir / "train.jsonl", sampled)
        write_jsonl(mode_dir / "val.jsonl", val_rows)
        write_jsonl(mode_dir / "test.jsonl", test_rows)
        summary.append(
            {
                "setting": f"neutral_x{multiplier}",
                "train_count": len(sampled),
                "val_count": len(val_rows),
                "test_count": len(test_rows),
                "train_positive": Counter(r.get("label") for r in sampled).get("positive", 0),
                "train_negative": Counter(r.get("label") for r in sampled).get("negative", 0),
                "train_neutral": Counter(r.get("label") for r in sampled).get("neutral", 0),
                "train_file": str(mode_dir / "train.jsonl"),
                "val_file": str(mode_dir / "val.jsonl"),
                "test_file": str(mode_dir / "test.jsonl"),
            }
        )
    summary_path = output_dir / "neutral_oversampling_data_summary.csv"
    pd.DataFrame(summary).to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"输入 train: {train_file}")
    print(f"输入 val/test 保持不变: {val_file}, {test_file}")
    print(f"输出目录: {output_dir}")
    print(f"采样统计: {summary_path}")
    print(f"原始训练标签分布: {dict(Counter(r.get('label') for r in train_rows))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
