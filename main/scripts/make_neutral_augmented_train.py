from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from common import ROOT, load_config, read_jsonl, resolve_path, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a neutral-oversampled training JSONL.")
    parser.add_argument("--config", default="configs/emotion_qlora.yaml")
    parser.add_argument("--input_train", default="data/train.jsonl")
    parser.add_argument("--output_train", default="data/train_neutral_x2.jsonl")
    parser.add_argument("--target_label", default="neutral")
    parser.add_argument("--factor", type=int, default=2, help="Total factor for target label. 2 means duplicate once.")
    parser.add_argument("--seed", type=int, default=-1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42)) if args.seed < 0 else args.seed
    random.seed(seed)

    input_train = resolve_path(args.input_train, base=ROOT)
    output_train = resolve_path(args.output_train, base=ROOT)
    records: List[Dict[str, Any]] = read_jsonl(input_train)
    if not records:
        raise RuntimeError(f"No training records found: {input_train}")
    if args.factor < 1:
        raise ValueError("--factor must be >= 1")

    target = [record for record in records if record.get("label") == args.target_label]
    augmented = list(records)
    for repeat_id in range(args.factor - 1):
        for record in target:
            copied = dict(record)
            copied["sample_id"] = f"{record['sample_id']}__oversample_{args.target_label}_{repeat_id + 1}"
            copied["oversampled_from"] = record["sample_id"]
            augmented.append(copied)

    random.shuffle(augmented)
    write_jsonl(output_train, augmented)

    summary = {
        "input_train": str(input_train),
        "output_train": str(output_train),
        "target_label": args.target_label,
        "factor": args.factor,
        "before_count": len(records),
        "after_count": len(augmented),
        "before_distribution": dict(Counter(record["label"] for record in records)),
        "after_distribution": dict(Counter(record["label"] for record in augmented)),
    }
    write_json(output_train.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

