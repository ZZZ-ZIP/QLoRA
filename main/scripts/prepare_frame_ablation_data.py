from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from common import ROOT, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare frame-count ablation jsonl files.")
    parser.add_argument("--input_dir", default="data_four_datasets")
    parser.add_argument("--output_dir", default="outputs/frame_ablation/data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modes", nargs="+", default=["text_only", "frame1", "frame2", "frame4", "random_frame2"])
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def uniform_sample(paths: List[str], count: int) -> List[str]:
    if count <= 0:
        return []
    if len(paths) <= count:
        return paths
    if count == 1:
        return [paths[len(paths) // 2]]
    idxs = [round(i * (len(paths) - 1) / (count - 1)) for i in range(count)]
    return [paths[i] for i in idxs]


def convert_record(record: Dict[str, Any], mode: str, rng: random.Random) -> Dict[str, Any]:
    item = dict(record)
    frames = list(item.get("frame_paths") or [])
    if mode == "text_only":
        item["frame_paths"] = []
        item["image_path"] = ""
        item["frame_sampling"] = "none_text_only"
    elif mode == "frame1":
        item["frame_paths"] = uniform_sample(frames, 1)
        item["frame_sampling"] = "middle_frame"
    elif mode == "frame2":
        item["frame_paths"] = uniform_sample(frames, 2)
        item["frame_sampling"] = "uniform_2_frames"
    elif mode == "frame4":
        item["frame_paths"] = uniform_sample(frames, 4)
        item["frame_sampling"] = "uniform_4_frames"
    elif mode == "random_frame2":
        item["frame_paths"] = rng.sample(frames, k=min(2, len(frames))) if frames else []
        item["frame_sampling"] = "random_2_frames_seeded"
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return item


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    input_dir = resolve(args.input_dir)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: List[Dict[str, Any]] = []

    for split in ["train", "val", "test"]:
        source = input_dir / f"{split}.jsonl"
        rows = read_jsonl(source)
        for mode in args.modes:
            converted = [convert_record(row, mode, rng) for row in rows]
            path = output_dir / mode / f"{split}.jsonl"
            write_jsonl(path, converted)
            frame_counts = Counter(len(r.get("frame_paths") or []) for r in converted)
            labels = Counter(str(r.get("label")) for r in converted)
            summary.append(
                {
                    "mode": mode,
                    "split": split,
                    "path": str(path),
                    "count": len(converted),
                    "frame_count_distribution": json.dumps(dict(frame_counts), ensure_ascii=False),
                    "label_distribution": json.dumps(dict(labels), ensure_ascii=False),
                }
            )

    import pandas as pd

    summary_path = output_dir / "frame_ablation_data_summary.csv"
    pd.DataFrame(summary).to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"样本统计: {summary_path}")
    print(f"随机种子: {args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
