from __future__ import annotations

import argparse
from typing import Any, Dict, List

from common import ROOT, load_config, read_jsonl, resolve_path, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare base and adapter predictions sample by sample.")
    parser.add_argument("--config", default="configs/emotion_qlora.yaml")
    parser.add_argument("--base_predictions", default="outputs/base_test_predictions.jsonl")
    parser.add_argument("--adapter_predictions", default="outputs/adapter_test_predictions.jsonl")
    parser.add_argument("--output_json", default="outputs/base_vs_adapter_comparison.json")
    parser.add_argument("--error_cases", default="outputs/base_vs_adapter_cases.jsonl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_config(args.config)
    base_records = read_jsonl(resolve_path(args.base_predictions, base=ROOT))
    adapter_records = read_jsonl(resolve_path(args.adapter_predictions, base=ROOT))
    base_by_id = {r["sample_id"]: r for r in base_records}
    adapter_by_id = {r["sample_id"]: r for r in adapter_records}
    common_ids = sorted(set(base_by_id) & set(adapter_by_id))
    if not common_ids:
        raise RuntimeError("No overlapping sample_id values between base and adapter predictions.")

    buckets = {
        "both_correct": [],
        "repaired": [],
        "regressed": [],
        "both_wrong": [],
    }

    for sample_id in common_ids:
        base = base_by_id[sample_id]
        adapter = adapter_by_id[sample_id]
        gold = base["gold"]
        base_ok = base["pred"] == gold
        adapter_ok = adapter["pred"] == gold

        merged = {
            "sample_id": sample_id,
            "dataset": base.get("dataset", ""),
            "image_path": base.get("image_path", ""),
            "frame_paths": base.get("frame_paths", []),
            "transcript": base.get("transcript", ""),
            "gold": gold,
            "base_pred": base["pred"],
            "adapter_pred": adapter["pred"],
            "base_raw": base.get("raw_output", ""),
            "adapter_raw": adapter.get("raw_output", ""),
        }
        if base_ok and adapter_ok:
            buckets["both_correct"].append(merged)
        elif (not base_ok) and adapter_ok:
            buckets["repaired"].append(merged)
        elif base_ok and (not adapter_ok):
            buckets["regressed"].append(merged)
        else:
            buckets["both_wrong"].append(merged)

    summary: Dict[str, Any] = {
        "count": len(common_ids),
        "both_correct": len(buckets["both_correct"]),
        "repaired": len(buckets["repaired"]),
        "regressed": len(buckets["regressed"]),
        "both_wrong": len(buckets["both_wrong"]),
        "repair_minus_regression": len(buckets["repaired"]) - len(buckets["regressed"]),
        "repair_to_regression_ratio": (
            len(buckets["repaired"]) / len(buckets["regressed"])
            if buckets["regressed"]
            else None
        ),
        "decision_hint": "improved" if len(buckets["repaired"]) > len(buckets["regressed"]) else "not_improved",
    }

    cases: List[Dict[str, Any]] = []
    for bucket_name, items in buckets.items():
        for item in items:
            item["bucket"] = bucket_name
            cases.append(item)

    write_json(resolve_path(args.output_json, base=ROOT), summary)
    write_jsonl(resolve_path(args.error_cases, base=ROOT), cases)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
