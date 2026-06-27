from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from common import ROOT, read_jsonl, resolve_path
from paper_extension_utils import EXT_ROOT, dataset_name, ensure_extension_dirs, write_text


REASON_CANDIDATES = [
    "weak emotion expression",
    "ambiguous label boundary",
    "text-vision conflict",
    "sarcasm or irony",
    "insufficient visual information",
    "cross-lingual expression difference",
    "dataset annotation inconsistency",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze neutral-related error cases.")
    parser.add_argument("--predictions", default="outputs/adapter_r20_test_predictions.jsonl")
    parser.add_argument("--test_file", default="data/test.jsonl")
    parser.add_argument("--output_dir", default=str(EXT_ROOT / "error_analysis"))
    return parser.parse_args()


def _load_test_records(path: Path) -> Dict[str, Dict[str, Any]]:
    return {record["sample_id"]: record for record in read_jsonl(path)}


def _case_row(pred: Dict[str, Any], test_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    source = test_by_id.get(pred["sample_id"], {})
    return {
        "sample_id": pred["sample_id"],
        "dataset": dataset_name(pred.get("dataset") or source.get("dataset", "")),
        "transcript": pred.get("transcript") or source.get("transcript", ""),
        "true_label": pred.get("gold", source.get("label", "")),
        "pred_label": pred.get("pred", "invalid"),
        "frame_paths": pred.get("frame_paths") or source.get("frame_paths", []),
        "raw_model_output": pred.get("raw_output", ""),
        "parsed_emotion": pred.get("pred", "invalid"),
        "possible_reason": "; ".join(REASON_CANDIDATES),
    }


def _markdown_cases(title: str, rows: List[Dict[str, Any]]) -> str:
    lines = [f"## {title}", ""]
    if not rows:
        lines.append("_No cases found._")
        lines.append("")
        return "\n".join(lines)
    for item in rows[:20]:
        lines.extend(
            [
                f"### {item['sample_id']}",
                "",
                f"- dataset: {item['dataset']}",
                f"- true_label: {item['true_label']}",
                f"- pred_label: {item['pred_label']}",
                f"- transcript: {item['transcript']}",
                f"- raw_model_output: `{item['raw_model_output']}`",
                f"- frame_paths: `{item['frame_paths']}`",
                f"- possible_reason: {item['possible_reason']}",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    ensure_extension_dirs()
    output_dir = resolve_path(args.output_dir, base=ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    preds = read_jsonl(resolve_path(args.predictions, base=ROOT))
    test_by_id = _load_test_records(resolve_path(args.test_file, base=ROOT))
    rows = [_case_row(pred, test_by_id) for pred in preds]
    errors = [row for row in rows if row["true_label"] != row["pred_label"]]

    by_dataset: Dict[str, Counter] = defaultdict(Counter)
    by_pair: Counter = Counter()
    for row in rows:
        ds = row["dataset"]
        by_dataset[ds]["total"] += 1
        if row["true_label"] != row["pred_label"]:
            by_dataset[ds]["errors"] += 1
            by_pair[(row["true_label"], row["pred_label"])] += 1

    dataset_df = pd.DataFrame(
        [
            {
                "dataset": ds,
                "total": counts["total"],
                "errors": counts["errors"],
                "error_rate": counts["errors"] / counts["total"] if counts["total"] else 0.0,
            }
            for ds, counts in sorted(by_dataset.items())
        ]
    )
    pair_df = pd.DataFrame(
        [
            {"true_label": true, "pred_label": pred, "count": count}
            for (true, pred), count in by_pair.most_common()
        ]
    )
    dataset_df.to_csv(output_dir / "error_by_dataset.csv", index=False, encoding="utf-8-sig")
    pair_df.to_csv(output_dir / "error_by_label_pair.csv", index=False, encoding="utf-8-sig")

    neutral_groups = {
        "true neutral but predicted positive": [r for r in errors if r["true_label"] == "neutral" and r["pred_label"] == "positive"],
        "true neutral but predicted negative": [r for r in errors if r["true_label"] == "neutral" and r["pred_label"] == "negative"],
        "true positive but predicted neutral": [r for r in errors if r["true_label"] == "positive" and r["pred_label"] == "neutral"],
        "true negative but predicted neutral": [r for r in errors if r["true_label"] == "negative" and r["pred_label"] == "neutral"],
    }

    summary_lines = [
        "# Error Summary",
        "",
        f"- prediction_file: `{resolve_path(args.predictions, base=ROOT)}`",
        f"- total_samples: {len(rows)}",
        f"- error_count: {len(errors)}",
        f"- error_rate: {len(errors) / len(rows):.4f}" if rows else "- error_rate: NA",
        "",
        "## Errors by Dataset",
        "",
        dataset_df.round(4).to_markdown(index=False),
        "",
        "## Errors by Label Pair",
        "",
        pair_df.to_markdown(index=False) if not pair_df.empty else "_No errors found._",
    ]
    write_text(output_dir / "error_summary.md", "\n".join(summary_lines) + "\n")

    case_lines = ["# Neutral-Related Error Cases", ""]
    for title, group_rows in neutral_groups.items():
        case_lines.append(_markdown_cases(title, group_rows))
    write_text(output_dir / "neutral_error_cases.md", "\n".join(case_lines) + "\n")

    neutral_to_pos = len(neutral_groups["true neutral but predicted positive"])
    neutral_to_neg = len(neutral_groups["true neutral but predicted negative"])
    into_neutral = len(neutral_groups["true positive but predicted neutral"]) + len(neutral_groups["true negative but predicted neutral"])
    analysis = [
        "# Neutral Bottleneck Analysis",
        "",
        "This analysis is generated from completed predictions; it should be manually reviewed before being used as a final paper claim.",
        "",
        f"- Neutral -> positive errors: {neutral_to_pos}",
        f"- Neutral -> negative errors: {neutral_to_neg}",
        f"- Positive/negative -> neutral errors: {into_neutral}",
        "",
        "The current completed oversampling experiments show that simple duplicate-style neutral oversampling did not improve the neutral class. "
        "A cautious interpretation is that the neutral bottleneck is not only caused by sample count. It may also reflect weak affective cues, "
        "ambiguous annotation boundaries, text-vision conflicts, or dataset-specific labeling conventions.",
        "",
        "Before writing a strong claim, inspect `neutral_error_cases.md` and fill the `possible_reason` field for representative cases.",
    ]
    write_text(output_dir / "neutral_bottleneck_analysis.md", "\n".join(analysis) + "\n")
    print(f"Wrote error analysis to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
