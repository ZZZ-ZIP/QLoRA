from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import pandas as pd

from common import ROOT, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Accuracy, Macro F1, and Neutral F1 across LoRA ranks.")
    parser.add_argument("--metrics_dir", default="outputs")
    parser.add_argument("--ranks", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--auto_discover", action="store_true", help="Also include every outputs/adapter_r*_test_metrics.json file.")
    parser.add_argument("--output_dir", default="outputs/rank_sweep")
    parser.add_argument("--include_base", action="store_true", help="Include original base model metrics as rank 0.")
    parser.add_argument("--base_metrics", default="outputs/base_test_metrics.json")
    return parser.parse_args()


def discover_ranks(metrics_dir: Path) -> List[int]:
    ranks: List[int] = []
    for path in metrics_dir.glob("adapter_r*_test_metrics.json"):
        if "neutral" in path.name:
            continue
        raw = path.name.removeprefix("adapter_r").removesuffix("_test_metrics.json")
        if raw.isdigit():
            ranks.append(int(raw))
    return sorted(set(ranks))


def metric_path(metrics_dir: Path, rank: int) -> Path:
    if rank == 8:
        baseline = metrics_dir / "adapter_test_metrics.json"
        sweep = metrics_dir / "adapter_r8_test_metrics.json"
        return sweep if sweep.exists() else baseline
    return metrics_dir / f"adapter_r{rank}_test_metrics.json"


def row_from_metrics(name: str, rank: int, path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    report = data.get("classification_report", {})
    return {
        "model": name,
        "rank": rank,
        "accuracy": float(data["accuracy"]),
        "macro_f1": float(data["macro_f1"]),
        "neutral_f1": float(report.get("neutral", {}).get("f1-score", 0.0)),
        "balanced_accuracy": float(data.get("balanced_accuracy", 0.0)),
        "weighted_f1": float(data.get("weighted_f1", 0.0)),
        "parse_rate": float(data.get("parse_rate", 0.0)),
        "metrics_file": str(path),
    }


def load_rank_metrics(metrics_dir: Path, ranks: List[int], include_base: bool, base_metrics: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    if include_base:
        if base_metrics.exists():
            rows.append(row_from_metrics("base", 0, base_metrics))
        else:
            missing.append(str(base_metrics))

    for rank in ranks:
        path = metric_path(metrics_dir, rank)
        if not path.exists():
            missing.append(str(path))
            continue
        rows.append(row_from_metrics(f"r{rank}", rank, path))

    if missing:
        print("Skipped missing metrics:")
        for item in missing:
            print(f"  {item}")
    if not rows:
        raise RuntimeError("No rank metrics found.")
    return pd.DataFrame(rows).sort_values("rank")


def plot_metrics(df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8.5, 5.2), dpi=160)
    series = [
        ("accuracy", "Accuracy", "#2563eb"),
        ("macro_f1", "Macro F1", "#059669"),
        ("neutral_f1", "Neutral F1", "#dc2626"),
    ]
    for column, label, color in series:
        plt.plot(df["rank"], df[column], marker="o", linewidth=2.2, markersize=6, label=label, color=color)
        for x, y in zip(df["rank"], df[column]):
            plt.text(x, y + 0.006, f"{y:.3f}", ha="center", va="bottom", fontsize=8)

    tick_labels = ["base" if int(rank) == 0 else f"r{int(rank)}" for rank in df["rank"]]
    plt.xticks(df["rank"], tick_labels)
    plt.ylim(max(0.0, float(df[["accuracy", "macro_f1", "neutral_f1"]].min().min()) - 0.05), 1.0)
    plt.xlabel("Model / LoRA rank")
    plt.ylabel("Score")
    plt.title("Qwen3-VL-2B QLoRA Rank Sweep")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def write_markdown(df: pd.DataFrame, path: Path) -> None:
    display = df[["model", "rank", "accuracy", "macro_f1", "neutral_f1", "balanced_accuracy", "weighted_f1"]].copy()
    for column in display.columns:
        if column not in {"model", "rank"}:
            display[column] = display[column].round(4)

    best_macro = display.loc[display["macro_f1"].idxmax()]
    best_neutral = display.loc[display["neutral_f1"].idxmax()]
    lines = [
        "# QLoRA Rank Sweep",
        "",
        display.to_markdown(index=False),
        "",
        f"- Best Macro F1: {best_macro['model']} ({best_macro['macro_f1']:.4f})",
        f"- Best Neutral F1: {best_neutral['model']} ({best_neutral['neutral_f1']:.4f})",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    metrics_dir = resolve_path(args.metrics_dir, base=ROOT)
    output_dir = resolve_path(args.output_dir, base=ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_metrics = resolve_path(args.base_metrics, base=ROOT)
    ranks = sorted(set(args.ranks + (discover_ranks(metrics_dir) if args.auto_discover else [])))
    df = load_rank_metrics(metrics_dir, ranks, args.include_base, base_metrics)
    csv_path = output_dir / "rank_sweep_metrics.csv"
    md_path = output_dir / "rank_sweep_metrics.md"
    png_path = output_dir / "rank_sweep_accuracy_macro_neutral.png"

    out_df = df.copy()
    for column in ["accuracy", "macro_f1", "neutral_f1", "balanced_accuracy", "weighted_f1", "parse_rate"]:
        out_df[column] = out_df[column].round(6)
    out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    write_markdown(df, md_path)
    plot_metrics(df, png_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
