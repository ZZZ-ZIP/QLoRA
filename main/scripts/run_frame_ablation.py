from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

from common import ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run r28 frame-count ablation with prepared jsonl files.")
    parser.add_argument("--base_config", default="configs/emotion_qlora_four_datasets.yaml")
    parser.add_argument("--data_root", default="outputs/frame_ablation/data")
    parser.add_argument("--output_root", default="outputs/frame_ablation")
    parser.add_argument("--run_root", default="runs/frame_ablation")
    parser.add_argument("--rank", type=int, default=28)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument("--modes", nargs="+", default=["frame1", "frame2", "frame4", "random_frame2"])
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def run(cmd: List[str], dry_run: bool) -> None:
    print("==>", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, cwd=ROOT, check=True)


def write_config(base_config: Path, mode: str, run_root: Path, rank: int, alpha: int) -> Path:
    cfg = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    cfg["project_name"] = f"qwen3vl_qlora_r{rank}_frame_{mode}"
    cfg["training"]["output_dir"] = str((run_root / mode).relative_to(ROOT))
    cfg["qlora"]["lora_r"] = rank
    cfg["qlora"]["lora_alpha"] = alpha
    cfg["training"]["per_device_train_batch_size"] = 1
    cfg["training"]["per_device_eval_batch_size"] = 1
    cfg["training"]["gradient_checkpointing"] = True
    cfg["generation"]["do_sample"] = False
    cfg["generation"]["temperature"] = 0.0
    out = ROOT / "configs" / "frame_ablation" / f"emotion_qlora_r{rank}_{mode}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return out


def read_metric(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"status": "not_completed", "metrics_file": str(path)}
    data = json.loads(path.read_text(encoding="utf-8"))
    report = data.get("classification_report", {})
    return {
        "status": "completed",
        "metrics_file": str(path),
        "accuracy": data.get("accuracy"),
        "macro_f1": data.get("macro_f1"),
        "weighted_f1": data.get("weighted_f1"),
        "positive_f1": report.get("positive", {}).get("f1-score"),
        "negative_f1": report.get("negative", {}).get("f1-score"),
        "neutral_f1": report.get("neutral", {}).get("f1-score"),
    }


def main() -> int:
    args = parse_args()
    base_config = resolve(args.base_config)
    data_root = resolve(args.data_root)
    output_root = resolve(args.output_root)
    run_root = resolve(args.run_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    for mode in args.modes:
        mode_data = data_root / mode
        train_file = mode_data / "train.jsonl"
        val_file = mode_data / "val.jsonl"
        test_file = mode_data / "test.jsonl"
        mode_out = output_root / mode
        metrics_file = mode_out / "metrics.json"
        if not train_file.exists():
            print(f"[WARN] missing prepared data for {mode}: {train_file}")
            rows.append({"mode": mode, "status": "missing_data", "metrics_file": str(metrics_file)})
            continue
        cfg_path = write_config(base_config, mode, run_root, args.rank, args.alpha)
        mode_out.mkdir(parents=True, exist_ok=True)
        if args.skip_existing and metrics_file.exists():
            rows.append({"mode": mode, **read_metric(metrics_file)})
            continue
        run([sys.executable, "scripts/train_qlora_emotion.py", "--config", str(cfg_path), "--train_file", str(train_file), "--val_file", str(val_file)], args.dry_run)
        run(
            [
                sys.executable,
                "scripts/predict_emotion.py",
                "--config",
                str(cfg_path),
                "--model_kind",
                "adapter",
                "--input_file",
                str(test_file),
                "--output_file",
                str(mode_out / "predictions.jsonl"),
            ],
            args.dry_run,
        )
        run(
            [
                sys.executable,
                "scripts/evaluate_emotion.py",
                "--config",
                str(cfg_path),
                "--predictions",
                str(mode_out / "predictions.jsonl"),
                "--output_file",
                str(metrics_file),
                "--output_dir",
                str(mode_out),
            ],
            args.dry_run,
        )
        rows.append({"mode": mode, **read_metric(metrics_file)})

    summary_file = output_root / "frame_ablation_summary.csv"
    pd.DataFrame(rows).to_csv(summary_file, index=False, encoding="utf-8-sig")
    print(f"输入数据根目录: {data_root}")
    print(f"输出目录: {output_root}")
    print(f"汇总表: {summary_file}")
    print(f"rank/alpha: r{args.rank}/alpha{args.alpha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
