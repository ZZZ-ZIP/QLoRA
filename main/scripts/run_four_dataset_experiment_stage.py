from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
import yaml

from common import ROOT, read_jsonl, write_jsonl


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

BASE_CONFIG = ROOT / "configs" / "emotion_qlora_four_datasets.yaml"
RESULT_ROOT = ROOT / "results_paper_extension"
RANK_RESULT_DIR = RESULT_ROOT / "rank_2_32_four_datasets"
RANK_RUN_ROOT = ROOT / "runs" / "rank_sweep_four_datasets"
PROJECT_PREFIX = "qwen3vl_2b_emotion_qlora_four_datasets"
LABELS = ["positive", "negative", "neutral"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch four-dataset paper experiment stages.")
    parser.add_argument(
        "--stage",
        required=True,
        choices=[
            "stage2_main_table",
            "modality_ablation",
            "cross_dataset",
            "seed_stability",
            "lora_module_ablation",
            "alpha_scaling",
            "mechanism_ablation",
            "neutral_error_analysis",
        ],
    )
    parser.add_argument("--rank_best", type=int, default=20)
    parser.add_argument("--rank_stable", type=int, default=16)
    parser.add_argument("--rank", type=int, default=20)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def run(cmd: List[str], dry_run: bool) -> None:
    print("==>", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True, env=os.environ.copy())


def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def read_metrics(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rank_config_path(rank: int) -> Path:
    path = ROOT / "configs" / "rank_sweep_four_datasets" / f"emotion_qlora_r{rank}.yaml"
    if path.exists():
        return path
    cfg = load_yaml(BASE_CONFIG)
    cfg["project_name"] = f"{PROJECT_PREFIX}_r{rank}"
    cfg["training"]["output_dir"] = f"runs/rank_sweep_four_datasets/{PROJECT_PREFIX}_r{rank}"
    cfg["qlora"]["lora_r"] = rank
    cfg["qlora"]["lora_alpha"] = rank * 2
    write_yaml(path, cfg)
    return path


def adapter_dir_for_rank(rank: int) -> Path:
    return RANK_RUN_ROOT / f"{PROJECT_PREFIX}_r{rank}" / "final_adapter"


def result_metric_path(rank: int) -> Path:
    return RANK_RESULT_DIR / f"adapter_r{rank}_test_metrics.json"


def result_prediction_path(rank: int) -> Path:
    return RANK_RESULT_DIR / f"adapter_r{rank}_test_predictions.jsonl"


def train_predict_eval(
    config_path: Path,
    output_dir: Path,
    dry_run: bool,
    skip_existing: bool,
    train_file: Path | None = None,
    val_file: Path | None = None,
    test_file: Path | None = None,
    adapter_dir: Path | None = None,
    limit: int = 0,
) -> None:
    prediction_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not skip_existing or not metrics_path.exists():
        if adapter_dir is None:
            train_cmd = [sys.executable, "scripts/train_qlora_emotion.py", "--config", str(config_path)]
            if train_file is not None:
                train_cmd += ["--train_file", str(train_file)]
            if val_file is not None:
                train_cmd += ["--val_file", str(val_file)]
            run(train_cmd, dry_run)

        predict_cmd = [
            sys.executable,
            "scripts/predict_emotion.py",
            "--config",
            str(config_path),
            "--model_kind",
            "adapter",
            "--output_file",
            str(prediction_path),
        ]
        if test_file is not None:
            predict_cmd += ["--input_file", str(test_file)]
        else:
            predict_cmd += ["--split", "test"]
        if adapter_dir is not None:
            predict_cmd += ["--adapter_dir", str(adapter_dir)]
        if limit > 0:
            predict_cmd += ["--limit", str(limit)]
        run(predict_cmd, dry_run)

        run(
            [
                sys.executable,
                "scripts/evaluate_emotion.py",
                "--config",
                str(config_path),
                "--predictions",
                str(prediction_path),
                "--output_file",
                str(metrics_path),
                "--output_dir",
                str(output_dir),
            ],
            dry_run,
        )


def class_f1(metrics: Dict[str, Any], label: str) -> float:
    return float(metrics.get("classification_report", {}).get(label, {}).get("f1-score", 0.0))


def write_stage2_table(rank_best: int, rank_stable: int) -> None:
    rows: List[Dict[str, Any]] = []
    candidates = [("base", 0), ("r8", 8), ("r16", 16), ("r20", 20), (f"r_best_macro_r{rank_best}", rank_best), (f"r_stable_r{rank_stable}", rank_stable)]
    seen: set[int] = set()
    for name, rank in candidates:
        if rank in seen and rank != 0:
            continue
        seen.add(rank)
        metrics_path = RANK_RESULT_DIR / "base_test_metrics.json" if rank == 0 else result_metric_path(rank)
        if not metrics_path.exists():
            rows.append({"model": name, "rank": rank, "status": f"missing: {metrics_path}"})
            continue
        metrics = read_metrics(metrics_path)
        rows.append(
            {
                "model": name,
                "rank": rank,
                "accuracy": metrics.get("accuracy", 0.0),
                "macro_f1": metrics.get("macro_f1", 0.0),
                "positive_f1": class_f1(metrics, "positive"),
                "negative_f1": class_f1(metrics, "negative"),
                "neutral_f1": class_f1(metrics, "neutral"),
                "weighted_f1": metrics.get("weighted_f1", 0.0),
                "valid_json_rate": metrics.get("valid_json_rate", metrics.get("parse_rate", 0.0)),
            }
        )
    df = pd.DataFrame(rows)
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].round(4)
    out = RESULT_ROOT / "paper_tables" / "main_four_dataset_results.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# Main Four-Dataset Results\n\n" + df.to_markdown(index=False) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


def sample_frames(paths: List[str], n: int) -> List[str]:
    if n <= 0 or len(paths) <= n:
        return paths
    if n == 1:
        return [paths[len(paths) // 2]]
    idxs = [round(i * (len(paths) - 1) / (n - 1)) for i in range(n)]
    return [paths[i] for i in idxs]


def make_modality_files(seed: int = 42) -> Dict[str, Path]:
    source = ROOT / "data_four_datasets" / "test.jsonl"
    records = read_jsonl(source)
    rng = random.Random(seed)
    all_frames = [(r["sample_id"], r.get("frame_paths", [])) for r in records if r.get("frame_paths")]
    out_dir = RESULT_ROOT / "modality_ablation" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, List[Dict[str, Any]]] = {k: [] for k in ["video_text", "text_only", "vision_only", "random_frame_text"]}

    for record in records:
        outputs["video_text"].append(dict(record))

        text_only = dict(record)
        text_only["frame_paths"] = []
        text_only.pop("image_path", None)
        outputs["text_only"].append(text_only)

        vision_only = dict(record)
        vision_only["transcript"] = "No transcript is provided."
        outputs["vision_only"].append(vision_only)

        random_frame = dict(record)
        choices = [item for item in all_frames if item[0] != record["sample_id"]]
        _, replacement = rng.choice(choices)
        random_frame["frame_paths"] = replacement
        outputs["random_frame_text"].append(random_frame)

    paths: Dict[str, Path] = {}
    for name, items in outputs.items():
        path = out_dir / f"{name}.jsonl"
        write_jsonl(path, items)
        paths[name] = path
    return paths


def run_modality(rank: int, dry_run: bool, skip_existing: bool, limit: int) -> None:
    config = rank_config_path(rank)
    adapter = adapter_dir_for_rank(rank)
    paths = make_modality_files()
    rows = []
    for mode, input_file in paths.items():
        out_dir = RESULT_ROOT / "modality_ablation" / f"r{rank}_{mode}"
        train_predict_eval(config, out_dir, dry_run, skip_existing, test_file=input_file, adapter_dir=adapter, limit=limit)
        metrics_path = out_dir / "metrics.json"
        if metrics_path.exists():
            metrics = read_metrics(metrics_path)
            rows.append(
                {
                    "setting": mode,
                    "rank": rank,
                    "accuracy": metrics.get("accuracy", 0.0),
                    "macro_f1": metrics.get("macro_f1", 0.0),
                    "positive_f1": class_f1(metrics, "positive"),
                    "negative_f1": class_f1(metrics, "negative"),
                    "neutral_f1": class_f1(metrics, "neutral"),
                    "weighted_f1": metrics.get("weighted_f1", 0.0),
                    "valid_json_rate": metrics.get("valid_json_rate", metrics.get("parse_rate", 0.0)),
                }
            )
    if rows:
        df = pd.DataFrame(rows).round(4)
        out = RESULT_ROOT / "modality_ablation" / "modality_ablation_metrics.md"
        out.write_text("# Modality Ablation Metrics\n\n" + df.to_markdown(index=False) + "\n", encoding="utf-8")


def make_config_variant(base_config: Path, out_path: Path, **updates: Any) -> Path:
    cfg = load_yaml(base_config)
    for dotted, value in updates.items():
        target = cfg
        keys = dotted.split("__")
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
    write_yaml(out_path, cfg)
    return out_path


def run_cross_dataset(rank: int, dry_run: bool, skip_existing: bool, limit: int) -> None:
    base_rank_config = rank_config_path(rank)
    split_root = ROOT / "data_four_datasets" / "cross_dataset"
    for split_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
        cfg_path = ROOT / "configs" / "cross_dataset_runs" / f"{split_dir.name}_r{rank}.yaml"
        make_config_variant(
            base_rank_config,
            cfg_path,
            project_name=f"{PROJECT_PREFIX}_cross_{split_dir.name}_r{rank}",
            training__output_dir=f"runs/cross_dataset_four/{split_dir.name}_r{rank}",
        )
        out_dir = RESULT_ROOT / "cross_dataset" / f"{split_dir.name}_r{rank}"
        train_predict_eval(
            cfg_path,
            out_dir,
            dry_run,
            skip_existing,
            train_file=split_dir / "train.jsonl",
            val_file=split_dir / "val.jsonl",
            test_file=split_dir / "test.jsonl",
            limit=limit,
        )


def run_seed_stability(ranks: Iterable[int], seeds: Iterable[int], dry_run: bool, skip_existing: bool, limit: int) -> None:
    for rank in ranks:
        base_rank_config = rank_config_path(rank)
        for seed in seeds:
            cfg_path = ROOT / "configs" / "seed_stability_runs" / f"r{rank}_seed{seed}.yaml"
            make_config_variant(
                base_rank_config,
                cfg_path,
                project_name=f"{PROJECT_PREFIX}_r{rank}_seed{seed}",
                seed=int(seed),
                training__output_dir=f"runs/seed_stability_four/r{rank}_seed{seed}",
            )
            out_dir = RESULT_ROOT / "seed_stability" / f"r{rank}_seed{seed}"
            train_predict_eval(cfg_path, out_dir, dry_run, skip_existing, limit=limit)


def run_lora_module_ablation(rank: int, dry_run: bool, skip_existing: bool, limit: int) -> None:
    base_rank_config = rank_config_path(rank)
    module_sets = {
        "attention_only": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp_only": ["gate_proj", "up_proj", "down_proj"],
        "attention_mlp": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    }
    for name, modules in module_sets.items():
        cfg_path = ROOT / "configs" / "lora_module_runs" / f"{name}_r{rank}.yaml"
        make_config_variant(
            base_rank_config,
            cfg_path,
            project_name=f"{PROJECT_PREFIX}_{name}_r{rank}",
            training__output_dir=f"runs/lora_module_four/{name}_r{rank}",
            qlora__target_modules=modules,
        )
        out_dir = RESULT_ROOT / "lora_module_ablation" / f"{name}_r{rank}"
        train_predict_eval(cfg_path, out_dir, dry_run, skip_existing, limit=limit)


def run_alpha_scaling(ranks: Iterable[int], dry_run: bool, skip_existing: bool, limit: int) -> None:
    for rank in ranks:
        base_rank_config = rank_config_path(rank)
        for ratio in [1, 2, 4]:
            alpha = rank * ratio
            cfg_path = ROOT / "configs" / "alpha_scaling_runs" / f"r{rank}_alpha{alpha}.yaml"
            make_config_variant(
                base_rank_config,
                cfg_path,
                project_name=f"{PROJECT_PREFIX}_r{rank}_alpha{alpha}",
                training__output_dir=f"runs/alpha_scaling_four/r{rank}_alpha{alpha}",
                qlora__lora_alpha=int(alpha),
            )
            out_dir = RESULT_ROOT / "alpha_scaling" / f"r{rank}_alpha{alpha}"
            train_predict_eval(cfg_path, out_dir, dry_run, skip_existing, limit=limit)


def run_neutral_error(rank: int, dry_run: bool) -> None:
    pred = result_prediction_path(rank)
    if not pred.exists():
        pred = RESULT_ROOT / "modality_ablation" / f"r{rank}_video_text" / "predictions.jsonl"
    run(
        [
            sys.executable,
            "scripts/analyze_errors.py",
            "--predictions",
            str(pred),
            "--test_file",
            str(ROOT / "data_four_datasets" / "test.jsonl"),
            "--output_dir",
            str(RESULT_ROOT / "error_analysis" / f"four_dataset_r{rank}"),
        ],
        dry_run,
    )


def main() -> int:
    args = parse_args()
    if args.stage == "stage2_main_table":
        write_stage2_table(args.rank_best, args.rank_stable)
    elif args.stage == "modality_ablation":
        run_modality(args.rank, args.dry_run, args.skip_existing, args.limit)
    elif args.stage == "cross_dataset":
        run_cross_dataset(args.rank, args.dry_run, args.skip_existing, args.limit)
    elif args.stage == "seed_stability":
        run_seed_stability([args.rank_best, args.rank_stable], args.seeds, args.dry_run, args.skip_existing, args.limit)
    elif args.stage == "lora_module_ablation":
        run_lora_module_ablation(args.rank, args.dry_run, args.skip_existing, args.limit)
    elif args.stage == "alpha_scaling":
        run_alpha_scaling([16, args.rank], args.dry_run, args.skip_existing, args.limit)
    elif args.stage == "mechanism_ablation":
        run_lora_module_ablation(args.rank, args.dry_run, args.skip_existing, args.limit)
        run_alpha_scaling([16, args.rank], args.dry_run, args.skip_existing, args.limit)
    elif args.stage == "neutral_error_analysis":
        run_neutral_error(args.rank, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
