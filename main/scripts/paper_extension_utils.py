from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

try:
    from common import ROOT, read_jsonl
except ModuleNotFoundError:
    from .common import ROOT, read_jsonl


EXT_ROOT = ROOT / "results_paper_extension"
PAPER_TABLE_DIR = EXT_ROOT / "paper_tables"
DATASETS = {
    "CMU-MOSEI": "MOSEI",
    "MOSEI": "MOSEI",
    "CMU-MOSI": "MOSI",
    "MOSI": "MOSI",
    "CH-SIMSv2": "CHSIMSv2",
    "ch-simsv2s": "CHSIMSv2",
    "CHSIMS": "CHSIMSv2",
    "CHSIMSv2": "CHSIMSv2",
    "SIMS": "SIMS",
    "sims": "SIMS",
}
LABELS = ["positive", "negative", "neutral"]


def ensure_extension_dirs() -> None:
    for name in [
        "modality_ablation",
        "frame_ablation",
        "cross_dataset",
        "seed_stability",
        "lora_module_ablation",
        "alpha_scaling",
        "error_analysis",
        "paper_tables",
    ]:
        (EXT_ROOT / name).mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def round_float(value: Any, ndigits: int = 4) -> Any:
    if value is None:
        return value
    try:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return value


def metric_row(model: str, metrics_path: Path, rank: int | None = None) -> Dict[str, Any]:
    data = read_json(metrics_path)
    report = data.get("classification_report", {})
    row = {
        "model": model,
        "rank": rank,
        "accuracy": float(data.get("accuracy", 0.0)),
        "macro_f1": float(data.get("macro_f1", 0.0)),
        "positive_f1": float(report.get("positive", {}).get("f1-score", 0.0)),
        "negative_f1": float(report.get("negative", {}).get("f1-score", 0.0)),
        "neutral_f1": float(report.get("neutral", {}).get("f1-score", 0.0)),
        "weighted_f1": float(data.get("weighted_f1", 0.0)),
        "valid_json_rate": float(data.get("valid_json_rate", data.get("parse_rate", 0.0))),
        "metrics_file": str(metrics_path),
    }
    return row


def existing_metric_rows() -> List[Dict[str, Any]]:
    outputs = ROOT / "outputs"
    candidates: List[tuple[str, Path, int | None]] = [
        ("base", outputs / "base_test_metrics.json", 0),
        ("r8", outputs / "adapter_r8_test_metrics.json", 8),
        ("r16", outputs / "adapter_r16_test_metrics.json", 16),
        ("r20", outputs / "adapter_r20_test_metrics.json", 20),
        ("r16_neutral_x2", outputs / "adapter_r16_neutral_x2_test_metrics.json", 16),
        ("r8_neutral_x2", outputs / "adapter_r8_neutral_x2_test_metrics.json", 8),
    ]
    rows = []
    for model, path, rank in candidates:
        if path.exists():
            rows.append(metric_row(model, path, rank))
    return rows


def rank_metric_rows() -> List[Dict[str, Any]]:
    outputs = ROOT / "outputs"
    rows: List[Dict[str, Any]] = []
    base_path = outputs / "base_test_metrics.json"
    if base_path.exists():
        rows.append(metric_row("base", base_path, 0))
    for path in sorted(outputs.glob("adapter_r*_test_metrics.json")):
        if "neutral" in path.name:
            continue
        raw = path.name.removeprefix("adapter_r").removesuffix("_test_metrics.json")
        if raw.isdigit():
            rows.append(metric_row(f"r{raw}", path, int(raw)))
    rows.sort(key=lambda item: int(item["rank"] or 0))
    return rows


def dataframe_to_markdown(path: Path, title: str, rows: Iterable[Dict[str, Any]], columns: List[str]) -> pd.DataFrame:
    df = pd.DataFrame(list(rows), columns=columns)
    display = df.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].round(4)
    body = display.to_markdown(index=False) if not display.empty else "_No completed runs found yet._"
    write_text(path, f"# {title}\n\n{body}\n")
    return df


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def dataset_name(value: str) -> str:
    return DATASETS.get(str(value), str(value))


def active_data_dir() -> Path:
    four_dataset_dir = ROOT / "data_four_datasets"
    return four_dataset_dir if four_dataset_dir.exists() else ROOT / "data"


def load_all_records() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    data_dir = active_data_dir()
    for split in ["train", "val", "test"]:
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        for record in read_jsonl(path):
            item = dict(record)
            item["split"] = split
            item["dataset"] = dataset_name(item.get("dataset", ""))
            records.append(item)
    return records
