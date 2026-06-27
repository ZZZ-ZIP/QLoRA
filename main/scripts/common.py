from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent


def load_config(path: str | Path) -> Dict[str, Any]:
    cfg_path = resolve_path(path, base=ROOT)
    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["_config_path"] = str(cfg_path)
    return cfg


def resolve_path(value: str | Path, base: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base or ROOT) / path
    return path.resolve()


def resolve_model_path(cfg: Dict[str, Any]) -> Path:
    return resolve_path(cfg["model"]["base_model"], base=ROOT)


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str | Path, records: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_label(value: Any, labels: List[str], aliases: Dict[str, str] | None = None) -> str:
    aliases = aliases or {}
    label = str(value).strip().lower().replace("_", "-")
    label = re.sub(r"\s+", "-", label)
    label = aliases.get(label, label)
    return label if label in labels else ""


def parse_emotion_from_text(text: str, labels: List[str], aliases: Dict[str, str] | None = None) -> str:
    aliases = aliases or {}
    raw = str(text).strip()
    if not raw:
        return "invalid"

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            value = obj.get("emotion", obj.get("label", ""))
            label = normalize_label(value, labels, aliases)
            return label or "invalid"
    except json.JSONDecodeError:
        pass

    match = re.search(r'"?emotion"?\s*[:=]\s*"?([A-Za-z_-]+)"?', raw, flags=re.IGNORECASE)
    if match:
        label = normalize_label(match.group(1), labels, aliases)
        return label or "invalid"

    lower = raw.lower()
    for label in labels:
        if re.search(rf"\b{re.escape(label)}\b", lower):
            return label

    for alias, canonical in aliases.items():
        if re.search(rf"\b{re.escape(alias)}\b", lower):
            return canonical

    return "invalid"


def _record_image_content(record: Dict[str, Any]) -> List[Dict[str, str]]:
    frame_paths = record.get("frame_paths", [])
    if isinstance(frame_paths, str):
        frame_paths = [item.strip() for item in frame_paths.split(";") if item.strip()]
    if frame_paths:
        return [{"type": "image", "image": str(path)} for path in frame_paths]
    if record.get("image_path"):
        return [{"type": "image", "image": str(record["image_path"])}]
    return []


def build_messages(record: Dict[str, Any], cfg: Dict[str, Any], include_answer: bool) -> List[Dict[str, Any]]:
    labels = cfg["task"]["labels"]
    user_prompt = cfg["task"]["user_prompt_template"].format(
        labels=", ".join(labels),
        transcript=str(record.get("transcript", "")).strip() or "(no transcript provided)",
    )
    content = _record_image_content(record)
    content.append({"type": "text", "text": user_prompt})
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": cfg["task"]["system_prompt"]}]},
        {"role": "user", "content": content},
    ]
    if include_answer:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": json.dumps({"emotion": record["label"]})}],
            }
        )
    return messages


def load_processor(cfg: Dict[str, Any]):
    from transformers import AutoProcessor

    kwargs = {
        "trust_remote_code": bool(cfg["model"].get("trust_remote_code", True)),
        "local_files_only": bool(cfg["model"].get("local_files_only", False)),
    }
    max_pixels = cfg["model"].get("max_pixels")
    if max_pixels:
        kwargs["max_pixels"] = int(max_pixels)
    return AutoProcessor.from_pretrained(str(resolve_model_path(cfg)), **kwargs)


def load_base_model(cfg: Dict[str, Any], quantized: bool = False):
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

    kwargs: Dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": bool(cfg["model"].get("trust_remote_code", True)),
        "local_files_only": bool(cfg["model"].get("local_files_only", False)),
    }

    if quantized:
        compute_dtype = torch.bfloat16 if cfg["qlora"].get("bnb_4bit_compute_dtype") == "bfloat16" else torch.float16
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=bool(cfg["qlora"].get("load_in_4bit", True)),
            bnb_4bit_quant_type=str(cfg["qlora"].get("bnb_4bit_quant_type", "nf4")),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=bool(cfg["qlora"].get("bnb_4bit_use_double_quant", True)),
        )
    else:
        kwargs["torch_dtype"] = "auto"

    return AutoModelForImageTextToText.from_pretrained(str(resolve_model_path(cfg)), **kwargs)


def model_device(model: torch.nn.Module) -> torch.device:
    try:
        return model.device
    except Exception:
        return next(model.parameters()).device
