from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Dict

import torch

from common import ROOT, load_config, read_jsonl, resolve_model_path, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check environment, model path, and prepared data files.")
    parser.add_argument("--config", default="configs/emotion_qlora.yaml")
    return parser.parse_args()


def check_package(name: str) -> Dict[str, str]:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        return {"status": "ok", "version": str(version)}
    except Exception as exc:
        return {"status": "missing", "error": str(exc)}


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    model_path = resolve_model_path(cfg)
    data_dir = resolve_path(cfg["data"]["output_dir"], base=ROOT)

    report = {
        "python_packages": {
            name: check_package(name)
            for name in ["torch", "transformers", "accelerate", "peft", "bitsandbytes", "sklearn", "PIL", "yaml"]
        },
        "cuda": {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "devices": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": torch.cuda.get_device_capability(index),
                }
                for index in range(torch.cuda.device_count())
            ],
        },
        "model": {
            "path": str(model_path),
            "exists": model_path.exists(),
            "has_config": (model_path / "config.json").exists(),
        },
        "data": {},
    }

    for split in ["train", "val", "test"]:
        path = data_dir / f"{split}.jsonl"
        item = {"path": str(path), "exists": path.exists(), "count": 0}
        if path.exists():
            item["count"] = len(read_jsonl(path))
        report["data"][split] = item

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

