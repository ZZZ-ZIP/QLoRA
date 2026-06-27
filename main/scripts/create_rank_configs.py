from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import yaml

from common import ROOT, load_config, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create QLoRA rank sweep config files.")
    parser.add_argument("--base_config", default="configs/emotion_qlora.yaml")
    parser.add_argument("--ranks", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--output_dir", default="configs/rank_sweep")
    parser.add_argument("--alpha_multiplier", type=int, default=2)
    parser.add_argument("--run_root", default="runs/rank_sweep")
    parser.add_argument("--project_prefix", default="qwen3vl_2b_emotion_qlora")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_cfg = load_config(args.base_config)
    base_cfg.pop("_config_path", None)
    output_dir = resolve_path(args.output_dir, base=ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    for rank in args.ranks:
        cfg = yaml.safe_load(yaml.safe_dump(base_cfg, sort_keys=False))
        cfg["project_name"] = f"{args.project_prefix}_r{rank}"
        cfg["training"]["output_dir"] = f"{args.run_root}/{args.project_prefix}_r{rank}"
        cfg["qlora"]["lora_r"] = int(rank)
        cfg["qlora"]["lora_alpha"] = int(rank * args.alpha_multiplier)
        path = output_dir / f"emotion_qlora_r{rank}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        written.append(path)

    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
