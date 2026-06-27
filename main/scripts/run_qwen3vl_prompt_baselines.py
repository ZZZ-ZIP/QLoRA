from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

from common import (
    ROOT,
    build_messages,
    load_base_model,
    load_config,
    load_processor,
    model_device,
    parse_emotion_from_text,
    read_jsonl,
    write_jsonl,
)


LABELS = ["positive", "negative", "neutral"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen3-VL zero-shot/few-shot prompt baselines.")
    parser.add_argument("--config", default="configs/emotion_qlora_four_datasets.yaml")
    parser.add_argument("--mode", required=True, choices=["zeroshot_videotext", "zeroshot_textonly", "fewshot_videotext"])
    parser.add_argument("--train_file", default="data_four_datasets/train.jsonl")
    parser.add_argument("--test_file", default="data_four_datasets/test.jsonl")
    parser.add_argument("--output_file", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shots_per_class", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_fewshot_examples(rows: List[Dict[str, Any]], shots_per_class: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    by_label: Dict[str, List[Dict[str, Any]]] = {label: [] for label in LABELS}
    for row in rows:
        label = str(row.get("label"))
        if label in by_label:
            by_label[label].append(row)
    examples: List[Dict[str, Any]] = []
    for label in LABELS:
        candidates = by_label[label][:]
        rng.shuffle(candidates)
        examples.extend(candidates[:shots_per_class])
    return examples


def build_prompt(record: Dict[str, Any], mode: str, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    item = dict(record)
    if mode == "zeroshot_textonly":
        item["frame_paths"] = []
        item["image_path"] = ""
    if examples:
        lines = ["参考示例："]
        for ex in examples:
            lines.append(f"Transcript: {ex.get('transcript', '')}")
            lines.append(json.dumps({"emotion": ex.get("label")}, ensure_ascii=False))
        lines.append("请根据下面样本判断情绪。")
        item["transcript"] = "\n".join(lines) + "\nTranscript: " + str(record.get("transcript", ""))
    return item


def generate_one(model: Any, processor: Any, cfg: Dict[str, Any], record: Dict[str, Any]) -> str:
    messages = build_messages(record, cfg, include_answer=False)
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    inputs = inputs.to(model_device(model))
    gen_cfg = cfg["generation"]
    kwargs = {
        "max_new_tokens": int(gen_cfg.get("max_new_tokens", 32)),
        "do_sample": bool(gen_cfg.get("do_sample", False)),
    }
    if kwargs["do_sample"]:
        kwargs["temperature"] = float(gen_cfg.get("temperature", 0.7))
    with torch.no_grad():
        generated_ids = model.generate(**inputs, **kwargs)
    trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    cfg = load_config(args.config)
    train_file = resolve(args.train_file)
    test_file = resolve(args.test_file)
    output_file = resolve(args.output_file) if args.output_file else ROOT / "outputs" / "baselines" / f"qwen3vl_{args.mode}_predictions.jsonl"
    train_rows = read_jsonl(train_file)
    test_rows = read_jsonl(test_file)
    if args.limit > 0:
        test_rows = test_rows[: args.limit]
    examples = choose_fewshot_examples(train_rows, args.shots_per_class, args.seed) if args.mode == "fewshot_videotext" else []

    processor = load_processor(cfg)
    model = load_base_model(cfg, quantized=True)
    model.eval()
    aliases = cfg["task"].get("label_aliases", {})

    predictions: List[Dict[str, Any]] = []
    for row in tqdm(test_rows, desc=args.mode):
        prompt_row = build_prompt(row, args.mode, examples)
        raw = generate_one(model, processor, cfg, prompt_row)
        pred = parse_emotion_from_text(raw, LABELS, aliases)
        predictions.append(
            {
                "sample_id": row.get("sample_id"),
                "dataset": row.get("dataset"),
                "frame_paths": prompt_row.get("frame_paths", []),
                "transcript": row.get("transcript", ""),
                "gold": row.get("label"),
                "pred": pred,
                "raw_output": raw,
                "model_kind": f"qwen3vl_{args.mode}",
                "split": "test",
            }
        )
    write_jsonl(output_file, predictions)
    print(f"输入 train: {train_file}")
    print(f"输入 test: {test_file}")
    print(f"输出 predictions: {output_file}")
    print(f"样本数: {len(predictions)}")
    print(f"标签分布: {dict(Counter([r.get('gold') for r in predictions]))}")
    print("下一步可运行 evaluate_emotion.py 生成 metrics。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
