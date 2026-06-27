from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import PeftModel
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
    resolve_path,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen3-VL emotion prediction.")
    parser.add_argument("--config", default="configs/emotion_qlora.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--input_file", default="")
    parser.add_argument("--output_file", default="")
    parser.add_argument("--model_kind", default="base", choices=["base", "adapter"])
    parser.add_argument("--adapter_dir", default="")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def load_model_for_prediction(cfg: Dict[str, Any], model_kind: str, adapter_dir: str):
    model = load_base_model(cfg, quantized=(model_kind == "adapter"))
    if model_kind == "adapter":
        adapter = resolve_path(adapter_dir, base=ROOT) if adapter_dir else resolve_path(cfg["training"]["output_dir"], base=ROOT) / "final_adapter"
        if not adapter.exists():
            raise FileNotFoundError(f"Adapter directory not found: {adapter}")
        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    return model


def predict_one(model: Any, processor: Any, cfg: Dict[str, Any], record: Dict[str, Any]) -> str:
    messages = build_messages(record, cfg, include_answer=False)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model_device(model))

    gen_cfg = cfg["generation"]
    generate_kwargs = {
        "max_new_tokens": int(gen_cfg.get("max_new_tokens", 32)),
        "do_sample": bool(gen_cfg.get("do_sample", False)),
    }
    if generate_kwargs["do_sample"]:
        generate_kwargs["temperature"] = float(gen_cfg.get("temperature", 0.7))

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **generate_kwargs)

    trimmed_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    text = processor.batch_decode(trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return text.strip()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    data_dir = resolve_path(cfg["data"]["output_dir"], base=ROOT)
    input_file = resolve_path(args.input_file, base=ROOT) if args.input_file else data_dir / f"{args.split}.jsonl"
    output_dir = resolve_path("outputs", base=ROOT)
    output_file = (
        resolve_path(args.output_file, base=ROOT)
        if args.output_file
        else output_dir / f"{args.model_kind}_{args.split}_predictions.jsonl"
    )

    records = read_jsonl(input_file)
    if args.limit:
        records = records[: args.limit]

    processor = load_processor(cfg)
    model = load_model_for_prediction(cfg, args.model_kind, args.adapter_dir)
    labels = cfg["task"]["labels"]
    aliases = cfg["task"].get("label_aliases", {})

    predictions: List[Dict[str, Any]] = []
    for record in tqdm(records, desc=f"predict {args.model_kind}:{args.split}"):
        raw_output = predict_one(model, processor, cfg, record)
        pred = parse_emotion_from_text(raw_output, labels, aliases)
        predictions.append(
            {
                "sample_id": record["sample_id"],
                "dataset": record.get("dataset", ""),
                "image_path": record.get("image_path", ""),
                "frame_paths": record.get("frame_paths", []),
                "transcript": record.get("transcript", ""),
                "gold": record["label"],
                "pred": pred,
                "raw_output": raw_output,
                "model_kind": args.model_kind,
                "split": args.split,
            }
        )

    write_jsonl(output_file, predictions)
    print(f"Wrote {len(predictions)} predictions to {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
