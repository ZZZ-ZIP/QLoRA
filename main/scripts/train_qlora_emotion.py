from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from common import (
    ROOT,
    build_messages,
    load_base_model,
    load_config,
    load_processor,
    read_jsonl,
    resolve_path,
    set_seed,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for Qwen3-VL emotion recognition.")
    parser.add_argument("--config", default="configs/emotion_qlora.yaml")
    parser.add_argument("--train_file", default="")
    parser.add_argument("--val_file", default="")
    return parser.parse_args()


class EmotionDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.records[index]


class SingleSampleCollator:
    def __init__(self, processor: Any, cfg: Dict[str, Any]):
        self.processor = processor
        self.cfg = cfg

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if len(examples) != 1:
            raise ValueError("This collator expects per_device_train_batch_size=1 for stable VLM QLoRA training.")

        record = examples[0]
        full_messages = build_messages(record, self.cfg, include_answer=True)
        prompt_messages = build_messages(record, self.cfg, include_answer=False)

        full_inputs = self.processor.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )
        prompt_inputs = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        labels = full_inputs["input_ids"].clone()
        prompt_len = int(prompt_inputs["input_ids"].shape[-1])
        labels[:, :prompt_len] = -100
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        full_inputs["labels"] = labels
        return full_inputs


def freeze_non_lora_backbone(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        if name.startswith("visual") or "visual" in name or "vision" in name:
            param.requires_grad = False


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    data_dir = resolve_path(cfg["data"]["output_dir"], base=ROOT)
    train_file = resolve_path(args.train_file, base=ROOT) if args.train_file else data_dir / "train.jsonl"
    val_file = resolve_path(args.val_file, base=ROOT) if args.val_file else data_dir / "val.jsonl"

    train_records = read_jsonl(train_file)
    val_records = read_jsonl(val_file)
    if not train_records:
        raise RuntimeError(f"No training records in {train_file}")
    if not val_records:
        raise RuntimeError(f"No validation records in {val_file}")

    processor = load_processor(cfg)
    model = load_base_model(cfg, quantized=True)
    model.config.use_cache = False

    if bool(cfg["training"].get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()

    model = prepare_model_for_kbit_training(model)
    freeze_non_lora_backbone(model)

    lora_cfg = LoraConfig(
        r=int(cfg["qlora"]["lora_r"]),
        lora_alpha=int(cfg["qlora"]["lora_alpha"]),
        lora_dropout=float(cfg["qlora"].get("lora_dropout", 0.05)),
        target_modules=list(cfg["qlora"]["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_cfg = cfg["training"]
    output_dir = resolve_path(train_cfg["output_dir"], base=ROOT)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(train_cfg["num_train_epochs"]),
        learning_rate=float(train_cfg["learning_rate"]),
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(train_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        logging_steps=int(train_cfg["logging_steps"]),
        save_steps=int(train_cfg["save_steps"]),
        eval_steps=int(train_cfg["eval_steps"]),
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=int(train_cfg["save_total_limit"]),
        max_grad_norm=float(train_cfg["max_grad_norm"]),
        bf16=bool(train_cfg.get("bf16", True)),
        fp16=bool(train_cfg.get("fp16", False)),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        optim=str(train_cfg.get("optim", "paged_adamw_8bit")),
        report_to=str(train_cfg.get("report_to", "none")),
        remove_unused_columns=bool(train_cfg.get("remove_unused_columns", False)),
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=EmotionDataset(train_records),
        eval_dataset=EmotionDataset(val_records),
        data_collator=SingleSampleCollator(processor, cfg),
    )

    trainer.train()
    trainer.save_model(str(output_dir / "final_adapter"))
    processor.save_pretrained(str(output_dir / "final_adapter"))
    write_json(output_dir / "experiment_config.json", cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

