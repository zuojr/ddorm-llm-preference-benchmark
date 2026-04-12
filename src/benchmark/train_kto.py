
from __future__ import annotations

import argparse

from peft import LoraConfig, TaskType
from transformers import AutoTokenizer
from trl.experimental.kto import KTOConfig, KTOTrainer

from .data import load_ultrafeedback_binarized
from .utils import infer_lora_targets, is_local_peft_adapter, load_causal_lm, maybe_set_hf_cache, pick_torch_dtype, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train KTO baseline.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset_name", default="HuggingFaceH4/ultrafeedback_binarized")
    parser.add_argument("--split", default="train_prefs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_prompt_length", type=int, default=768)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    maybe_set_hf_cache()
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_causal_lm(args.model_name_or_path, torch_dtype=pick_torch_dtype(), is_trainable=True)
    model.config.pad_token_id = tokenizer.pad_token_id

    train_dataset = load_ultrafeedback_binarized(args.split, dataset_name=args.dataset_name)

    peft_config = None
    if args.use_lora and not is_local_peft_adapter(args.model_name_or_path):
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            target_modules=infer_lora_targets(args.model_name_or_path),
        )

    training_args = KTOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        beta=args.beta,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = KTOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
