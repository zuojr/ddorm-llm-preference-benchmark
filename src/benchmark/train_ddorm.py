
from __future__ import annotations

import argparse

import torch

from datasets import load_from_disk
from transformers import AutoTokenizer, TrainingArguments

from .data import load_ultrafeedback_binarized
from .trainer_ddorm import DDORMCollator, DDORMTrainer
from .utils import load_causal_lm, load_sequence_classification_model, maybe_set_hf_cache, pick_torch_dtype, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DDO-RM / RIPLM candidate-distribution distillation policy.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset_name", default="HuggingFaceH4/ultrafeedback_binarized")
    parser.add_argument("--dataset_path", default=None, help="Optional local dataset saved with datasets.save_to_disk")
    parser.add_argument("--split", default="train_prefs")
    parser.add_argument("--reward_model_name_or_path", default=None)
    parser.add_argument("--use_gold_rewards", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=2e-6)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--ddorm_eta", type=float, default=1.0)
    parser.add_argument("--decision_temperature", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    maybe_set_hf_cache()
    set_seed(args.seed)

    policy_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if policy_tokenizer.pad_token is None:
        policy_tokenizer.pad_token = policy_tokenizer.eos_token

    model = load_causal_lm(args.model_name_or_path, torch_dtype=pick_torch_dtype(), is_trainable=True)
    model.config.pad_token_id = policy_tokenizer.pad_token_id

    if args.dataset_path:
        train_dataset = load_from_disk(args.dataset_path)
    else:
        train_dataset = load_ultrafeedback_binarized(args.split, dataset_name=args.dataset_name)

    reward_model = None
    reward_tokenizer = None
    if not args.use_gold_rewards:
        if not args.reward_model_name_or_path:
            raise ValueError("Please provide --reward_model_name_or_path or use --use_gold_rewards.")
        reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_name_or_path, use_fast=True)
        if reward_tokenizer.pad_token is None:
            reward_tokenizer.pad_token = reward_tokenizer.eos_token
        reward_model = load_sequence_classification_model(
            args.reward_model_name_or_path,
            num_labels=1,
            torch_dtype=pick_torch_dtype(),
            is_trainable=False,
        )
        reward_model.config.pad_token_id = reward_tokenizer.pad_token_id
        device = "cuda" if torch.cuda.is_available() else "cpu"
        reward_model = reward_model.to(device)
        reward_model.eval()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
    )

    trainer = DDORMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DDORMCollator(),
        reward_model=reward_model,
        reward_tokenizer=reward_tokenizer,
        policy_tokenizer=policy_tokenizer,
        ddorm_eta=args.ddorm_eta,
        decision_temperature=args.decision_temperature,
        max_length=args.max_length,
        use_gold_rewards=args.use_gold_rewards,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    policy_tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
