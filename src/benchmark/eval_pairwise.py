from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer

from .data import load_ultrafeedback_binarized
from .scoring import batched_reward_scores, batched_sequence_scores
from .utils import (
    load_causal_lm,
    load_sequence_classification_model,
    maybe_set_hf_cache,
    pick_torch_dtype,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pairwise preference accuracy on UltraFeedback binarized.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--model_type", choices=["policy", "reward"], required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--dataset_name", default="HuggingFaceH4/ultrafeedback_binarized")
    parser.add_argument("--split", default="test_prefs")
    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    maybe_set_hf_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = load_ultrafeedback_binarized(args.split, dataset_name=args.dataset_name)
    prompts = ds["prompt"]
    chosen = ds["chosen"]
    rejected = ds["rejected"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.model_type == "policy":
        model = load_causal_lm(
            args.model_name_or_path,
            torch_dtype=pick_torch_dtype(),
            is_trainable=False,
        )
        model.config.pad_token_id = tokenizer.pad_token_id
        model = model.to(device)
        model.eval()

        c = (
            batched_sequence_scores(
                model, tokenizer, prompts, chosen,
                max_length=args.max_length,
                batch_size=args.batch_size,
                device=device,
            )
            .avg_logprob.detach().float().cpu().numpy()
        )
        r = (
            batched_sequence_scores(
                model, tokenizer, prompts, rejected,
                max_length=args.max_length,
                batch_size=args.batch_size,
                device=device,
            )
            .avg_logprob.detach().float().cpu().numpy()
        )
    else:
        model = load_sequence_classification_model(
            args.model_name_or_path,
            num_labels=1,
            torch_dtype=pick_torch_dtype(),
            is_trainable=False,
        )
        model.config.pad_token_id = tokenizer.pad_token_id
        model = model.to(device)
        model.eval()

        c = batched_reward_scores(
            model, tokenizer, prompts, chosen,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=device,
        ).detach().float().cpu().numpy()
        r = batched_reward_scores(
            model, tokenizer, prompts, rejected,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=device,
        ).detach().float().cpu().numpy()

    margin = c - r
    y_true = np.concatenate([np.ones_like(c), np.zeros_like(r)])
    y_score = np.concatenate([c, r])

    metrics = {
        "split": args.split,
        "num_examples": int(len(c)),
        "pair_accuracy": float((margin > 0).mean()),
        "auc": float(roc_auc_score(y_true, y_score)),
        "mean_margin": float(margin.mean()),
        "std_margin": float(margin.std()),
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2))
    print(metrics)


if __name__ == "__main__":
    main()
