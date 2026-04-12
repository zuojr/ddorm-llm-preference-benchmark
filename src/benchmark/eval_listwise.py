
from __future__ import annotations

import argparse
import math

import numpy as np
from datasets import load_from_disk
from scipy.stats import kendalltau
from transformers import AutoModelForCausalLM, AutoTokenizer

from .scoring import batched_sequence_scores
from .utils import maybe_set_hf_cache, pick_torch_dtype, save_json


def dcg(scores: list[float]) -> float:
    return sum((2 ** s - 1) / math.log2(i + 2) for i, s in enumerate(scores))


def ndcg(pred_order: list[int], gold_scores: list[float]) -> float:
    pred_scores = [gold_scores[i] for i in pred_order]
    ideal_scores = sorted(gold_scores, reverse=True)
    denom = dcg(ideal_scores)
    return 0.0 if denom == 0 else dcg(pred_scores) / denom


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate policy model on fixed-size listwise dataset.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--max_length", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    maybe_set_hf_cache()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=pick_torch_dtype())
    model.eval()

    ds = load_from_disk(args.dataset_path)

    top1 = 0
    ndcgs: list[float] = []
    taus: list[float] = []

    for ex in ds:
        prompt = ex["prompt"]
        candidates = ex["candidates"]
        gold_scores = ex["reward_scores"]
        prompts = [prompt] * len(candidates)
        pred_scores = batched_sequence_scores(model, tokenizer, prompts, candidates, max_length=args.max_length).avg_logprob.detach().cpu().numpy().tolist()
        pred_order = list(np.argsort(pred_scores)[::-1])
        gold_order = list(np.argsort(gold_scores)[::-1])
        top1 += int(pred_order[0] == gold_order[0])
        ndcgs.append(ndcg(pred_order, gold_scores))
        tau, _ = kendalltau(pred_order, gold_order)
        taus.append(0.0 if np.isnan(tau) else float(tau))

    metrics = {
        "num_examples": len(ds),
        "top1_accuracy": top1 / len(ds),
        "mean_ndcg": float(np.mean(ndcgs)),
        "mean_kendall_tau": float(np.mean(taus)),
    }
    save_json(metrics, args.output_path)
    print(metrics)


if __name__ == "__main__":
    main()
