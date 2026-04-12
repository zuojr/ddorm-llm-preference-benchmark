
from __future__ import annotations

import argparse
import statistics
from typing import Any

from datasets import Dataset, load_dataset

USER_TAG = "<|user|>"
ASSISTANT_TAG = "<|assistant|>"
SYSTEM_TAG = "<|system|>"
OTHER_TAG = "<|other|>"


def _role_tag(role: str) -> str:
    role = role.lower().strip()
    if role == "user":
        return USER_TAG
    if role == "assistant":
        return ASSISTANT_TAG
    if role == "system":
        return SYSTEM_TAG
    return f"{OTHER_TAG}:{role}"


def render_messages(messages: list[dict[str, Any]], *, add_assistant_prefix: bool = False) -> str:
    chunks: list[str] = []
    for msg in messages:
        tag = _role_tag(msg["role"])
        content = str(msg["content"]).strip()
        chunks.append(f"{tag}\n{content}\n")
    if add_assistant_prefix:
        if not messages or str(messages[-1].get("role", "")).lower() != "assistant":
            chunks.append(f"{ASSISTANT_TAG}\n")
    return "\n".join(chunks).strip() + ("\n" if chunks else "")


def split_prompt_and_completion(full_messages: list[dict[str, Any]]) -> tuple[str, str]:
    if len(full_messages) < 2:
        raise ValueError("Expected at least one prompt message and one assistant completion.")
    prompt_messages = full_messages[:-1]
    completion_message = full_messages[-1]
    if completion_message["role"].lower() != "assistant":
        raise ValueError("Last message must be assistant completion.")
    prompt = render_messages(prompt_messages, add_assistant_prefix=True)
    completion = str(completion_message["content"]).strip()
    return prompt, completion


def convert_binarized_pref_example(example: dict[str, Any]) -> dict[str, Any]:
    prompt_chosen, chosen = split_prompt_and_completion(example["chosen"])
    prompt_rejected, rejected = split_prompt_and_completion(example["rejected"])
    if prompt_chosen != prompt_rejected:
        raise ValueError("Chosen and rejected prompts do not match.")
    return {
        "prompt": prompt_chosen,
        "chosen": chosen,
        "rejected": rejected,
        "candidates": [chosen, rejected],
        "reward_scores": [float(example.get("score_chosen", 1.0)), float(example.get("score_rejected", 0.0))],
        "prompt_id": example.get("prompt_id", ""),
    }


def convert_binarized_sft_example(example: dict[str, Any]) -> dict[str, Any]:
    messages = example.get("messages") or example.get("chosen")
    prompt, completion = split_prompt_and_completion(messages)
    return {"prompt": prompt, "completion": completion, "prompt_id": example.get("prompt_id", "")}


def convert_binarized_gen_example(example: dict[str, Any]) -> dict[str, Any]:
    messages = example.get("messages")
    if messages:
        if messages[-1]["role"].lower() == "assistant":
            prompt_messages = messages[:-1]
        else:
            prompt_messages = messages
        prompt = render_messages(prompt_messages, add_assistant_prefix=True)
    else:
        prompt = f"{USER_TAG}\n{str(example['prompt']).strip()}\n\n{ASSISTANT_TAG}\n"
    return {"prompt": prompt, "prompt_id": example.get("prompt_id", "")}


def load_ultrafeedback_binarized(split: str, dataset_name: str = "HuggingFaceH4/ultrafeedback_binarized") -> Dataset:
    ds = load_dataset(dataset_name, split=split)
    if split.endswith("prefs"):
        return ds.map(convert_binarized_pref_example, remove_columns=ds.column_names)
    if split.endswith("sft"):
        return ds.map(convert_binarized_sft_example, remove_columns=ds.column_names)
    if split.endswith("gen"):
        return ds.map(convert_binarized_gen_example, remove_columns=ds.column_names)
    raise ValueError(f"Unknown split: {split}")


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "None":
            return None
        return float(value)
    except Exception:
        return None


def ultrafeedback_completion_score(completion: dict[str, Any], *, mode: str = "mean_rating") -> float:
    if mode == "mean_rating":
        annotations = completion.get("annotations", {})
        ratings: list[float] = []
        for aspect in annotations.values():
            if isinstance(aspect, dict):
                rating = _safe_float(aspect.get("Rating"))
                if rating is not None:
                    ratings.append(rating)
        if not ratings:
            raise ValueError("No numeric ratings found in completion annotations.")
        return float(statistics.mean(ratings))
    if mode == "helpfulness":
        rating = _safe_float(completion.get("annotations", {}).get("helpfulness", {}).get("Rating"))
        if rating is None:
            raise ValueError("No helpfulness rating found.")
        return float(rating)
    raise ValueError(f"Unknown scoring mode: {mode}")


def convert_openbmb_ultrafeedback_to_listwise(
    split: str = "train",
    dataset_name: str = "openbmb/UltraFeedback",
    score_mode: str = "mean_rating",
    max_examples: int | None = None,
    start: int = 0,
    shuffle_seed: int | None = None,
) -> Dataset:
    ds = load_dataset(dataset_name, split=split)
    if shuffle_seed is not None:
        ds = ds.shuffle(seed=shuffle_seed)
    if start:
        ds = ds.select(range(start, len(ds)))
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))

    def _convert(example: dict[str, Any]) -> dict[str, Any]:
        prompt = f"{USER_TAG}\n{str(example['instruction']).strip()}\n\n{ASSISTANT_TAG}\n"
        candidates: list[str] = []
        gold_scores: list[float] = []
        model_names: list[str] = []
        for comp in example["completions"]:
            text = comp.get("response", comp.get("completion", comp.get("text", "")))
            candidates.append(str(text).strip())
            gold_scores.append(ultrafeedback_completion_score(comp, mode=score_mode))
            model_names.append(str(comp.get("model", "unknown")))
        return {
            "prompt": prompt,
            "candidates": candidates,
            "reward_scores": gold_scores,
            "models": model_names,
            "source": example.get("source", ""),
            "instruction_id": example.get("id", ""),
        }

    return ds.map(_convert, remove_columns=ds.column_names)


def convert_nectar_to_listwise(
    split: str = "train",
    dataset_name: str = "berkeley-nest/Nectar",
    max_examples: int | None = None,
    start: int = 0,
    shuffle_seed: int | None = None,
) -> Dataset:
    ds = load_dataset(dataset_name, split=split)
    if shuffle_seed is not None:
        ds = ds.shuffle(seed=shuffle_seed)
    if start:
        ds = ds.select(range(start, len(ds)))
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))

    def _convert(example: dict[str, Any]) -> dict[str, Any]:
        prompt = str(example["prompt"])
        if not prompt.strip().endswith("Assistant:"):
            prompt = prompt.rstrip() + "\n\nAssistant:"
        candidates = [str(ans["answer"]).strip() for ans in example["answers"]]
        reward_scores = [float(-int(ans["rank"])) for ans in example["answers"]]
        models = [str(ans.get("model", "unknown")) for ans in example["answers"]]
        return {
            "prompt": prompt,
            "candidates": candidates,
            "reward_scores": reward_scores,
            "models": models,
            "num_responses": int(example.get("num_response", len(candidates))),
        }

    return ds.map(_convert, remove_columns=ds.column_names)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset conversion helpers for RIPLM/DDO LLM benchmarks.")
    parser.add_argument("--dataset", choices=["uf_binarized", "uf_listwise", "nectar"], required=True)
    parser.add_argument("--split", default="train_prefs")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--score_mode", default="mean_rating")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--shuffle_seed", type=int, default=None)
    args = parser.parse_args()

    if args.dataset == "uf_binarized":
        ds = load_ultrafeedback_binarized(args.split)
    elif args.dataset == "uf_listwise":
        ds = convert_openbmb_ultrafeedback_to_listwise(
            split=args.split,
            score_mode=args.score_mode,
            max_examples=args.max_examples,
            start=args.start,
            shuffle_seed=args.shuffle_seed,
        )
    else:
        ds = convert_nectar_to_listwise(
            split=args.split,
            max_examples=args.max_examples,
            start=args.start,
            shuffle_seed=args.shuffle_seed,
        )

    ds.save_to_disk(args.output_dir)
    print(f"Saved {len(ds)} rows to {args.output_dir}")


if __name__ == "__main__":
    main()
