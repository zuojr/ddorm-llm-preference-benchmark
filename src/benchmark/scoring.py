from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class SequenceScoreOutput:
    avg_logprob: torch.Tensor
    sum_logprob: torch.Tensor
    token_count: torch.Tensor


def _pad_batch(sequences: list[torch.Tensor], pad_value: int) -> torch.Tensor:
    max_len = max(seq.size(0) for seq in sequences)
    out = torch.full((len(sequences), max_len), pad_value, dtype=sequences[0].dtype)
    for idx, seq in enumerate(sequences):
        out[idx, : seq.size(0)] = seq
    return out


def build_completion_only_batch(
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    completions: Sequence[str],
    *,
    max_length: int = 1024,
    add_eos: bool = True,
) -> dict[str, torch.Tensor]:
    input_ids_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    attention_masks: list[torch.Tensor] = []

    for prompt, completion in zip(prompts, completions):
        prompt_ids = tokenizer(
            prompt, add_special_tokens=False, truncation=False, return_attention_mask=False
        )["input_ids"]
        full_text = prompt + completion + (tokenizer.eos_token if add_eos and tokenizer.eos_token else "")
        full_ids = tokenizer(
            full_text, add_special_tokens=False, truncation=False, return_attention_mask=False
        )["input_ids"]

        if len(prompt_ids) >= len(full_ids):
            continue

        if len(full_ids) > max_length:
            full_ids = full_ids[-max_length:]
            prompt_ids = prompt_ids[-max_length:]

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        prompt_len = min(len(prompt_ids), len(full_ids) - 1)
        labels[:prompt_len] = -100

        input_ids_list.append(input_ids)
        attention_masks.append(attention_mask)
        labels_list.append(labels)

    if not input_ids_list:
        raise ValueError("No valid examples were produced while building completion-only batch.")

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    return {
        "input_ids": _pad_batch(input_ids_list, pad_token_id),
        "attention_mask": _pad_batch(attention_masks, 0),
        "labels": _pad_batch(labels_list, -100),
    }


def batched_sequence_scores(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    completions: Sequence[str],
    *,
    max_length: int = 1024,
    batch_size: int = 8,
    device: torch.device | None = None,
) -> SequenceScoreOutput:
    if device is None:
        device = next(model.parameters()).device

    all_avg = []
    all_sum = []
    all_count = []

    for start in range(0, len(prompts), batch_size):
        sub_prompts = prompts[start:start + batch_size]
        sub_completions = completions[start:start + batch_size]

        batch = build_completion_only_batch(
            tokenizer, sub_prompts, sub_completions, max_length=max_length
        )
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        logits = outputs.logits[:, :-1, :]
        labels = batch["labels"][:, 1:]

        valid = labels.ne(-100)
        safe_labels = labels.masked_fill(~valid, 0)
        token_logprobs = F.log_softmax(logits, dim=-1).gather(
            dim=-1, index=safe_labels.unsqueeze(-1)
        ).squeeze(-1)
        token_logprobs = token_logprobs * valid

        sum_logprob = token_logprobs.sum(dim=-1)
        token_count = valid.sum(dim=-1).clamp_min(1)
        avg_logprob = sum_logprob / token_count

        all_avg.append(avg_logprob)
        all_sum.append(sum_logprob)
        all_count.append(token_count)

        del batch, outputs, logits, labels, valid, safe_labels, token_logprobs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return SequenceScoreOutput(
        avg_logprob=torch.cat(all_avg, dim=0),
        sum_logprob=torch.cat(all_sum, dim=0),
        token_count=torch.cat(all_count, dim=0),
    )


@torch.no_grad()
def batched_reward_scores(
    reward_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    completions: Sequence[str],
    *,
    max_length: int = 1024,
    batch_size: int = 8,
    device: torch.device | None = None,
) -> torch.Tensor:
    if device is None:
        device = next(reward_model.parameters()).device

    all_scores = []

    for start in range(0, len(prompts), batch_size):
        sub_prompts = prompts[start:start + batch_size]
        sub_completions = completions[start:start + batch_size]
        texts = [p + c for p, c in zip(sub_prompts, sub_completions)]

        batch = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = reward_model(**batch).logits.squeeze(-1)
        all_scores.append(logits)

        del batch, logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return torch.cat(all_scores, dim=0)
