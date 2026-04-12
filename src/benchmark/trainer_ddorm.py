
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase, Trainer

from .ddorm import ddo_cross_entropy_loss, ddo_target_distribution
from .scoring import batched_reward_scores, batched_sequence_scores


@dataclass
class DDORMCollator:
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "prompts": [f["prompt"] for f in features],
            "candidates": [f["candidates"] for f in features],
            "reward_scores": [f.get("reward_scores") for f in features],
        }


class DDORMTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        reward_model: PreTrainedModel | None = None,
        reward_tokenizer: PreTrainedTokenizerBase | None = None,
        policy_tokenizer: PreTrainedTokenizerBase | None = None,
        ddorm_eta: float = 1.0,
        decision_temperature: float = 1.0,
        max_length: int = 1024,
        use_gold_rewards: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.reward_model = reward_model
        self.reward_tokenizer = reward_tokenizer or policy_tokenizer
        self.policy_tokenizer = policy_tokenizer
        self.ddorm_eta = ddorm_eta
        self.decision_temperature = decision_temperature
        self.max_length = max_length
        self.use_gold_rewards = use_gold_rewards
        if self.reward_model is not None:
            self.reward_model.eval()
            for param in self.reward_model.parameters():
                param.requires_grad = False

    def compute_loss(self, model: PreTrainedModel, inputs: dict[str, Any], return_outputs: bool = False, **kwargs: Any):
        prompts = inputs["prompts"]
        candidates = inputs["candidates"]
        batch_size = len(prompts)
        num_candidates = len(candidates[0])
        if any(len(c) != num_candidates for c in candidates):
            raise ValueError("All examples in a batch must have the same candidate count.")

        flat_prompts = [prompt for prompt, cand_list in zip(prompts, candidates) for _ in cand_list]
        flat_candidates = [cand for cand_list in candidates for cand in cand_list]

        seq_scores = batched_sequence_scores(
            model,
            self.policy_tokenizer,
            flat_prompts,
            flat_candidates,
            max_length=self.max_length,
            device=model.device,
        ).avg_logprob.view(batch_size, num_candidates)

        if self.use_gold_rewards:
            reward_scores = torch.tensor(inputs["reward_scores"], dtype=seq_scores.dtype, device=seq_scores.device)
        else:
            if self.reward_model is None or self.reward_tokenizer is None:
                raise ValueError("Reward model/tokenizer must be provided when use_gold_rewards=False")
            policy_device = next(model.parameters()).device
            reward_device = next(self.reward_model.parameters()).device
            if reward_device != policy_device:
                self.reward_model = self.reward_model.to(policy_device)
            reward_scores = batched_reward_scores(
                self.reward_model,
                self.reward_tokenizer,
                flat_prompts,
                flat_candidates,
                max_length=self.max_length,
                device=policy_device,
            ).view(batch_size, num_candidates)

        p, q, centered = ddo_target_distribution(
            seq_scores,
            reward_scores,
            eta=self.ddorm_eta,
            temperature=self.decision_temperature,
        )
        loss = ddo_cross_entropy_loss(seq_scores, q, temperature=self.decision_temperature)
        if return_outputs:
            return loss, {
                "policy_scores": seq_scores.detach(),
                "policy_distribution": p.detach(),
                "target_distribution": q.detach(),
                "reward_scores": reward_scores.detach(),
                "centered_reward": centered.detach(),
            }
        return loss
