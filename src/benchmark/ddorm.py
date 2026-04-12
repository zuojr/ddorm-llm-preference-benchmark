
from __future__ import annotations

import torch
import torch.nn.functional as F


def ddo_target_distribution(
    policy_scores: torch.Tensor,
    reward_scores: torch.Tensor,
    *,
    eta: float = 1.0,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p = torch.softmax(policy_scores / temperature, dim=-1)
    centered_reward = reward_scores - (p * reward_scores).sum(dim=-1, keepdim=True)
    next_scores = policy_scores + eta * centered_reward
    q = torch.softmax(next_scores / temperature, dim=-1)
    return p, q, centered_reward


def ddo_cross_entropy_loss(policy_scores: torch.Tensor, target_q: torch.Tensor, *, temperature: float = 1.0) -> torch.Tensor:
    logp = F.log_softmax(policy_scores / temperature, dim=-1)
    return -(target_q * logp).sum(dim=-1).mean()
