"""Small CEM planner matching the LeWM checkpoint model interface."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable
import warnings

import torch


@dataclass(frozen=True)
class CEMConfig:
    """Intrinsic Cross-Entropy Method optimizer parameters."""

    num_samples: int = 64
    num_elites: int = 8
    num_iters: int = 5
    init_std: float = 1.0


@dataclass(frozen=True)
class CEMResult:
    """CEM output with executable future actions and diagnostic traces."""

    actions: torch.Tensor
    action_sequence: torch.Tensor
    costs: list[float]
    average_costs: list[float]
    mean: torch.Tensor
    std: torch.Tensor


def steps_to_blocks(num_steps: int, action_block: int) -> int:
    """Convert raw environment-action steps to LeWM action blocks."""
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}.")
    if num_steps % action_block != 0:
        raise ValueError(
            f"num_steps={num_steps} must be divisible by action_block={action_block} "
            "for the current LeWM action-block representation."
        )
    return num_steps // action_block


def steps_to_blocks_ceil(
    num_steps: int,
    action_block: int,
    *,
    name: str = "num_steps",
) -> int:
    """Convert raw env steps to enough LeWM blocks, warning on partial blocks."""
    if num_steps <= 0:
        raise ValueError(f"{name} must be positive, got {num_steps}.")
    blocks = math.ceil(num_steps / action_block)
    effective_steps = blocks * action_block
    if effective_steps != num_steps:
        warnings.warn(
            f"{name}={num_steps} is not divisible by action_block={action_block}; "
            f"using {blocks} LeWM block(s), equivalent to {effective_steps} optimized "
            f"raw step(s). Execution will be clipped to the requested {num_steps} step(s).",
            stacklevel=2,
        )
    return blocks


def plan_cem(
    model,
    info: dict[str, torch.Tensor],
    *,
    planner_horizon: int = 25,
    env_action_dim: int = 2,
    action_block: int = 5,
    optimizer: CEMConfig | None = None,
    constraint: Callable[[torch.Tensor], torch.Tensor] | None = None,
    constraint_weight: float = 0.0,
    device: str | torch.device | None = None,
) -> CEMResult:
    """Optimize future env actions with CEM.

    LeWM conditions on a fixed action-history prefix, so the sampled tensor sent
    to the model is longer than the executable future action sequence returned
    in ``CEMResult.actions``.
    """
    optimizer = optimizer or CEMConfig()
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    future_blocks = steps_to_blocks_ceil(
        planner_horizon,
        action_block,
        name="planner_horizon",
    )
    history = info["pixels"].shape[2]
    horizon_blocks = history + future_blocks
    action_dim = env_action_dim * action_block
    model = model.to(device).eval()
    mean = torch.zeros(horizon_blocks, action_dim, device=device)
    std = torch.full_like(mean, optimizer.init_std)
    prefix = info["action"][0, 0, :history].to(device)

    costs_trace: list[float] = []
    average_costs_trace: list[float] = []
    with torch.inference_mode():
        for _ in range(optimizer.num_iters):
            candidates = mean + std * torch.randn(
                optimizer.num_samples, horizon_blocks, action_dim, device=device
            )
            candidates[0] = mean
            candidates[:, : prefix.shape[0]] = prefix

            expanded = {}
            for key, value in info.items():
                expanded[key] = value.to(device).repeat(
                    1, optimizer.num_samples, *([1] * (value.ndim - 2))
                )

            costs = model.get_cost(expanded, candidates.unsqueeze(0))
            costs = costs.squeeze(0)
            if constraint is not None and constraint_weight:
                costs = costs + constraint_weight * constraint(candidates[:, history:])
            elite_idx = torch.argsort(costs)[: optimizer.num_elites]
            elites = candidates[elite_idx]
            mean = elites.mean(dim=0)
            std = elites.std(dim=0).clamp_min(1e-4)
            mean[: prefix.shape[0]] = prefix
            std[: prefix.shape[0]] = 1e-4
            costs_trace.append(float(costs[elite_idx[0]].detach().cpu()))
            average_costs_trace.append(float(costs.mean().detach().cpu()))

    return CEMResult(
        actions=mean[history:].detach().cpu(),
        action_sequence=mean.detach().cpu(),
        costs=costs_trace,
        average_costs=average_costs_trace,
        mean=mean[history:].detach().cpu(),
        std=std[history:].detach().cpu(),
    )
