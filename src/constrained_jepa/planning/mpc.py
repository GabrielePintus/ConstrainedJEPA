"""Receding-horizon planning helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from stable_worldmodel.data import HDF5Dataset

from constrained_jepa.data import StartGoalBatch, make_start_goal_from_episode
from constrained_jepa.planning.cem import CEMConfig, CEMResult, plan_cem, steps_to_blocks_ceil


@dataclass(frozen=True)
class MPCStep:
    """One replan in a receding-horizon loop."""

    batch: StartGoalBatch
    result: CEMResult
    executed_blocks: torch.Tensor


def plan_dataset_mpc(
    model,
    dataset: HDF5Dataset,
    initial_batch: StartGoalBatch,
    *,
    planner_horizon: int = 50,
    replan_frequency: int = 5,
    trajectory_horizon: int = 40,
    env_action_dim: int = 2,
    action_block: int = 5,
    action_mean: torch.Tensor | None = None,
    action_std: torch.Tensor | None = None,
    optimizer: CEMConfig | None = None,
    constraint: Callable[[torch.Tensor], torch.Tensor] | None = None,
    constraint_weight: float = 0.0,
    device: str | torch.device | None = None,
) -> list[MPCStep]:
    """Run teacher-forced receding-horizon planning over a dataset episode.

    This replans from observations at later dataset timesteps while keeping the
    original goal fixed. It exercises the MPC optimization pattern, but does not
    claim environment closed-loop execution because the current observation
    comes from the dataset rather than from stepping planned actions in an env.
    """
    optimizer = optimizer or CEMConfig()
    replan_blocks = steps_to_blocks_ceil(
        replan_frequency,
        action_block,
        name="replan_frequency",
    )
    steps_to_blocks_ceil(planner_horizon, action_block, name="planner_horizon")
    steps_to_blocks_ceil(trajectory_horizon, action_block, name="trajectory_horizon")

    steps: list[MPCStep] = []
    current_step = initial_batch.start_step
    goal_step = initial_batch.goal_step
    final_step = min(goal_step, initial_batch.start_step + trajectory_horizon)
    raw_step_advance = replan_frequency

    while current_step < final_step:
        if current_step + raw_step_advance > final_step:
            break

        batch = make_start_goal_from_episode(
            dataset,
            episode_row=initial_batch.episode_row,
            start_step=current_step,
            goal_step=goal_step,
            history=initial_batch.info["pixels"].shape[2],
            action_block=action_block,
            env_action_dim=env_action_dim,
            action_mean=action_mean,
            action_std=action_std,
        )
        result = plan_cem(
            model,
            batch.info,
            planner_horizon=planner_horizon,
            env_action_dim=env_action_dim,
            action_block=action_block,
            optimizer=optimizer,
            constraint=constraint,
            constraint_weight=constraint_weight,
            device=device,
        )
        executed_blocks = result.actions[:replan_blocks]
        steps.append(MPCStep(batch=batch, result=result, executed_blocks=executed_blocks))
        current_step += raw_step_advance

    return steps
