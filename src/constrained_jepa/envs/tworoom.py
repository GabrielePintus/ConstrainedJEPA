"""Utilities for running LeWM plans in the real TwoRoom environment."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from stable_worldmodel.data import HDF5Dataset

from constrained_jepa.data import StartGoalBatch, image_to_imagenet_tensor
from constrained_jepa.planning.cem import CEMConfig, plan_cem, steps_to_blocks_ceil


@dataclass(frozen=True)
class EnvRollout:
    frames: list[np.ndarray]
    positions: list[np.ndarray]
    distances: list[float]
    actions: np.ndarray
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class EnvMPCStep:
    start_step: int
    costs: list[float]
    average_costs: list[float]
    executed_blocks: torch.Tensor
    distance_after: float


@dataclass(frozen=True)
class EnvMPCResult:
    frames: list[np.ndarray]
    positions: list[np.ndarray]
    distances: list[float]
    steps: list[EnvMPCStep]
    terminated: bool
    truncated: bool


class TwoRoomSuccessThreshold(gym.Wrapper):
    """Override TwoRoom termination with a configurable distance threshold."""

    def __init__(self, env, success_threshold: float):
        super().__init__(env)
        self.success_threshold = float(success_threshold)

    def step(self, action):
        obs, reward, _terminated, truncated, info = self.env.step(action)
        terminated = float(info["distance_to_target"]) < self.success_threshold
        return obs, reward, terminated, truncated, info


def make_tworoom_env(*, render_target: bool = True, success_threshold: float = 16.0):
    import stable_worldmodel.envs  # noqa: F401 - registers swm/* env ids

    env = gym.make(
        "swm/TwoRoom-v1",
        render_mode="rgb_array",
        render_target=render_target,
        disable_env_checker=True,
    )
    return TwoRoomSuccessThreshold(env, success_threshold)


def reset_tworoom_to_batch(env, dataset: HDF5Dataset, batch: StartGoalBatch):
    """Reset env to the dataset start position and use the sampled future state as goal."""
    start = dataset.get_row_data(batch.start_index)
    goal = dataset.get_row_data(batch.goal_index)
    obs, info = env.reset(
        options={
            "state": start["proprio"].astype(np.float32),
            "target_state": goal["proprio"].astype(np.float32),
        }
    )
    return obs, info


def action_blocks_to_raw_actions(
    action_blocks: torch.Tensor,
    *,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    env_action_dim: int = 2,
    action_block: int = 5,
) -> np.ndarray:
    """Convert standardized LeWM action blocks to raw env actions."""
    blocks = action_blocks.detach().cpu().reshape(-1, action_block, env_action_dim)
    mean = action_mean.detach().cpu().reshape(1, 1, env_action_dim)
    std = action_std.detach().cpu().reshape(1, 1, env_action_dim)
    actions = blocks * std + mean
    return actions.reshape(-1, env_action_dim).numpy().clip(-1.0, 1.0)


def build_info_from_history(
    *,
    frames: list[np.ndarray],
    goal_frame: np.ndarray,
    action_blocks: torch.Tensor,
    history: int = 3,
) -> dict[str, torch.Tensor]:
    """Build a LeWM info dict from recent rendered frames and action-block history."""
    recent_frames = frames[-history:]
    if len(recent_frames) < history:
        recent_frames = [recent_frames[0]] * (history - len(recent_frames)) + recent_frames

    recent_actions = action_blocks[-history:]
    if recent_actions.shape[0] < history:
        pad = recent_actions[:1].repeat(history - recent_actions.shape[0], 1)
        recent_actions = torch.cat([pad, recent_actions], dim=0)

    pixels = torch.stack([image_to_imagenet_tensor(frame) for frame in recent_frames])
    goal = image_to_imagenet_tensor(goal_frame).unsqueeze(0)
    return {
        "pixels": pixels.unsqueeze(0).unsqueeze(0),
        "goal": goal.unsqueeze(0).unsqueeze(0),
        "action": recent_actions.unsqueeze(0).unsqueeze(0),
    }


def rollout_action_blocks(
    env,
    action_blocks: torch.Tensor,
    *,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    env_action_dim: int = 2,
    action_block: int = 5,
    num_steps: int | None = None,
) -> EnvRollout:
    """Execute a planned action-block sequence open-loop in the env."""
    raw_actions = action_blocks_to_raw_actions(
        action_blocks,
        action_mean=action_mean,
        action_std=action_std,
        env_action_dim=env_action_dim,
        action_block=action_block,
    )
    if num_steps is not None:
        raw_actions = raw_actions[:num_steps]
    frames = [env.render()]
    positions = [env.unwrapped.agent_position.detach().cpu().numpy().copy()]
    distances = []
    terminated = truncated = False

    for action in raw_actions:
        _, _, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        positions.append(env.unwrapped.agent_position.detach().cpu().numpy().copy())
        distances.append(float(info["distance_to_target"]))
        if terminated or truncated:
            break

    return EnvRollout(
        frames=frames,
        positions=positions,
        distances=distances,
        actions=raw_actions[: len(frames) - 1],
        terminated=terminated,
        truncated=truncated,
    )


def run_env_mpc(
    model,
    env,
    dataset: HDF5Dataset,
    batch: StartGoalBatch,
    *,
    planner_horizon: int,
    replan_frequency: int,
    trajectory_horizon: int,
    env_action_dim: int = 2,
    action_block: int = 5,
    history: int = 3,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    optimizer: CEMConfig,
    device: str | torch.device | None = None,
) -> EnvMPCResult:
    """Run real closed-loop MPC in TwoRoom using rendered env observations."""
    reset_tworoom_to_batch(env, dataset, batch)
    replan_blocks = steps_to_blocks_ceil(
        replan_frequency,
        action_block,
        name="replan_frequency",
    )
    steps_to_blocks_ceil(planner_horizon, action_block, name="planner_horizon")
    steps_to_blocks_ceil(trajectory_horizon, action_block, name="trajectory_horizon")

    frames = [*batch.raw_pixels_history[-history:], env.render()]
    positions = [env.unwrapped.agent_position.detach().cpu().numpy().copy()]
    distances: list[float] = []
    action_history = batch.info["action"][0, 0].detach().cpu()
    steps: list[EnvMPCStep] = []
    terminated = truncated = False
    elapsed = 0

    while elapsed < trajectory_horizon and not (terminated or truncated):
        info = build_info_from_history(
            frames=frames,
            goal_frame=batch.raw_goal_pixels,
            action_blocks=action_history,
            history=history,
        )
        plan = plan_cem(
            model,
            info,
            planner_horizon=planner_horizon,
            env_action_dim=env_action_dim,
            action_block=action_block,
            optimizer=optimizer,
            device=device,
        )
        executed_blocks = plan.actions[:replan_blocks]
        raw_actions = action_blocks_to_raw_actions(
            executed_blocks,
            action_mean=action_mean,
            action_std=action_std,
            env_action_dim=env_action_dim,
            action_block=action_block,
        )[:replan_frequency]

        distance_after = float("nan")
        for action in raw_actions:
            _, _, terminated, truncated, step_info = env.step(action)
            frames.append(env.render())
            positions.append(env.unwrapped.agent_position.detach().cpu().numpy().copy())
            distance_after = float(step_info["distance_to_target"])
            distances.append(distance_after)
            elapsed += 1
            if elapsed >= trajectory_horizon or terminated or truncated:
                break

        action_history = torch.cat([action_history, executed_blocks], dim=0)
        steps.append(
            EnvMPCStep(
                start_step=elapsed,
                costs=plan.costs,
                average_costs=plan.average_costs,
                executed_blocks=executed_blocks,
                distance_after=distance_after,
            )
        )

    return EnvMPCResult(
        frames=frames,
        positions=positions,
        distances=distances,
        steps=steps,
        terminated=terminated,
        truncated=truncated,
    )
