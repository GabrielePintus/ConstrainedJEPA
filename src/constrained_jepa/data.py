"""Dataset sampling helpers for LeWM planning notebooks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from stable_worldmodel.data import HDF5Dataset


@dataclass(frozen=True)
class StartGoalBatch:
    """A single start/goal planning problem sampled from an HDF5 dataset."""

    info: dict[str, torch.Tensor]
    start_index: int
    goal_index: int
    episode_row: int
    episode_id: int
    start_step: int
    goal_step: int
    raw_pixels_history: np.ndarray
    raw_start_pixels: np.ndarray
    raw_goal_pixels: np.ndarray


def _episode_column(dataset: HDF5Dataset) -> str:
    if "episode_idx" in dataset.column_names:
        return "episode_idx"
    if "ep_idx" in dataset.column_names:
        return "ep_idx"
    raise KeyError("Dataset has neither 'episode_idx' nor 'ep_idx'.")


def image_to_imagenet_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert an HWC uint8 image to the normalized CHW tensor used by LeWM eval."""
    tensor = torch.as_tensor(image).permute(2, 0, 1).float() / 255.0
    mean = tensor.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = tensor.new_tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std


def fit_action_stats(dataset: HDF5Dataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean/std tensors for dataset action standardization."""
    actions = dataset.get_col_data("action").astype(np.float32)
    actions = actions[np.isfinite(actions).all(axis=1)]
    mean = torch.from_numpy(actions.mean(axis=0))
    std = torch.from_numpy(actions.std(axis=0))
    return mean, std.clamp_min(1e-6)


def sample_start_goal(
    dataset: HDF5Dataset,
    *,
    rng: np.random.Generator | None = None,
    history: int = 3,
    action_block: int = 5,
    goal_offset: int = 25,
    env_action_dim: int | None = None,
    action_mean: torch.Tensor | None = None,
    action_std: torch.Tensor | None = None,
) -> StartGoalBatch:
    """Sample one reachable start/goal pair and build a LeWM `info_dict`.

    The returned tensors use shape `(B, S, T, ...)` with `B=S=1`, matching the
    LeWM `get_cost` implementation copied from the released code.
    """
    rng = rng or np.random.default_rng()
    episode_col = _episode_column(dataset)
    episode_ids = np.unique(dataset.get_col_data(episode_col))

    lengths = dataset.get_col_data("ep_len")
    offsets = dataset.get_col_data("ep_offset")
    valid = np.flatnonzero(lengths >= history + goal_offset + 1)
    if len(valid) == 0:
        raise ValueError("No episode is long enough for the requested sampling parameters.")

    for _ in range(100):
        episode_row = int(rng.choice(valid))
        episode_id = int(episode_ids[episode_row]) if len(episode_ids) == len(lengths) else episode_row
        max_start = int(lengths[episode_row] - goal_offset - 1)
        start_step = int(rng.integers(history - 1, max_start + 1))
        goal_step = start_step + goal_offset

        start_index = int(offsets[episode_row] + start_step)
        goal_index = int(offsets[episode_row] + goal_step)
        hist_start = start_index - history + 1

        if np.isfinite(dataset.get_row_data(list(range(hist_start, start_index + 1)))["action"]).all():
            break
    else:
        raise ValueError("Could not sample a finite action history after 100 attempts.")

    return make_start_goal_from_episode(
        dataset,
        episode_row=episode_row,
        start_step=start_step,
        goal_step=goal_step,
        history=history,
        action_block=action_block,
        env_action_dim=env_action_dim,
        action_mean=action_mean,
        action_std=action_std,
    )


def make_start_goal_from_episode(
    dataset: HDF5Dataset,
    *,
    episode_row: int,
    start_step: int,
    goal_step: int,
    history: int = 3,
    action_block: int = 5,
    env_action_dim: int | None = None,
    action_mean: torch.Tensor | None = None,
    action_std: torch.Tensor | None = None,
) -> StartGoalBatch:
    """Build a LeWM planning problem from a specific episode/start/goal."""
    episode_col = _episode_column(dataset)
    episode_ids = np.unique(dataset.get_col_data(episode_col))
    lengths = dataset.get_col_data("ep_len")
    offsets = dataset.get_col_data("ep_offset")

    if start_step < history - 1:
        raise ValueError(f"start_step must be at least {history - 1}, got {start_step}.")
    if goal_step >= int(lengths[episode_row]):
        raise ValueError(
            f"goal_step {goal_step} is outside episode {episode_row} length {lengths[episode_row]}."
        )

    episode_id = int(episode_ids[episode_row]) if len(episode_ids) == len(lengths) else episode_row
    start_index = int(offsets[episode_row] + start_step)
    goal_index = int(offsets[episode_row] + goal_step)
    hist_start = start_index - history + 1

    rows = dataset.get_row_data(list(range(hist_start, start_index + 1)))
    goal = dataset.get_row_data(goal_index)

    if not np.isfinite(rows["action"]).all():
        raise ValueError("Selected action history contains non-finite values.")

    pixels = torch.stack([image_to_imagenet_tensor(img) for img in rows["pixels"]])
    goal_pixels = image_to_imagenet_tensor(goal["pixels"]).unsqueeze(0)

    base_actions = torch.as_tensor(rows["action"], dtype=torch.float32)
    if action_mean is not None and action_std is not None:
        base_actions = (base_actions - action_mean) / action_std

    action_dim = int(env_action_dim or base_actions.shape[-1])
    if base_actions.shape[-1] != action_dim:
        raise ValueError(
            f"Expected raw action dimension {action_dim}, got {base_actions.shape[-1]}."
        )
    action_blocks = torch.zeros(history, action_dim * action_block, dtype=torch.float32)
    for t in range(history):
        action_blocks[t] = base_actions[t].repeat(action_block)

    info = {
        "pixels": pixels.unsqueeze(0).unsqueeze(0),
        "goal": goal_pixels.unsqueeze(0).unsqueeze(0),
        "action": action_blocks.unsqueeze(0).unsqueeze(0),
    }

    return StartGoalBatch(
        info=info,
        start_index=start_index,
        goal_index=goal_index,
        episode_row=episode_row,
        episode_id=episode_id,
        start_step=start_step,
        goal_step=goal_step,
        raw_pixels_history=rows["pixels"],
        raw_start_pixels=rows["pixels"][-1],
        raw_goal_pixels=goal["pixels"],
    )
