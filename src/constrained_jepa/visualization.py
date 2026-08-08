"""Plotting helpers for notebooks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_start_goal(start_image, goal_image):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(start_image)
    axes[0].set_title("Start")
    axes[0].axis("off")
    axes[1].imshow(goal_image)
    axes[1].set_title("Goal")
    axes[1].axis("off")
    fig.tight_layout()
    return fig


def plot_plan(
    actions: torch.Tensor,
    costs: list[float],
    average_costs: list[float] | None = None,
):
    del actions
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(costs, marker="o", label="best")
    if average_costs is not None:
        ax.plot(average_costs, marker="o", label="average")
        ax.legend()
    ax.set_title("CEM Cost")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")
    fig.tight_layout()
    return fig


def write_mp4(frames, path: str | Path, *, fps: int = 10) -> Path:
    """Write RGB frames to an MP4 file."""
    import imageio.v3 as iio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    video = np.asarray(frames, dtype=np.uint8)
    iio.imwrite(path, video, fps=fps, codec="libx264", macro_block_size=1)
    return path


def plot_mpc_costs(
    step_costs: list[list[float]],
    step_average_costs: list[list[float]] | None = None,
):
    fig, ax = plt.subplots(figsize=(6, 3))
    final_costs = [costs[-1] for costs in step_costs if costs]
    ax.plot(final_costs, marker="o", label="best")
    if step_average_costs is not None:
        final_average_costs = [costs[-1] for costs in step_average_costs if costs]
        ax.plot(final_average_costs, marker="o", label="average")
        ax.legend()
    ax.set_title("Replanning CEM Cost")
    ax.set_xlabel("Replan")
    ax.set_ylabel("Final CEM Cost")
    fig.tight_layout()
    return fig


def plot_env_trajectory(frames, positions, every: int = 5):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(frames[-1])
    axes[0].set_title("Final Render")
    axes[0].axis("off")

    pos = torch.as_tensor(positions).float()
    axes[1].plot(pos[:, 0], pos[:, 1], marker="o", markersize=3)
    axes[1].scatter(pos[0, 0], pos[0, 1], label="start")
    axes[1].scatter(pos[-1, 0], pos[-1, 1], label="end")
    axes[1].set_xlim(0, 224)
    axes[1].set_ylim(224, 0)
    axes[1].set_aspect("equal")
    axes[1].set_title(f"Env Trajectory ({max(1, every)}-step markers)")
    axes[1].legend()
    fig.tight_layout()
    return fig
