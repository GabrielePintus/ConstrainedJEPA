"""Action-space constraints for test-time planning."""

from __future__ import annotations

import torch


def action_l2(actions: torch.Tensor) -> torch.Tensor:
    """Return per-candidate mean squared action magnitude.

    Args:
        actions: Tensor shaped `(num_samples, horizon, action_dim)`.
    """
    return actions.square().mean(dim=(1, 2))
