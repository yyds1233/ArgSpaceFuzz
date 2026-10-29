"""Small module-level callable for the O7 tracing control."""
from __future__ import annotations

import torch


def tiny_fn(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x) + 1.0


class TinyModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x) + 1.0
