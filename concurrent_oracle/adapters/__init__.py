"""Framework adapters."""
from __future__ import annotations

from .base import FrameworkAdapter
from .pytorch import PyTorchAdapter
from .tensorflow import TensorFlowAdapter


def get_adapter(name: str) -> FrameworkAdapter:
    if name == "pytorch":
        return PyTorchAdapter()
    if name == "tensorflow":
        return TensorFlowAdapter()
    return FrameworkAdapter()
