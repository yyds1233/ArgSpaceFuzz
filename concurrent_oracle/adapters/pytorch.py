"""PyTorch adapter."""
from __future__ import annotations

import importlib

from .base import FrameworkAdapter


class PyTorchAdapter(FrameworkAdapter):
    name = "pytorch"

    def resolve_target(self, api_name: str):
        module_name, _, attr = api_name.rpartition(".")
        if not module_name:
            raise ValueError(f"invalid API name: {api_name}")
        return getattr(importlib.import_module(module_name), attr)

    def smoke(self) -> dict:
        try:
            import torch
            return {"framework": self.name, "ok": True, "version": torch.__version__}
        except Exception as exc:
            return {"framework": self.name, "ok": False, "reason": str(exc)}
