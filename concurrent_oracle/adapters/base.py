"""Framework adapter interface."""
from __future__ import annotations

from typing import Any


class FrameworkAdapter:
    name = "base"

    def resolve_target(self, api_name: str) -> Any:
        raise NotImplementedError

    def smoke(self) -> dict:
        return {"framework": self.name, "ok": False, "reason": "not implemented"}
