"""Configuration and output-directory helpers."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUTS = MODULE_ROOT / "outputs"
DEFAULT_CONTRACTS = MODULE_ROOT / "contracts"


@dataclass
class EngineConfig:
    outputs_dir: Path = DEFAULT_OUTPUTS
    python: str = sys.executable
    use_llm: bool = False
    model: str = os.environ.get("CONCURRENT_ORACLE_MODEL", "gpt-5.4-mini")
    base_url: Optional[str] = os.environ.get("OPENAI_BASE_URL")
    temperature: float = 0.0
    max_tokens: int = 1500
    external_threads: List[int] = field(default_factory=lambda: [2, 4])
    internal_threads: List[int] = field(default_factory=lambda: [1, 4])
    trials: int = 10
    timeout_seconds: int = 60
    max_repair_rounds: int = 2
    contract_dir: Path = DEFAULT_CONTRACTS
    framework: str = "pytorch"
    seed: int = 0

    def ensure_outputs(self) -> None:
        for name in (
            "artifacts", "plans", "generated_harnesses", "logs",
            "oracle_results", "reproducers", "reports", "validation",
            "cases", "runs",
        ):
            (self.outputs_dir / name).mkdir(parents=True, exist_ok=True)

    def path(self, kind: str, name: str) -> Path:
        p = self.outputs_dir / kind / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


def default_config(**overrides) -> EngineConfig:
    cfg = EngineConfig()
    for k, v in overrides.items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.ensure_outputs()
    return cfg
