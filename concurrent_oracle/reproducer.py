"""Repeat a plan and report a reproduction rate."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

from .config import EngineConfig
from .contract_registry import ContractRegistry
from .oracle_engine import evaluate_oracles
from .runner import run_plan
from .schemas import ConcurrentTestPlan


def reproduce_plan(
    plan: ConcurrentTestPlan,
    generated_harness: Path,
    cfg: EngineConfig,
    *,
    repeats: int = 5,
) -> Dict:
    registry = ContractRegistry(cfg.contract_dir)
    statuses = []
    results = []
    for _ in range(max(1, repeats)):
        observations = run_plan(Path(generated_harness), plan, cfg)
        result = evaluate_oracles(plan, observations, registry)
        results.append(result)
        statuses.append([c.status for c in result.oracle_results])
    triggers = sum(1 for s in statuses if any(x in ("VIOLATION", "INCONCLUSIVE") for x in s))
    return {
        "plan_id": plan.plan_id,
        "repeats": len(statuses),
        "triggers": triggers,
        "reproduction_rate": triggers / len(statuses) if statuses else 0.0,
        "statuses": statuses,
        "last_result": results[-1].to_dict() if results else None,
    }
