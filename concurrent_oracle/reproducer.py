"""Repeat a plan and report distinct reproduction metrics.

``VIOLATION`` and ``INCONCLUSIVE`` are intentionally never merged into one
"defect reproduction rate".
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from .config import EngineConfig
from .contract_registry import ContractRegistry
from .oracle_engine import evaluate_oracles
from .runner import run_plan
from .schemas import ConcurrentTestPlan


def summarize_statuses(statuses: List[List[str]], confirmed: Optional[int] = None) -> Dict:
    repeats = len(statuses)
    violation_hits = sum(1 for s in statuses if "VIOLATION" in s)
    inconclusive_hits = sum(1 for s in statuses if "INCONCLUSIVE" in s)
    candidate_hits = sum(1 for s in statuses if any(x in ("VIOLATION", "INCONCLUSIVE") for x in s))

    def rate(n: int):
        return (n / repeats) if repeats else 0.0

    return {
        "repeats": repeats,
        "violation_hits": violation_hits,
        "candidate_hits": candidate_hits,
        "inconclusive_hits": inconclusive_hits,
        "violation_reproduction_rate": rate(violation_hits),
        "candidate_reproduction_rate": rate(candidate_hits),
        "inconclusive_rate": rate(inconclusive_hits),
        "confirmed_hits": confirmed,
        "confirmed_reproduction_rate": rate(confirmed) if confirmed is not None else None,
        # Backward-compatible name; deliberately the violation rate, not the
        # candidate rate, so INCONCLUSIVE is never reported as a confirmed defect.
        "reproduction_rate": rate(violation_hits),
    }


def reproduce_plan(
    plan: ConcurrentTestPlan,
    generated_harness: Path,
    cfg: EngineConfig,
    *,
    repeats: int = 5,
    confirmed: Optional[int] = None,
) -> Dict:
    registry = ContractRegistry(cfg.contract_dir)
    statuses = []
    results = []
    for _ in range(max(1, repeats)):
        observations = run_plan(Path(generated_harness), plan, cfg)
        result = evaluate_oracles(plan, observations, registry)
        results.append(result)
        statuses.append([c.status for c in result.oracle_results])
    out = summarize_statuses(statuses, confirmed=confirmed)
    out.update({
        "plan_id": plan.plan_id,
        "statuses": statuses,
        "last_result": results[-1].to_dict() if results else None,
    })
    return out
