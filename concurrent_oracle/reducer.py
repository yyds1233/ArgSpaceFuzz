"""Failure reduction: configuration search plus variant generation.

The reducer preserves the original plan, case and logs.  It only proposes and
re-runs smaller configurations; it never overwrites the original evidence.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .config import EngineConfig
from .contract_registry import ContractRegistry
from .oracle_engine import evaluate_oracles
from .runner import run_plan
from .schemas import ConcurrentTestPlan, new_id


def make_variants(plan: ConcurrentTestPlan) -> List[ConcurrentTestPlan]:
    """Smaller configuration variants, ordered from smallest to largest."""
    threads = [int(x) for x in plan.execution.get("external_threads", [2])]
    trials = int(plan.execution.get("trials", 1))
    internal = [int(x) for x in plan.execution.get("internal_threads", [1])]
    max_threads = max(threads) if threads else 2
    max_internal = max(internal) if internal else 1

    candidates = []
    for n in sorted({2, max(2, max_threads // 2)}):
        if n > max_threads:
            continue
        for tr in sorted({1, max(1, trials // 2)}):
            for t in sorted({1, max_internal}):
                candidates.append((n, tr, t))
    # Deduplicate and keep the original if it is already the smallest.
    seen = set()
    ordered = []
    for n, tr, t in sorted(candidates, key=lambda x: (x[0] * x[1] * x[2], x[0], x[1], x[2])):
        key = (n, tr, t)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)

    variants: List[ConcurrentTestPlan] = []
    for n, tr, t in ordered:
        p = ConcurrentTestPlan.from_dict(plan.to_dict())
        p.plan_id = new_id("reduce")
        p.execution = dict(plan.execution)
        p.execution["external_threads"] = [n]
        p.execution["trials"] = tr
        p.execution["internal_threads"] = [t]
        p.notes = list(plan.notes) + [f"reduced from {plan.plan_id}: threads={n}, trials={tr}, internal={t}"]
        variants.append(p)
    return variants


def reduce_configuration(
    plan: ConcurrentTestPlan,
    generated_harness: Path,
    cfg: EngineConfig,
    *,
    oracle_ids: Optional[Sequence[str]] = None,
    trigger_statuses: Sequence[str] = ("VIOLATION",),
    max_variants: int = 8,
) -> Dict:
    """Try smaller configurations and keep the smallest one that still triggers.

    A variant is accepted when at least one selected Oracle returns a status in
    ``trigger_statuses``.  Original logs and the original plan are untouched.
    """
    registry = ContractRegistry(cfg.contract_dir)
    wanted = {x.upper() for x in (oracle_ids or plan.oracles)}
    history = []
    best = None
    for variant in make_variants(plan)[:max_variants]:
        observations = run_plan(Path(generated_harness), variant, cfg)
        result = evaluate_oracles(variant, observations, registry)
        checks = [
            {"oracle": c.oracle_id, "status": c.status, "reason": c.reason}
            for c in result.oracle_results
            if c.oracle_id in wanted
        ]
        triggered = any(c["status"] in set(trigger_statuses) for c in checks)
        history.append({
            "plan_id": variant.plan_id,
            "threads": variant.execution.get("external_threads"),
            "trials": variant.execution.get("trials"),
            "internal_threads": variant.execution.get("internal_threads"),
            "execution_status": result.execution_status,
            "checks": checks,
            "triggered": triggered,
        })
        if triggered and best is None:
            best = variant
            # Keep searching only for an even smaller candidate; variants are
            # ordered small -> large, so the first trigger is the smallest.
            break

    return {
        "original_plan_id": plan.plan_id,
        "oracle_ids": sorted(wanted),
        "trigger_statuses": list(trigger_statuses),
        "best": best.to_dict() if best else None,
        "history": history,
        "note": "original plan, case, and logs are preserved",
    }
