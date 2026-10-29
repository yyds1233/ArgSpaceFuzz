"""Oracle dispatch and result aggregation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .oracles import get_oracle
from .schemas import (
    COMPLETED, CRASH, ERROR, INCONCLUSIVE, NOT_APPLICABLE, PASS,
    TIMEOUT, VIOLATION, OracleCheckResult, OracleResult, Observation,
)


def _execution_status(observations: List[Observation]) -> str:
    statuses = [o.execution_status for o in observations]
    for s in (TIMEOUT, CRASH, ERROR):
        if s in statuses:
            return s
    return COMPLETED


def _overall(checks: List[OracleCheckResult]) -> str:
    statuses = [c.status for c in checks]
    if VIOLATION in statuses:
        return VIOLATION
    if INCONCLUSIVE in statuses:
        return INCONCLUSIVE
    if PASS in statuses:
        return PASS
    return NOT_APPLICABLE


def evaluate_oracles(plan, observations: List[Observation], registry) -> OracleResult:
    if not observations:
        return OracleResult(
            plan_id=plan.plan_id, scenario=plan.scenario,
            execution_status=ERROR,
            oracle_results=[OracleCheckResult(o, INCONCLUSIVE, "no observations") for o in plan.oracles],
            summary="no observations",
        )
    baseline = observations[0]
    concurrent = observations[1:]
    checks: List[OracleCheckResult] = []
    for oracle_id in plan.oracles:
        mod = get_oracle(oracle_id)
        if mod is None:
            checks.append(OracleCheckResult(oracle_id, NOT_APPLICABLE, "oracle not registered"))
            continue
        try:
            checks.append(mod.evaluate(plan, baseline, concurrent, registry))
        except Exception as exc:
            checks.append(OracleCheckResult(
                oracle_id, INCONCLUSIVE,
                f"oracle raised {type(exc).__name__}: {exc}",
            ))
    overall = _overall(checks)
    execution_status = _execution_status(observations)

    obs_summary = {
        "baseline": {
            "status": baseline.execution_status,
            "calls": baseline.calls,
            "exceptions": len(baseline.exceptions),
            "timeouts": baseline.timeouts,
            "elapsed_seconds": baseline.elapsed_seconds,
        },
        "concurrent": [
            {
                "threads": o.thread_count,
                "status": o.execution_status,
                "calls": o.calls,
                "exceptions": len(o.exceptions),
                "timeouts": o.timeouts,
                "elapsed_seconds": o.elapsed_seconds,
            }
            for o in concurrent
        ],
    }
    summary = "; ".join(f"{c.oracle_id}={c.status}: {c.reason}" for c in checks)
    return OracleResult(
        plan_id=plan.plan_id,
        scenario=plan.scenario,
        execution_status=execution_status,
        observations=obs_summary,
        oracle_results=checks,
        reproduction_rate=None,
        failure_artifacts=[],
        summary=f"overall={overall}; " + summary,
    )


def save_oracle_result(result: OracleResult, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path
