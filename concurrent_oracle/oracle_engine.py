"""Oracle dispatch and per-configuration result aggregation."""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

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


def _overall_status(statuses: List[str]) -> str:
    if VIOLATION in statuses:
        return VIOLATION
    if INCONCLUSIVE in statuses:
        return INCONCLUSIVE
    if PASS in statuses:
        return PASS
    return NOT_APPLICABLE


def _groups(observations: List[Observation]):
    """Return OrderedDict config_id -> (baseline, [concurrent])."""
    groups = OrderedDict()
    for idx, obs in enumerate(observations):
        cid = obs.config_id or "default"
        groups.setdefault(cid, {"baseline": None, "concurrent": []})
        role = obs.role
        if role == "baseline" or (not role and idx == 0):
            if groups[cid]["baseline"] is None:
                groups[cid]["baseline"] = obs
            else:
                groups[cid]["concurrent"].append(obs)
        else:
            groups[cid]["concurrent"].append(obs)
    # Legacy observations without config/role: first is baseline, rest concurrent.
    if len(groups) == 1 and "default" in groups:
        group = groups["default"]
        if group["baseline"] is None and observations:
            group["baseline"] = observations[0]
            group["concurrent"] = observations[1:]
    return groups


def evaluate_oracles(plan, observations: List[Observation], registry) -> OracleResult:
    if not observations:
        return OracleResult(
            plan_id=plan.plan_id, scenario=plan.scenario,
            execution_status=ERROR,
            oracle_results=[OracleCheckResult(o, INCONCLUSIVE, "no observations") for o in plan.oracles],
            summary="no observations",
        )

    groups = _groups(observations)
    per_oracle: Dict[str, List[OracleCheckResult]] = {o: [] for o in plan.oracles}
    config_results = []
    for config_id, group in groups.items():
        baseline = group["baseline"]
        concurrent = group["concurrent"]
        checks = []
        for oracle_id in plan.oracles:
            mod = get_oracle(oracle_id)
            if mod is None:
                check = OracleCheckResult(oracle_id, NOT_APPLICABLE, "oracle not registered")
            elif baseline is None:
                check = OracleCheckResult(oracle_id, INCONCLUSIVE, "missing baseline observation")
            else:
                try:
                    check = mod.evaluate(plan, baseline, concurrent, registry)
                except Exception as exc:
                    check = OracleCheckResult(
                        oracle_id, INCONCLUSIVE,
                        f"oracle raised {type(exc).__name__}: {exc}",
                    )
            per_oracle[oracle_id].append(check)
            checks.append(check)
        config_results.append({
            "config_id": config_id,
            "execution_status": _execution_status(
                ([baseline] if baseline else []) + concurrent
            ),
            "checks": [c.to_dict() for c in checks],
        })

    overall_checks = []
    for oracle_id in plan.oracles:
        checks = per_oracle[oracle_id]
        status = _overall_status([c.status for c in checks])
        reason = "; ".join(
            f"{cid}:{c.status}" for cid, c in zip(groups.keys(), checks)
        )
        overall_checks.append(OracleCheckResult(
            oracle_id, status,
            f"per-config: {reason}",
            {"configs": [
                {"config_id": cid, "status": c.status, "reason": c.reason}
                for cid, c in zip(groups.keys(), checks)
            ]},
        ))

    obs_summary = {
        "configs": [
            {
                "config_id": cid,
                "baseline": (
                    {
                        "status": g["baseline"].execution_status,
                        "calls": g["baseline"].calls,
                        "exceptions": len(g["baseline"].exceptions),
                    } if g["baseline"] else None
                ),
                "concurrent": [
                    {
                        "threads": o.thread_count,
                        "internal_threads": o.internal_threads,
                        "status": o.execution_status,
                        "calls": o.calls,
                        "exceptions": len(o.exceptions),
                        "timeouts": o.timeouts,
                        "elapsed_seconds": o.elapsed_seconds,
                    }
                    for o in g["concurrent"]
                ],
            }
            for cid, g in groups.items()
        ]
    }
    summary = "; ".join(f"{c.oracle_id}={c.status}" for c in overall_checks)
    return OracleResult(
        plan_id=plan.plan_id,
        scenario=plan.scenario,
        execution_status=_execution_status(observations),
        observations=obs_summary,
        oracle_results=overall_checks,
        reproduction_rate=None,
        failure_artifacts=[],
        config_results=config_results,
        summary=summary,
    )


def save_oracle_result(result: OracleResult, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path
