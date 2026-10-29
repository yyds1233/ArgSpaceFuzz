"""Normalize generated-harness observations."""
from __future__ import annotations

import json
from pathlib import Path

from .schemas import COMPLETED, CRASH, ERROR, TIMEOUT, Observation


def load_observation(path: Path, plan_id: str, returncode: int | None, stderr: str = "",
                     config_id: str = "", role: str = "") -> Observation:
    path = Path(path)
    if not path.exists():
        status = TIMEOUT if returncode in (124, 137) else CRASH
        return Observation(
            plan_id=plan_id, execution_status=status, config_id=config_id, role=role,
            completeness=status.lower(),
            returncode=returncode, stderr=stderr[-8000:], raw={},
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return Observation(
            plan_id=plan_id, execution_status=ERROR, config_id=config_id, role=role,
            returncode=returncode, stderr=(stderr + f"\nobservation parse error: {exc}")[-8000:],
            raw={},
        )
    status = data.get("status", COMPLETED)
    if status not in ("COMPLETED", "TIMEOUT", "CRASH", "ERROR"):
        status = COMPLETED
    calls = int(data.get("calls", 0))
    timeouts = int(data.get("timeouts", 0))
    if status == COMPLETED and calls > 0 and timeouts == 0:
        completeness = "complete"
    elif status == COMPLETED:
        completeness = "partial"
    else:
        completeness = status.lower()
    return Observation(
        plan_id=plan_id,
        execution_status=status,
        config_id=config_id,
        role=role,
        completeness=completeness,
        calls=calls,
        successful_calls=int(data.get("successful_calls", 0)),
        exceptions=list(data.get("exceptions", [])),
        timeouts=timeouts,
        elapsed_seconds=float(data.get("elapsed_seconds", 0.0)),
        thread_count=int(data.get("threads", 1)),
        internal_threads=int(data.get("internal_threads", 1)),
        trials=int(data.get("trials", 0)),
        result_digests=data.get("result_digests", {}),
        shared_digest=data.get("shared_digest"),
        shared_path=data.get("shared_path"),
        input_digests_before=data.get("input_digests_before", {}),
        input_digests_after=data.get("input_digests_after", {}),
        thread_roles=data.get("thread_roles", []),
        case_map=data.get("case_map", []),
        context_probe=data.get("context_probe", {}),
        main_grad_enabled=data.get("main_grad_enabled"),
        main_rng_restored=data.get("main_rng_restored"),
        recovery=data.get("recovery", {}),
        raw=data,
        stderr=stderr[-8000:],
        returncode=returncode,
    )
