"""O7: tracing state and lifecycle."""
from __future__ import annotations

from ..schemas import (
    INCONCLUSIVE, NOT_APPLICABLE, PASS, SUPPORTED, VIOLATION,
    OracleCheckResult,
)


def evaluate(plan, baseline, concurrent, registry) -> OracleCheckResult:
    if not concurrent:
        return OracleCheckResult("O7", NOT_APPLICABLE, "No concurrent observation.")
    calls = sum(obs.successful_calls for obs in concurrent)
    exceptions = [e for obs in concurrent for e in (obs.exceptions or [])]
    recovery_failures = []
    for obs in concurrent:
        rec = obs.recovery or {}
        if rec.get("attempted") and not rec.get("success"):
            recovery_failures.append({"threads": obs.thread_count,
                                      "type": rec.get("type"), "message": rec.get("message")})
    if calls == 0 and not exceptions:
        return OracleCheckResult("O7", NOT_APPLICABLE, "No trace call completed.")

    if plan.contract_status == SUPPORTED:
        if exceptions or recovery_failures:
            return OracleCheckResult(
                "O7", VIOLATION,
                "Tracing raised or its post-run recovery failed under a SUPPORTED contract.",
                {"exception_count": len(exceptions), "recovery_failures": recovery_failures[:4]},
            )
        return OracleCheckResult("O7", PASS, "Tracing and post-run recovery succeeded.")

    # UNKNOWN: record evidence, do not turn a tracing exception into a defect.
    return OracleCheckResult(
        "O7", INCONCLUSIVE,
        "Tracing lifecycle evidence collected, but the concurrency contract is UNKNOWN.",
        {"exception_count": len(exceptions), "recovery_failures": recovery_failures[:4]},
    )
