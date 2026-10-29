"""O4: read-only state preservation."""
from __future__ import annotations

from ..schemas import (
    INCONCLUSIVE, NOT_APPLICABLE, PASS, SUPPORTED, VIOLATION,
    OracleCheckResult,
)
from .common import digests_equal, incomplete_reason, is_complete


def evaluate(plan, baseline, concurrent, registry) -> OracleCheckResult:
    incomplete = [o for o in concurrent if not is_complete(o)]
    if incomplete:
        return OracleCheckResult(
            "O4", INCONCLUSIVE,
            "Incomplete observation(s) cannot be used for state-preservation checks: "
            + incomplete_reason(incomplete),
        )
    checks = []
    for obs in concurrent:
        before = obs.input_digests_before or {}
        after = obs.input_digests_after or {}
        if not before or not after:
            continue
        changed = not digests_equal(before, after)
        checks.append({"threads": obs.thread_count, "changed": changed})
    if not checks:
        return OracleCheckResult("O4", NOT_APPLICABLE, "No input snapshots available for O4.")

    if any(c["changed"] for c in checks):
        if plan.contract_status == SUPPORTED:
            return OracleCheckResult(
                "O4", VIOLATION,
                "A read-only input/state changed during concurrent execution under a SUPPORTED contract.",
                {"checks": checks},
            )
        return OracleCheckResult(
            "O4", INCONCLUSIVE,
            "Input/state changed, but the read-only contract is UNKNOWN or the API mutates by design.",
            {"checks": checks},
        )
    if plan.contract_status == SUPPORTED:
        return OracleCheckResult("O4", PASS, "Input/state snapshots are unchanged after concurrent execution.")
    return OracleCheckResult(
        "O4", INCONCLUSIVE,
        "Input/state snapshots are unchanged, but the read-only contract is UNKNOWN.",
    )
