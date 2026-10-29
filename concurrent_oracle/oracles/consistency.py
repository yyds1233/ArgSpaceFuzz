"""O2: serial/parallel semantic consistency for shared state."""
from __future__ import annotations

from ..schemas import (
    INCONCLUSIVE, NOT_APPLICABLE, PASS, SUPPORTED, VIOLATION,
    OracleCheckResult,
)
from .common import digests_equal, first_mismatch


def evaluate(plan, baseline, concurrent, registry) -> OracleCheckResult:
    if baseline.shared_digest is None:
        return OracleCheckResult(
            "O2", NOT_APPLICABLE,
            "No shared output/state digest captured for this case.",
        )
    mismatches = []
    for obs in concurrent:
        if obs.shared_digest is None:
            mismatches.append({"threads": obs.thread_count, "reason": "missing shared digest"})
            continue
        if not digests_equal(obs.shared_digest, baseline.shared_digest):
            mismatches.append({
                "threads": obs.thread_count,
                "mismatch_at": first_mismatch(obs.shared_digest, baseline.shared_digest),
                "concurrent": obs.shared_digest,
                "baseline": baseline.shared_digest,
            })

    if plan.contract_status != SUPPORTED:
        if mismatches:
            return OracleCheckResult(
                "O2", INCONCLUSIVE,
                "Serial/concurrent shared state differs, but shared-write semantics are not confirmed (contract UNKNOWN).",
                {"mismatches": mismatches[:4]},
            )
        return OracleCheckResult(
            "O2", INCONCLUSIVE,
            "Shared-state contract is UNKNOWN; no violation can be asserted even when values match.",
        )

    if mismatches:
        return OracleCheckResult(
            "O2", VIOLATION,
            "Concurrent shared-state result differs from the serial baseline under a SUPPORTED contract.",
            {"mismatches": mismatches[:4]},
        )
    return OracleCheckResult("O2", PASS, "Concurrent shared-state result matches the serial baseline.")
