"""O3: independent-call isolation."""
from __future__ import annotations

from ..schemas import (
    INCONCLUSIVE, NOT_APPLICABLE, PASS, SUPPORTED, VIOLATION,
    OracleCheckResult,
)
from .common import digests_equal, first_mismatch


def _baseline_digests(baseline):
    digs = baseline.result_digests or {}
    for key in ("0", 0):
        if key in digs and digs[key]:
            return digs[key]
    for v in digs.values():
        if v:
            return v
    return []


def evaluate(plan, baseline, concurrent, registry) -> OracleCheckResult:
    base = _baseline_digests(baseline)
    if not base:
        return OracleCheckResult("O3", NOT_APPLICABLE, "No baseline result digest available.")
    mismatches = []
    for obs in concurrent:
        if obs.exceptions:
            mismatches.append({"threads": obs.thread_count, "reason": "exceptions", "count": len(obs.exceptions)})
            continue
        for tid, seq in (obs.result_digests or {}).items():
            if not seq:
                mismatches.append({"threads": obs.thread_count, "thread": tid, "reason": "no result"})
                continue
            for idx, d in enumerate(seq):
                ref = base[min(idx, len(base) - 1)]
                if not digests_equal(d, ref):
                    mismatches.append({
                        "threads": obs.thread_count,
                        "thread": tid,
                        "trial": idx,
                        "mismatch_at": first_mismatch(d, ref),
                    })
                    break

    if not mismatches:
        if plan.contract_status == SUPPORTED:
            return OracleCheckResult("O3", PASS, "Independent concurrent calls match the serial baseline.")
        return OracleCheckResult(
            "O3", INCONCLUSIVE,
            "No difference observed, but the API contract is UNKNOWN.",
        )
    if plan.contract_status == SUPPORTED:
        return OracleCheckResult(
            "O3", VIOLATION,
            "Independent concurrent calls differ from the serial baseline under a SUPPORTED contract.",
            {"mismatches": mismatches[:6]},
        )
    return OracleCheckResult(
        "O3", INCONCLUSIVE,
        "Independent concurrent calls differ, but the API contract is UNKNOWN or the op is nondeterministic.",
        {"mismatches": mismatches[:6]},
    )
