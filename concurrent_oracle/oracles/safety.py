"""O1: invalid-input safety rejection."""
from __future__ import annotations

from ..schemas import (
    INCONCLUSIVE, NOT_APPLICABLE, PASS, SUPPORTED, UNKNOWN, VIOLATION,
    OracleCheckResult,
)
from .common import has_crash_signal, selected_exception_types

EXPECTED_TYPES = {"RuntimeError", "ValueError", "IndexError", "TypeError", "AssertionError", "NotImplementedError"}


def evaluate(plan, baseline, concurrent, registry) -> OracleCheckResult:
    status = plan.contract_status
    observations = [baseline] + list(concurrent)
    any_exc = [e for obs in observations for e in (obs.exceptions or [])]
    crash = any(has_crash_signal(obs) for obs in observations)
    types = selected_exception_types(any_exc)

    if not any_exc and not crash:
        return OracleCheckResult(
            "O1", NOT_APPLICABLE,
            "No invalid-input exception/crash observed in this case.",
        )
    if crash:
        if status == SUPPORTED:
            return OracleCheckResult(
                "O1", VIOLATION,
                "Native crash/abort observed on an invalid-input safety scenario.",
                {"crash": True, "types": sorted(types)},
            )
        return OracleCheckResult(
            "O1", INCONCLUSIVE,
            "Crash/abort observed but the invalid-input contract is not confirmed.",
            {"crash": True, "types": sorted(types)},
        )
    if types & EXPECTED_TYPES:
        return OracleCheckResult(
            "O1", PASS,
            f"Invalid input was rejected with expected exception types: {sorted(types & EXPECTED_TYPES)}",
            {"types": sorted(types)},
        )
    return OracleCheckResult(
        "O1", INCONCLUSIVE,
        f"Unexpected exception types for safety scenario: {sorted(types)}",
        {"types": sorted(types)},
    )
