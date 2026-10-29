"""O1: invalid-input safety rejection (per-observation judgment).

The Oracle never aggregates exceptions across configurations before deciding.
Each baseline/concurrent observation must independently match the declared
expected behavior.  Broad exception types alone are not proof of correct
rejection.
"""
from __future__ import annotations

from ..schemas import (
    CRASH, ERROR, INCONCLUSIVE, NOT_APPLICABLE, PASS, SUPPORTED, TIMEOUT,
    VIOLATION, OracleCheckResult,
)


def evaluate(plan, baseline, concurrent, registry) -> OracleCheckResult:
    expected = {str(x) for x in (plan.expected_exception_types or [])}
    if plan.expected_behavior != "raise" or not expected:
        return OracleCheckResult(
            "O1", NOT_APPLICABLE,
            "No invalid-input contract/case was defined for this plan.",
        )

    observations = [("baseline", baseline)] + [
        (f"concurrent[{o.thread_count}t]", o) for o in concurrent
    ]
    findings = []
    for label, obs in observations:
        if obs.execution_status in (TIMEOUT, CRASH, ERROR):
            findings.append({"run": label, "kind": obs.execution_status.lower(),
                             "status": obs.execution_status})
            continue
        if obs.calls == 0:
            findings.append({"run": label, "kind": "missing_observation"})
            continue
        exc_types = {str(e.get("type")) for e in (obs.exceptions or [])}
        if obs.successful_calls > 0:
            findings.append({"run": label, "kind": "accepted_invalid_input",
                             "successful_calls": obs.successful_calls})
            continue
        if exc_types & expected:
            extra = exc_types - expected
            if extra:
                findings.append({"run": label, "kind": "unexpected_exception",
                                 "types": sorted(extra)})
            # otherwise this observation correctly rejected the input
            continue
        if exc_types:
            findings.append({"run": label, "kind": "unexpected_exception",
                             "types": sorted(exc_types)})
        else:
            findings.append({"run": label, "kind": "no_exception"})

    if not findings:
        return OracleCheckResult(
            "O1", PASS,
            f"Every observation rejected the invalid input with expected types {sorted(expected)}.",
            {"expected_types": sorted(expected), "runs": len(observations)},
        )

    hard = {"crash", "timeout", "error", "accepted_invalid_input"}
    status = VIOLATION if plan.contract_status == SUPPORTED and any(
        f["kind"] in hard for f in findings
    ) else INCONCLUSIVE
    return OracleCheckResult(
        "O1", status,
        "At least one observation did not match the declared invalid-input behavior.",
        {"findings": findings[:10], "expected_types": sorted(expected)},
    )
