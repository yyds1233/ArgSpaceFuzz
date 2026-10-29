"""O5: exception isolation and recovery."""
from __future__ import annotations

from ..schemas import (
    INCONCLUSIVE, NOT_APPLICABLE, PASS, SUPPORTED, VIOLATION,
    OracleCheckResult,
)

EXPECTED_TYPES = {"RuntimeError", "ValueError", "IndexError", "TypeError", "AssertionError", "NotImplementedError"}


def _role_exceptions(obs):
    by_role = {}
    for exc in obs.exceptions or []:
        role = str(exc.get("role") or "unknown")
        by_role.setdefault(role, []).append(exc)
    return by_role


def evaluate(plan, baseline, concurrent, registry) -> OracleCheckResult:
    if not concurrent:
        return OracleCheckResult("O5", NOT_APPLICABLE, "No concurrent observation.")
    any_exc = [e for obs in concurrent for e in (obs.exceptions or [])]
    if not any_exc:
        return OracleCheckResult("O5", NOT_APPLICABLE, "No exceptions observed in the mixed-case scenario.")

    findings = []
    expected_reject_seen = False
    for obs in concurrent:
        by_role = _role_exceptions(obs)
        invalid_excs = by_role.get("invalid", [])
        valid_excs = by_role.get("valid", [])
        if invalid_excs:
            types = {str(e.get("type")) for e in invalid_excs}
            if types & EXPECTED_TYPES:
                expected_reject_seen = True
            else:
                findings.append({"threads": obs.thread_count, "reason": "invalid role raised unexpected types",
                                 "types": sorted(types)})
        else:
            findings.append({"threads": obs.thread_count, "reason": "invalid role did not raise"})
        if valid_excs:
            findings.append({"threads": obs.thread_count, "reason": "valid role was polluted",
                             "types": sorted({str(e.get("type")) for e in valid_excs})})
        recovery = obs.recovery or {}
        if not recovery.get("attempted"):
            findings.append({"threads": obs.thread_count, "reason": "recovery was not attempted"})
        elif not recovery.get("success"):
            findings.append({"threads": obs.thread_count, "reason": "recovery call failed",
                             "type": recovery.get("type")})

    if findings:
        status = VIOLATION if plan.contract_status == SUPPORTED else INCONCLUSIVE
        return OracleCheckResult(
            "O5", status,
            "Mixed-case exception isolation/recovery check found a problem." if status == VIOLATION
            else "Mixed-case evidence is insufficient or contract is UNKNOWN.",
            {"findings": findings[:6], "expected_reject_seen": expected_reject_seen},
        )
    if expected_reject_seen:
        return OracleCheckResult(
            "O5", PASS,
            "Invalid task raised while valid tasks and the recovery call succeeded.",
        )
    return OracleCheckResult(
        "O5", INCONCLUSIVE,
        "No expected rejection was observed for the invalid role.",
    )
