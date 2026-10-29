"""O6: thread-context isolation (grad mode and fork_rng)."""
from __future__ import annotations

from ..schemas import (
    INCONCLUSIVE, NOT_APPLICABLE, PASS, SUPPORTED, VIOLATION,
    OracleCheckResult,
)


def _expected_for_role(role: str):
    if role == "grad_true":
        return True
    if role == "grad_false":
        return False
    return None


def evaluate(plan, baseline, concurrent, registry) -> OracleCheckResult:
    if not concurrent:
        return OracleCheckResult("O6", NOT_APPLICABLE, "No concurrent observation.")
    mismatches = []
    any_probe = False
    checked_grad = 0
    checked_rng = 0
    for obs in concurrent:
        probe = obs.context_probe or {}
        roles = obs.thread_roles or []
        if probe:
            any_probe = True
        for tid, states in probe.items():
            if not states:
                continue
            last = states[-1]
            if "rng_restored" in last:
                checked_rng += 1
                if not last.get("rng_restored"):
                    mismatches.append({
                        "threads": obs.thread_count,
                        "thread": tid,
                        "reason": "fork_rng did not restore the RNG state",
                    })
                continue
            try:
                idx = int(tid)
            except Exception:
                idx = 0
            role = roles[idx] if idx < len(roles) else ""
            expected = _expected_for_role(role)
            if expected is None:
                continue
            checked_grad += 1
            actual = last.get("grad_enabled")
            if actual is not expected:
                mismatches.append({
                    "threads": obs.thread_count,
                    "thread": tid,
                    "role": role,
                    "expected": expected,
                    "actual": actual,
                })
        if obs.main_grad_enabled is False:
            mismatches.append({
                "threads": obs.thread_count,
                "reason": "main thread grad mode leaked to False",
            })
        if obs.main_rng_restored is False:
            mismatches.append({
                "threads": obs.thread_count,
                "reason": "main thread RNG state changed after fork_rng",
            })

    if not any_probe or (checked_grad == 0 and checked_rng == 0):
        return OracleCheckResult(
            "O6", NOT_APPLICABLE,
            "No per-thread grad-mode or RNG-restore probe was recorded.",
        )
    if mismatches:
        status = VIOLATION if plan.contract_status == SUPPORTED else INCONCLUSIVE
        return OracleCheckResult(
            "O6", status,
            "Thread-local context state did not match its role or leaked to the main thread."
            if status == VIOLATION else
            "Context probe differs, but the contract is UNKNOWN.",
            {"mismatches": mismatches[:8]},
        )
    if plan.contract_status == SUPPORTED:
        return OracleCheckResult(
            "O6", PASS,
            f"Context probes passed (grad_checked={checked_grad}, rng_checked={checked_rng}).",
        )
    return OracleCheckResult(
        "O6", INCONCLUSIVE,
        "Context probes are consistent, but the thread-isolation contract is UNKNOWN.",
        {"grad_checked": checked_grad, "rng_checked": checked_rng},
    )
