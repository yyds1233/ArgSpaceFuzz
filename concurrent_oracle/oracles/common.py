"""Shared Oracle helpers.

Tensor comparison is delegated to :mod:`concurrent_oracle.digest`, which uses
content hashes plus values/statistics (never element sum alone).
"""
from __future__ import annotations

from ..digest import digests_equal, first_mismatch

REL_TOL = 1e-6
ABS_TOL = 1e-8


def is_complete(obs) -> bool:
    return (
        getattr(obs, "execution_status", "") == "COMPLETED"
        and getattr(obs, "completeness", "complete") == "complete"
    )


def incomplete_reason(observations) -> str:
    parts = []
    for obs in observations:
        if not is_complete(obs):
            parts.append(
                f"{getattr(obs, 'config_id', '') or 'run'}:"
                f"{getattr(obs, 'role', '') or '?'}:"
                f"{getattr(obs, 'execution_status', '')}/"
                f"{getattr(obs, 'completeness', '')}"
            )
    return ", ".join(parts)


def selected_exception_types(exceptions: list) -> set:
    return {str(x.get("type")) for x in exceptions if isinstance(x, dict)}


def has_crash_signal(obs) -> bool:
    if obs.execution_status in ("CRASH", "ERROR"):
        return True
    for exc in obs.exceptions or []:
        t = str(exc.get("type", "")).lower()
        if "segmentation" in t or "sigsegv" in t or "abort" in t or "fatal" in t:
            return True
    return False


__all__ = [
    "digests_equal", "first_mismatch", "selected_exception_types",
    "has_crash_signal", "is_complete", "incomplete_reason",
]
