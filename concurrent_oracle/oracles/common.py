"""Shared digest comparison helpers."""
from __future__ import annotations

import math
from typing import Any, Optional

REL_TOL = 1e-6
ABS_TOL = 1e-8


def _num_equal(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is b
    try:
        af, bf = float(a), float(b)
    except Exception:
        return a == b
    if math.isnan(af) and math.isnan(bf):
        return True
    return math.isclose(af, bf, rel_tol=REL_TOL, abs_tol=ABS_TOL)


def digests_equal(a: Any, b: Any) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        # Generic snapshot dicts (e.g. {"args": [...], "kwargs": {...}}).
        if "kind" not in a and "kind" not in b:
            return set(a) == set(b) and all(digests_equal(a[k], b[k]) for k in a)
        if a.get("kind") != b.get("kind"):
            return False
        if a.get("kind") == "tensor":
            if a.get("dtype") != b.get("dtype"):
                return False
            if a.get("shape") != b.get("shape"):
                return False
            if a.get("numel") != b.get("numel"):
                return False
            return _num_equal(a.get("sum"), b.get("sum"))
        if "items" in a or "items" in b:
            ia, ib = a.get("items"), b.get("items")
            if isinstance(ia, dict) and isinstance(ib, dict):
                return set(ia) == set(ib) and all(digests_equal(ia[k], ib[k]) for k in ia)
            if isinstance(ia, list) and isinstance(ib, list):
                return len(ia) == len(ib) and all(digests_equal(x, y) for x, y in zip(ia, ib))
            return False
        return a.get("repr") == b.get("repr")
    return a == b


def first_mismatch(a: Any, b: Any, path: str = "") -> Optional[str]:
    if not digests_equal(a, b):
        return path or "root"
    if isinstance(a, dict) and isinstance(b, dict):
        ia, ib = a.get("items"), b.get("items")
        if isinstance(ia, list) and isinstance(ib, list):
            for i, (x, y) in enumerate(zip(ia, ib)):
                m = first_mismatch(x, y, f"{path}[{i}]")
                if m:
                    return m
        if isinstance(ia, dict) and isinstance(ib, dict):
            for k in ia:
                m = first_mismatch(ia[k], ib.get(k), f"{path}.{k}")
                if m:
                    return m
    return None


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
