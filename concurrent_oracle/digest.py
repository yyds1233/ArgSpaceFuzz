"""Content-aware tensor digests and comparison for Oracle evidence.

The digest is intentionally more than a checksum: it stores a strong content
hash plus enough values/statistics for both exact and tolerance comparisons.
"""
from __future__ import annotations

import hashlib
import io
import math
from typing import Any, Optional

import torch

MAX_VALUES = 4096


def _tensor_bytes(t: torch.Tensor) -> bytes:
    t = t.detach().cpu()
    if t.is_quantized:
        data = t.int_repr().detach().cpu().contiguous()
        return data.numpy().tobytes() if data.numel() else b""
    if t.layout != torch.strided:
        buf = io.BytesIO()
        torch.save(t, buf)
        return buf.getvalue()
    tc = t.contiguous()
    if tc.numel() == 0:
        return b""
    try:
        return tc.numpy().tobytes()
    except Exception:
        try:
            return tc.reshape(-1).view(torch.uint8).numpy().tobytes()
        except Exception:
            buf = io.BytesIO()
            torch.save(tc, buf)
            return buf.getvalue()


def _flatten_values(t: torch.Tensor):
    if t.layout != torch.strided:
        return None
    tc = t.detach().cpu().reshape(-1)
    if tc.numel() > MAX_VALUES:
        half = MAX_VALUES // 2
        tc = torch.cat([tc[:half], tc[-half:]])
    if tc.is_complex():
        return [[float(v.real), float(v.imag)] for v in tc]
    if tc.dtype == torch.bool:
        return [bool(v) for v in tc]
    if tc.is_floating_point():
        return [float(v) for v in tc]
    try:
        return [int(v) for v in tc]
    except Exception:
        return [str(v) for v in tc]


def tensor_digest(value: Any) -> Any:
    if not isinstance(value, torch.Tensor):
        if isinstance(value, tuple):
            return {"kind": "tuple", "items": [tensor_digest(v) for v in value]}
        if isinstance(value, list):
            return {"kind": "list", "items": [tensor_digest(v) for v in value]}
        if isinstance(value, dict):
            return {"kind": "dict", "items": {str(k): tensor_digest(v) for k, v in value.items()}}
        return {"kind": type(value).__name__, "repr": repr(value)[:300]}

    t = value.detach().cpu()
    numel = int(t.numel())
    layout = str(t.layout)
    is_complex = bool(t.is_complex())
    values = _flatten_values(t)

    sum_value = None
    sum_abs = None
    min_value = None
    max_value = None
    nan_count = 0
    posinf_count = 0
    neginf_count = 0
    if numel > 0:
        try:
            if is_complex:
                sum_abs = float(t.abs().sum().item())
            else:
                sum_abs = float(t.abs().to(torch.float64).sum().item())
                sum_value = float(t.to(torch.float64).sum().item())
                min_value = float(t.to(torch.float64).min().item())
                max_value = float(t.to(torch.float64).max().item())
        except Exception:
            pass
        if t.is_floating_point() or is_complex:
            try:
                nan_count = int(torch.isnan(t).sum().item())
                posinf_count = int(torch.isposinf(t).sum().item()) if hasattr(torch, "isposinf") else 0
                neginf_count = int(torch.isneginf(t).sum().item()) if hasattr(torch, "isneginf") else 0
            except Exception:
                pass

    digest = {
        "kind": "tensor",
        "dtype": str(t.dtype),
        "shape": list(t.shape),
        "numel": numel,
        "layout": layout,
        "device": str(t.device),
        "is_complex": is_complex,
        "is_quantized": bool(t.is_quantized),
        "content_hash": hashlib.sha256(_tensor_bytes(value)).hexdigest(),
        "sum": sum_value,
        "sum_abs": sum_abs,
        "min": min_value,
        "max": max_value,
        "nan_count": nan_count,
        "posinf_count": posinf_count,
        "neginf_count": neginf_count,
        "values": values,
        "values_sampled": numel > MAX_VALUES,
    }
    if t.is_quantized:
        try:
            digest["q_scale"] = float(t.q_scale())
            digest["q_zero_point"] = int(t.q_zero_point())
        except Exception:
            pass
    if t.layout != torch.strided:
        try:
            digest["nnz"] = int(t._nnz())
        except Exception:
            pass
    return digest


def _num_close(a: Any, b: Any, rel_tol: float, abs_tol: float) -> bool:
    if a is None or b is None:
        return a is b
    try:
        af, bf = float(a), float(b)
    except Exception:
        return a == b
    if math.isnan(af) or math.isnan(bf):
        return math.isnan(af) and math.isnan(bf)
    if math.isinf(af) or math.isinf(bf):
        return af == bf
    return math.isclose(af, bf, rel_tol=rel_tol, abs_tol=abs_tol)


def _values_equal(a: Any, b: Any, rel_tol: float, abs_tol: float) -> bool:
    if isinstance(a, list) and isinstance(b, list) and a and isinstance(a[0], list):
        return len(a) == len(b) and all(
            isinstance(x, list) and isinstance(y, list) and len(x) == 2 and len(y) == 2
            and _num_close(x[0], y[0], rel_tol, abs_tol)
            and _num_close(x[1], y[1], rel_tol, abs_tol)
            for x, y in zip(a, b)
        )
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        for x, y in zip(a, b):
            if isinstance(x, (int, float, bool)) and isinstance(y, (int, float, bool)):
                if isinstance(x, bool) or isinstance(y, bool):
                    if x is not y:
                        return False
                elif not _num_close(x, y, rel_tol, abs_tol):
                    return False
            elif x != y:
                return False
        return True
    return a == b


def digests_equal(a: Any, b: Any, rel_tol: float = 0.0, abs_tol: float = 0.0) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        if "kind" not in a and "kind" not in b:
            return set(a) == set(b) and all(
                digests_equal(a[k], b[k], rel_tol, abs_tol) for k in a
            )
        if a.get("kind") != b.get("kind"):
            return False
        kind = a.get("kind")
        if kind == "tensor":
            for key in ("dtype", "shape", "numel", "layout"):
                if a.get(key) != b.get(key):
                    return False
            if rel_tol == 0.0 and abs_tol == 0.0:
                if a.get("content_hash") and b.get("content_hash"):
                    return a["content_hash"] == b["content_hash"]
                if a.get("values") is not None and b.get("values") is not None:
                    return _values_equal(a["values"], b["values"], 0.0, 0.0)
                for key in ("sum", "sum_abs", "min", "max", "nan_count"):
                    if a.get(key) != b.get(key):
                        return False
                return True
            if a.get("values") is not None and b.get("values") is not None:
                return _values_equal(a["values"], b["values"], rel_tol, abs_tol)
            for key in ("sum", "sum_abs", "min", "max"):
                if not _num_close(a.get(key), b.get(key), rel_tol, abs_tol):
                    return False
            return a.get("nan_count") == b.get("nan_count")
        if "items" in a or "items" in b:
            ia, ib = a.get("items"), b.get("items")
            if isinstance(ia, dict) and isinstance(ib, dict):
                return set(ia) == set(ib) and all(
                    digests_equal(ia[k], ib[k], rel_tol, abs_tol) for k in ia
                )
            if isinstance(ia, list) and isinstance(ib, list):
                return len(ia) == len(ib) and all(
                    digests_equal(x, y, rel_tol, abs_tol) for x, y in zip(ia, ib)
                )
            return False
        return a.get("repr") == b.get("repr")
    return a == b


def first_mismatch(a: Any, b: Any, rel_tol: float = 0.0, abs_tol: float = 0.0,
                   path: str = "") -> Optional[str]:
    if not digests_equal(a, b, rel_tol, abs_tol):
        return path or "root"
    if isinstance(a, dict) and isinstance(b, dict):
        ia, ib = a.get("items"), b.get("items")
        if isinstance(ia, list) and isinstance(ib, list):
            for i, (x, y) in enumerate(zip(ia, ib)):
                m = first_mismatch(x, y, rel_tol, abs_tol, f"{path}[{i}]")
                if m:
                    return m
        if isinstance(ia, dict) and isinstance(ib, dict):
            for k in ia:
                m = first_mismatch(ia[k], ib.get(k), rel_tol, abs_tol, f"{path}.{k}")
                if m:
                    return m
    return None
