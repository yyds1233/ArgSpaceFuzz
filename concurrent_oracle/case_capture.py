"""Capture a deterministic case from an Atheris harness.

The original harness is a byte-driven fuzzing harness.  We run its
``TestOneInput`` with a set of fixed byte seeds and intercept the first
matching call to the target API.  The captured arguments (pre-call clones)
are saved as a ``torch.save`` case that the concurrent template can replay.

This module can be used as a library or as a subprocess CLI::

    python -m concurrent_oracle.case_capture \
        --harness llm.torch.addmm.py --target-api torch.addmm \
        --out-dir outputs/cases --require-out
"""
from __future__ import annotations

import argparse
import copy
import importlib
import importlib.util
import json
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .schemas import CaseSpec


class CaptureError(RuntimeError):
    pass


def _clone_value(value: Any) -> Any:
    try:
        import torch
        if isinstance(value, torch.Tensor):
            return value.detach().clone()
    except Exception:
        pass
    if isinstance(value, tuple):
        return tuple(_clone_value(v) for v in value)
    if isinstance(value, list):
        return [_clone_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _clone_value(v) for k, v in value.items()}
    return copy.deepcopy(value)


def _kind(value: Any) -> str:
    try:
        import torch
        if isinstance(value, torch.Tensor):
            return f"tensor({value.dtype},{tuple(value.shape)})"
    except Exception:
        pass
    if isinstance(value, (tuple, list, dict)):
        return type(value).__name__
    return type(value).__name__


def _seed_candidates(seed: int, max_seeds: int) -> List[bytes]:
    fixed = [
        b"\x00" * 128,
        b"\xff" * 128,
        b"\x01" * 128,
        b"\x7f" * 128,
        bytes(range(128)),
    ]
    out = list(fixed)
    for i in range(max(0, max_seeds - len(out))):
        out.append(random.Random(seed + i).randbytes(128))
    return out[:max_seeds]


def _case_score(case: Dict[str, Any]) -> Tuple[int, int, int]:
    """Prefer out= cases, then non-empty tensors, then larger tensors."""
    try:
        import torch
        values = list(case.get("args", ())) + list(case.get("kwargs", {}).values())
        tensors = [v for v in values if isinstance(v, torch.Tensor)]
        nonempty = int(bool(tensors) and all(t.numel() > 0 for t in tensors))
        numel = sum(int(t.numel()) for t in tensors)
        return int(bool(case.get("has_out"))), nonempty, numel
    except Exception:
        return 0, 0, 0


def capture_in_process(
    harness_path: Path,
    target_api: str,
    *,
    require_out: bool = False,
    max_seeds: int = 64,
    seed: int = 0,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Run the original TestOneInput and capture a pre-call case.

    Returns (case_dict, metadata).  case_dict is None when no call was seen.
    """
    harness_path = Path(harness_path).resolve()
    spec = importlib.util.spec_from_file_location("concurrent_oracle_original_harness", harness_path)
    if spec is None or spec.loader is None:
        raise CaptureError(f"cannot load harness module: {harness_path}")
    module = importlib.util.module_from_spec(spec)
    # Some fuzz_api_one harnesses call atheris.Setup/Fuzz at module top level
    # (not guarded by __main__).  Importing such a file must not start the
    # fuzzer; neutralize the two entry points while we load and capture.
    import atheris
    _orig_setup, _orig_fuzz = atheris.Setup, atheris.Fuzz
    atheris.Setup = lambda *a, **k: None
    atheris.Fuzz = lambda *a, **k: None
    try:
        spec.loader.exec_module(module)
    finally:
        atheris.Setup, atheris.Fuzz = _orig_setup, _orig_fuzz

    if not hasattr(module, "TestOneInput"):
        raise CaptureError("harness has no TestOneInput")

    module_name, _, attr = target_api.rpartition(".")
    if not module_name:
        raise CaptureError(f"invalid target_api: {target_api}")
    target_module = None
    for candidate in (module_name, "torch.nn.functional", "torch"):
        try:
            m = importlib.import_module(candidate)
            if hasattr(m, attr):
                target_module = m
                break
        except Exception:
            continue
    if target_module is None:
        raise CaptureError(f"target attribute not found: {target_api} ({module_name}.{attr})")
    original = getattr(target_module, attr)

    captured: List[Dict[str, Any]] = []
    call_index = {"n": 0}

    def wrapper(*args, **kwargs):
        idx = call_index["n"]
        call_index["n"] += 1
        try:
            clone_args = tuple(_clone_value(a) for a in args)
            clone_kwargs = {k: _clone_value(v) for k, v in kwargs.items()}
            captured.append({
                "index": idx,
                "args": clone_args,
                "kwargs": clone_kwargs,
                "has_out": "out" in kwargs,
                "line": None,
            })
        except Exception as exc:
            captured.append({"index": idx, "error": f"clone failed: {exc}", "has_out": "out" in kwargs})
        return original(*args, **kwargs)

    setattr(target_module, attr, wrapper)
    last_exc = None
    best = None
    best_meta = None
    try:
        seeds = _seed_candidates(seed, max_seeds)
        for s in seeds:
            captured.clear()
            call_index["n"] = 0
            try:
                import torch
                torch.manual_seed(seed)
                module.TestOneInput(s)
            except BaseException as exc:
                last_exc = f"{type(exc).__name__}: {exc}"
            candidates = [c for c in captured if "args" in c]
            if not candidates:
                continue
            if require_out:
                candidates = [c for c in candidates if c.get("has_out")]
            if not candidates:
                continue
            for chosen in candidates:
                case = {
                    "target_api": target_api,
                    "args": chosen["args"],
                    "kwargs": chosen["kwargs"],
                    "has_out": bool(chosen.get("has_out")),
                }
                score = _case_score(case)
                if best is None or score > _case_score(best):
                    best = case
                    best_meta = {
                        "seed_hex": s.hex(),
                        "call_index": chosen["index"],
                        "call_count": len(candidates),
                        "has_out": bool(chosen.get("has_out")),
                        "kwargs_keys": sorted(chosen.get("kwargs", {}).keys()),
                        "args_kinds": [_kind(a) for a in chosen.get("args", ())],
                        "captured_exception": last_exc,
                        "seed_count": len(seeds),
                        "score": list(score),
                    }
    finally:
        setattr(target_module, attr, original)

    if best is not None:
        return {"target_api": target_api, "args": best["args"], "kwargs": best["kwargs"]}, best_meta
    return None, {
        "error": "no target call captured",
        "last_exception": last_exc,
        "captured_calls": len(captured),
    }


def capture_case(
    harness_path: Path,
    target_api: str,
    out_dir: Path,
    *,
    require_out: bool = False,
    max_seeds: int = 64,
    seed: int = 0,
    timeout: int = 120,
) -> CaseSpec:
    """Capture a case in a subprocess and save it under out_dir."""
    import subprocess
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_out = out_dir / "capture.json"
    cmd = [
        sys.executable, "-m", "concurrent_oracle.case_capture",
        "--harness", str(harness_path),
        "--target-api", target_api,
        "--out-dir", str(out_dir),
        "--max-seeds", str(max_seeds),
        "--seed", str(seed),
    ]
    if require_out:
        cmd.append("--require-out")
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("TORCH_NUM_THREADS", "1")
    package_parent = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = package_parent + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    if proc.returncode != 0:
        raise CaptureError(
            f"capture subprocess failed (rc={proc.returncode})\n"
            f"stdout:\n{proc.stdout[-4000:]}\nstderr:\n{proc.stderr[-4000:]}"
        )
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    if not data.get("ok"):
        raise CaptureError(f"capture failed: {data}")
    return CaseSpec.from_dict(data["spec"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", required=True)
    ap.add_argument("--target-api", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--require-out", action="store_true")
    ap.add_argument("--max-seeds", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        case, meta = capture_in_process(
            Path(args.harness), args.target_api,
            require_out=args.require_out,
            max_seeds=args.max_seeds,
            seed=args.seed,
        )
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                          "traceback": traceback.format_exc()}, ensure_ascii=False))
        return 1
    if case is None:
        print(json.dumps({"ok": False, "error": "no target call captured", "meta": meta}, ensure_ascii=False))
        return 1

    case_path = out_dir / "case.pt"
    try:
        import torch
        torch.save(case, case_path)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"torch.save failed: {exc}", "meta": meta}, ensure_ascii=False))
        return 1
    spec = CaseSpec(
        target_api=args.target_api,
        case_path=str(case_path),
        seed_hex=meta.get("seed_hex", ""),
        call_line=meta.get("call_index"),
        has_out=bool(meta.get("has_out")),
        kwargs_keys=list(meta.get("kwargs_keys", [])),
        args_kinds=list(meta.get("args_kinds", [])),
        captured_exception=meta.get("captured_exception"),
        require_out=bool(args.require_out),
        notes=[f"call_count={meta.get('call_count')}", f"seed_count={meta.get('seed_count')}"],
    )
    (out_dir / "case.json").write_text(json.dumps(spec.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"ok": True, "spec": spec.to_dict(), "meta": meta}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
