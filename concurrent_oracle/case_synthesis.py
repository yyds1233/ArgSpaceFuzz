"""Synthetic invalid-input controls for O1.

A valid fuzz case captured from a harness is not an invalid-input case.  For
the first P0 version we provide explicit, auditable invalid cases for a few
APIs; the concurrent template executes them exactly like a captured case.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .schemas import CaseSpec


def _make(api: str) -> Optional[Tuple[Dict[str, Any], str]]:
    import torch

    if api == "torch.repeat_interleave":
        return {
            "target_api": api,
            "args": (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([-1, -1, -1])),
            "kwargs": {},
        }, "RuntimeError"
    if api == "torch.arange":
        return {
            "target_api": api,
            "args": (0, 10, 0),
            "kwargs": {},
        }, "RuntimeError"
    if api == "torch.cat":
        return {
            "target_api": api,
            "args": ([torch.zeros(2, 3), torch.zeros(2, 4)], 0),
            "kwargs": {},
        }, "RuntimeError"
    if api == "torch.gather":
        return {
            "target_api": api,
            "args": (torch.zeros(2, 2), 0, torch.tensor([[5]])),
            "kwargs": {},
        }, "RuntimeError"
    return None


def synthesize_fork_rng_case(api: str, out_dir: Path) -> Optional[CaseSpec]:
    """O6 control: enter/exit torch.random.fork_rng in each thread."""
    if api != "torch.random.fork_rng":
        return None
    import json
    import torch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    case = {
        "target_api": api,
        "args": (),
        "kwargs": {"devices": [], "enabled": True},
    }
    case_path = out_dir / "case.fork_rng.pt"
    try:
        torch.save(case, case_path)
    except Exception:
        return None
    spec = CaseSpec(
        target_api=api,
        case_path=str(case_path),
        seed_hex="",
        call_line=None,
        has_out=False,
        kwargs_keys=sorted(case["kwargs"].keys()),
        args_kinds=[],
        captured_exception=None,
        require_out=False,
        notes=["synthetic_fork_rng_case"],
    )
    (out_dir / "case.fork_rng.json").write_text(
        json.dumps(spec.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return spec


def synthesize_trace_case(api: str, out_dir: Path):
    """O7 control: return (CaseSpec, trace_mode) for jit/fx/make_fx."""
    import json
    import torch
    from concurrent_oracle.tests.fixtures.seeded.trace_models import TinyModule, tiny_fn

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x = torch.randn(2, 2)
    if api == "torch.jit.trace":
        mode = "jit"
        case = {"target_api": api, "args": (tiny_fn, (x,)), "kwargs": {}}
        args_kinds = ["function", "tuple"]
    elif api == "torch.fx.symbolic_trace":
        mode = "symbolic"
        case = {"target_api": api, "args": (TinyModule(),), "kwargs": {},
                "trace_input": x}
        args_kinds = ["module"]
    elif api == "torch.fx.experimental.proxy_tensor.make_fx":
        mode = "make_fx"
        case = {"target_api": api, "args": (tiny_fn,), "kwargs": {},
                "trace_input": x}
        args_kinds = ["function"]
    else:
        return None

    case_path = out_dir / f"case.trace.{mode}.pt"
    try:
        torch.save(case, case_path)
    except Exception:
        return None
    spec = CaseSpec(
        target_api=api,
        case_path=str(case_path),
        seed_hex="",
        call_line=None,
        has_out=False,
        kwargs_keys=[],
        args_kinds=args_kinds,
        captured_exception=None,
        require_out=False,
        notes=[f"synthetic_trace_case mode={mode}"],
    )
    (out_dir / f"case.trace.{mode}.json").write_text(
        json.dumps(spec.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return spec, mode


def synthesize_grad_mode_cases(api: str, out_dir: Path):
    """O6 control: two synthetic cases with grad mode True/False."""
    if api != "torch.set_grad_enabled":
        return None
    import json
    import torch
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = []
    for mode in (True, False):
        case = {"target_api": api, "args": (bool(mode),), "kwargs": {}}
        case_path = out_dir / f"case.grad_{str(mode).lower()}.pt"
        torch.save(case, case_path)
        spec = CaseSpec(
            target_api=api,
            case_path=str(case_path),
            seed_hex="",
            call_line=None,
            has_out=False,
            kwargs_keys=[],
            args_kinds=["bool"],
            captured_exception=None,
            require_out=False,
            notes=[f"synthetic_context_mode grad_enabled={mode}"],
        )
        (out_dir / f"case.grad_{str(mode).lower()}.json").write_text(
            json.dumps(spec.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        specs.append(spec)
    return specs


def synthesize_invalid_case(api: str, out_dir: Path) -> Optional[CaseSpec]:
    built = _make(api)
    if built is None:
        return None
    case, expected = built
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    case_path = out_dir / "case.invalid.pt"
    import torch
    torch.save(case, case_path)
    spec = CaseSpec(
        target_api=api,
        case_path=str(case_path),
        seed_hex="",
        call_line=None,
        has_out=False,
        kwargs_keys=sorted(case.get("kwargs", {}).keys()),
        args_kinds=[type(a).__name__ for a in case.get("args", ())],
        captured_exception=None,
        require_out=False,
        notes=[f"synthetic_invalid_case expected={expected}"],
    )
    (out_dir / "case.invalid.json").write_text(
        __import__("json").dumps(spec.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return spec
