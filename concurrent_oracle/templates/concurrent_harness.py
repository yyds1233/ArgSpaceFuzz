#!/usr/bin/env python3
"""Generated concurrent harness template.

Placeholders are replaced by concurrent_oracle.transformer:
  __TARGET_API__, __CASE_PATH__, __CASE_PATHS__, __PLAN_ID__,
  __SHARE_KEYS__, __THREAD_CASE_MAP__, __THREAD_ROLES__,
  __RECOVERY_CASE_INDEX__, __CONTEXT_PROBE__, __CONTEXT_MODE__

The generated file is standalone (stdlib + torch) and does not include a fuzz loop.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

import torch

TARGET_API = "__TARGET_API__"
CASE_PATH = "__CASE_PATH__"
CASE_PATHS = __CASE_PATHS__
PLAN_ID = "__PLAN_ID__"
SHARE_KEYS = __SHARE_KEYS__
THREAD_CASE_MAP = __THREAD_CASE_MAP__
THREAD_ROLES = __THREAD_ROLES__
RECOVERY_CASE_INDEX = __RECOVERY_CASE_INDEX__
CONTEXT_PROBE = __CONTEXT_PROBE__
CONTEXT_MODE = "__CONTEXT_MODE__"


def _load_one(path):
    case = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(case, dict) or "args" not in case:
        raise RuntimeError(f"invalid case file: {path}")
    return {
        "args": list(case.get("args") or []),
        "kwargs": dict(case.get("kwargs") or {}),
        "trace_input": case.get("trace_input"),
    }


if CASE_PATHS:
    BASE_CASES = [_load_one(p) for p in CASE_PATHS]
else:
    BASE_CASES = [_load_one(CASE_PATH)]


def _clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_clone(v) for v in value)
    if isinstance(value, list):
        return [_clone(v) for v in value]
    if isinstance(value, dict):
        return {k: _clone(v) for k, v in value.items()}
    return copy.deepcopy(value)


def _digest(value):
    if isinstance(value, torch.Tensor):
        t = value.detach()
        try:
            if t.is_complex():
                s = float(t.abs().sum().item())
            else:
                s = float(t.to(torch.float64).sum().item())
        except Exception:
            s = None
        return {
            "kind": "tensor",
            "dtype": str(t.dtype),
            "shape": list(t.shape),
            "numel": int(t.numel()),
            "sum": s,
        }
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [_digest(v) for v in value]}
    if isinstance(value, list):
        return {"kind": "list", "items": [_digest(v) for v in value]}
    if isinstance(value, dict):
        return {"kind": "dict", "items": {str(k): _digest(v) for k, v in value.items()}}
    return {"kind": type(value).__name__, "repr": repr(value)[:300]}


def _snapshot(case_index=0):
    case = BASE_CASES[case_index]
    return {
        "args": [_digest(v) for v in case["args"]],
        "kwargs": {str(k): _digest(v) for k, v in case["kwargs"].items()},
    }


def _build_case(shared, case_index=0):
    case = BASE_CASES[case_index]
    base_args = case["args"]
    base_kwargs = case["kwargs"]
    args = [_clone(v) for v in base_args]
    kwargs = {k: _clone(v) for k, v in base_kwargs.items()}
    if shared:
        if "out" in SHARE_KEYS and "out" in base_kwargs:
            kwargs["out"] = base_kwargs["out"]
        if "args" in SHARE_KEYS:
            args = list(base_args)
        if "args:0" in SHARE_KEYS and base_args:
            args[0] = base_args[0]
    return args, kwargs, _clone(case.get("trace_input"))


def _resolve_target():
    module_name, _, attr = TARGET_API.rpartition(".")
    if not module_name:
        raise RuntimeError(f"invalid target API: {TARGET_API}")
    for candidate in (module_name, "torch.nn.functional", "torch"):
        try:
            mod = importlib.import_module(candidate)
            if hasattr(mod, attr):
                return getattr(mod, attr)
        except Exception:
            continue
    raise RuntimeError(f"target attribute not found: {TARGET_API}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="JSON config with threads/internal_threads/trials/shared")
    ap.add_argument("--out", required=True, help="output observation JSON")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    threads = int(cfg.get("threads", 1))
    internal_threads = int(cfg.get("internal_threads", 1))
    trials = int(cfg.get("trials", 1))
    shared = bool(cfg.get("shared", False))
    timeout_seconds = float(cfg.get("timeout_seconds", 60))
    barrier_timeout = float(cfg.get("barrier_timeout", 15))
    enable_recovery = bool(cfg.get("enable_recovery", True))
    context_mode = str(cfg.get("context_mode") or CONTEXT_MODE or "call")

    case_map = list(cfg.get("thread_case_map") or THREAD_CASE_MAP or [])
    if not case_map:
        case_map = [0] * threads
    if len(case_map) < threads:
        case_map = case_map + [case_map[-1]] * (threads - len(case_map))
    roles = list(THREAD_ROLES or [""] * threads)
    if len(roles) < threads:
        roles = roles + [""] * (threads - len(roles))

    torch.set_num_threads(max(1, internal_threads))
    target = _resolve_target()

    result_digests = {str(i): [] for i in range(threads)}
    exceptions = []
    calls = {"n": 0, "ok": 0}
    probe = {str(i): [] for i in range(threads)}
    lock = threading.Lock()
    barrier = threading.Barrier(threads)
    before = _snapshot(case_map[0] if case_map else 0)
    main_rng_before = torch.get_rng_state()

    def worker(tid):
        case_index = case_map[tid] if tid < len(case_map) else 0
        role = roles[tid] if tid < len(roles) else ""
        try:
            barrier.wait(timeout=barrier_timeout)
        except Exception as exc:
            with lock:
                exceptions.append({"thread": tid, "role": role, "trial": -1,
                                   "type": type(exc).__name__,
                                   "message": f"barrier: {exc}"})
            return
        for trial in range(trials):
            call_args, call_kwargs, trace_input = _build_case(shared, case_index)
            with lock:
                calls["n"] += 1
            try:
                if context_mode == "fork_rng":
                    state_before = torch.get_rng_state()
                    with target(*call_args, **call_kwargs):
                        torch.rand(1)
                    state_after = torch.get_rng_state()
                    restored = bool(torch.equal(state_before, state_after))
                    result = {"kind": "rng_probe", "restored": restored}
                    with lock:
                        calls["ok"] += 1
                        result_digests[str(tid)].append(_digest(result))
                        probe[str(tid)].append({"rng_restored": restored})
                    continue

                if context_mode in ("symbolic", "make_fx"):
                    traced = target(*call_args, **call_kwargs)
                    if trace_input is not None and callable(traced):
                        result = traced(trace_input)
                    else:
                        result = traced
                else:
                    result = target(*call_args, **call_kwargs)

                with lock:
                    calls["ok"] += 1
                    result_digests[str(tid)].append(_digest(result))
                    if CONTEXT_PROBE and context_mode == "grad":
                        probe[str(tid)].append({"grad_enabled": bool(torch.is_grad_enabled())})
            except BaseException as exc:
                with lock:
                    exceptions.append({
                        "thread": tid,
                        "role": role,
                        "case_index": case_index,
                        "trial": trial,
                        "type": type(exc).__name__,
                        "message": str(exc)[:1000],
                        "traceback": traceback.format_exc()[-2000:],
                    })

    started = time.time()
    workers = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(threads)]
    for t in workers:
        t.start()
    deadline = started + timeout_seconds
    for t in workers:
        remaining = max(0.0, deadline - time.time())
        t.join(timeout=remaining)
    elapsed = time.time() - started
    timeouts = sum(1 for t in workers if t.is_alive())

    after = _snapshot(case_map[0] if case_map else 0)
    shared_digest = None
    shared_path = None
    primary = BASE_CASES[case_map[0] if case_map else 0]
    primary_args, primary_kwargs = primary["args"], primary["kwargs"]
    if shared and "out" in primary_kwargs and isinstance(primary_kwargs["out"], torch.Tensor):
        shared_digest = _digest(primary_kwargs["out"])
        shared_path = str(Path(args.out).with_suffix(".shared.pt"))
        torch.save(primary_kwargs["out"], shared_path)
    elif shared and "args:0" in SHARE_KEYS and primary_args and isinstance(primary_args[0], torch.Tensor):
        shared_digest = _digest(primary_args[0])
        shared_path = str(Path(args.out).with_suffix(".shared.pt"))
        torch.save(primary_args[0], shared_path)

    main_grad_enabled = bool(torch.is_grad_enabled()) if (CONTEXT_PROBE and context_mode == "grad") else None
    main_rng_restored = None
    if context_mode == "fork_rng":
        main_rng_after = torch.get_rng_state()
        main_rng_restored = bool(torch.equal(main_rng_before, main_rng_after))

    recovery = {"attempted": False}
    if enable_recovery and RECOVERY_CASE_INDEX is not None:
        recovery["attempted"] = True
        try:
            r_args, r_kwargs, r_trace_input = _build_case(False, int(RECOVERY_CASE_INDEX))
            if context_mode == "fork_rng":
                before_state = torch.get_rng_state()
                with target(*r_args, **r_kwargs):
                    torch.rand(1)
                recovery.update({"success": bool(torch.equal(before_state, torch.get_rng_state()))})
            elif context_mode in ("symbolic", "make_fx"):
                traced = target(*r_args, **r_kwargs)
                r_result = traced(r_trace_input) if r_trace_input is not None and callable(traced) else traced
                recovery.update({"success": True, "digest": _digest(r_result)})
            else:
                r_result = target(*r_args, **r_kwargs)
                recovery.update({"success": True, "digest": _digest(r_result)})
        except BaseException as exc:
            recovery.update({
                "success": False,
                "type": type(exc).__name__,
                "message": str(exc)[:1000],
                "traceback": traceback.format_exc()[-2000:],
            })

    status = "TIMEOUT" if timeouts else "COMPLETED"
    observation = {
        "plan_id": PLAN_ID,
        "target_api": TARGET_API,
        "status": status,
        "threads": threads,
        "internal_threads": internal_threads,
        "trials": trials,
        "shared": shared,
        "share_keys": SHARE_KEYS,
        "case_map": case_map,
        "thread_roles": roles,
        "context_mode": context_mode,
        "calls": calls["n"],
        "successful_calls": calls["ok"],
        "exceptions": exceptions,
        "timeouts": timeouts,
        "elapsed_seconds": elapsed,
        "result_digests": result_digests,
        "shared_digest": shared_digest,
        "shared_path": shared_path,
        "input_digests_before": before,
        "input_digests_after": after,
        "context_probe": probe,
        "main_grad_enabled": main_grad_enabled,
        "main_rng_restored": main_rng_restored,
        "recovery": recovery,
    }
    Path(args.out).write_text(json.dumps(observation, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if not timeouts else 2


if __name__ == "__main__":
    raise SystemExit(main())
