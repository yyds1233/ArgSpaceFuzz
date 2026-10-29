"""Run generated concurrent harnesses in isolated subprocesses."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

from .config import EngineConfig
from .observation import load_observation
from .schemas import ConcurrentTestPlan, Observation


def _run_once(
    harness_path: Path,
    plan: ConcurrentTestPlan,
    cfg: EngineConfig,
    *,
    threads: int,
    internal_threads: int,
    trials: int,
    shared: bool,
    tag: str,
    case_map: list | None = None,
    enable_recovery: bool = True,
    config_id: str = "",
    role: str = "",
) -> Observation:
    run_dir = cfg.outputs_dir / "runs" / plan.plan_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / f"{tag}.config.json"
    obs_path = run_dir / f"{tag}.observation.json"
    config = {
        "threads": threads,
        "internal_threads": internal_threads,
        "trials": trials,
        "shared": shared,
        "timeout_seconds": plan.timeout_seconds,
        "barrier_timeout": min(15, max(5, plan.timeout_seconds // 2)),
        "thread_case_map": list(case_map or []),
        "enable_recovery": bool(enable_recovery),
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(internal_threads)
    env["MKL_NUM_THREADS"] = str(internal_threads)
    env["TORCH_NUM_THREADS"] = str(internal_threads)
    env.setdefault("PYTHONUNBUFFERED", "1")
    package_parent = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = package_parent + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [cfg.python, str(harness_path), "--config", str(config_path), "--out", str(obs_path)]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=plan.timeout_seconds + 20, env=env,
        )
        rc = proc.returncode
        stderr = proc.stderr
    except subprocess.TimeoutExpired as exc:
        rc = 124
        stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
    return load_observation(obs_path, plan.plan_id, rc, stderr,
                            config_id=config_id, role=role)


def run_plan(harness_path: Path, plan: ConcurrentTestPlan, cfg: EngineConfig) -> List[Observation]:
    """Run C00 baseline plus every external-thread configuration in the plan."""
    harness_path = Path(harness_path)
    trials = int(plan.execution.get("trials", cfg.trials))
    shared = bool(plan.execution.get("shared", False))
    observations: List[Observation] = []

    external = [int(x) for x in plan.execution.get("external_threads", [2])] or [2]
    internal = [int(x) for x in plan.execution.get("internal_threads", [1])] or [1]
    baseline_shared = shared and plan.state_model in ("shared_mutable_tensor", "shared_readonly")
    baseline_case = int(plan.execution.get("baseline_case_index", 0))

    def baseline(config_id: str, n_ops: int):
        return _run_once(
            harness_path, plan, cfg,
            threads=1, internal_threads=1,
            trials=(trials * n_ops if baseline_shared else trials),
            shared=baseline_shared, tag=f"baseline_{config_id}",
            case_map=[baseline_case], enable_recovery=False,
            config_id=config_id, role="baseline",
        )

    for n in sorted(set(external)):
        config_id = f"n{n}"
        observations.append(baseline(config_id, n))
        observations.append(_run_once(
            harness_path, plan, cfg,
            threads=n, internal_threads=1, trials=trials,
            shared=shared, tag=f"c10_{config_id}",
            case_map=plan.thread_case_map or None,
            config_id=config_id, role="concurrent",
        ))

    for t in sorted({x for x in internal if x > 1}):
        config_id = f"t{t}"
        observations.append(baseline(config_id, 1))
        observations.append(_run_once(
            harness_path, plan, cfg,
            threads=1, internal_threads=t, trials=trials,
            shared=shared, tag=f"c01_{config_id}",
            case_map=[baseline_case], enable_recovery=False,
            config_id=config_id, role="concurrent",
        ))
        for n in sorted(set(external)):
            if n <= 1:
                continue
            cid = f"n{n}_t{t}"
            observations.append(baseline(cid, n))
            observations.append(_run_once(
                harness_path, plan, cfg,
                threads=n, internal_threads=t, trials=trials,
                shared=shared, tag=f"c11_{cid}",
                case_map=plan.thread_case_map or None,
                config_id=cid, role="concurrent",
            ))
    return observations
