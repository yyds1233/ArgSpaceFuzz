"""End-to-end P0 pipeline: analyze -> plan -> capture -> generate -> run -> oracle."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .api_suitability_analyzer import analyze_artifact
from .case_capture import capture_case
from .case_synthesis import (
    synthesize_fork_rng_case, synthesize_grad_mode_cases,
    synthesize_invalid_case, synthesize_trace_case,
)
from .config import EngineConfig
from .contract_registry import ContractRegistry
from .harness_loader import load_harness
from .harness_validator import validate_harness
from .oracle_engine import evaluate_oracles, save_oracle_result
from .reporter import write_json, write_summary
from .runner import run_plan
from .scenario_generator import generate_plans
from .transformer import transform_harness


def run_pipeline(
    harness_path: Path,
    cfg: EngineConfig,
    *,
    psr_path: Optional[Path] = None,
    max_plans: Optional[int] = None,
    execute: bool = True,
    oracle_filter: Optional[List[str]] = None,
) -> Dict[str, Any]:
    cfg.ensure_outputs()
    registry = ContractRegistry(cfg.contract_dir)
    artifact = load_harness(harness_path, psr_path=psr_path, framework=cfg.framework)
    artifact_path = cfg.path("artifacts", f"{artifact.artifact_id}.json")
    artifact.save(artifact_path)

    analysis, report = analyze_artifact(artifact, registry, cfg)
    report_path = cfg.path("reports", f"{artifact.artifact_id}.suitability.json")
    write_json(report_path, report)

    plans = generate_plans(
        report, registry, cfg,
        target_api=artifact.api_name,
        has_out=analysis.has_out_kwarg,
    )
    if oracle_filter:
        wanted = {x.strip().upper() for x in oracle_filter if x.strip()}
        plans = [p for p in plans if set(p.oracles) & wanted]
    if max_plans is not None:
        plans = plans[:max_plans]

    plan_results: List[Dict[str, Any]] = []
    for plan in plans:
        plan_path = cfg.path("plans", f"{plan.plan_id}.json")
        result_row: Dict[str, Any] = {
            "plan_id": plan.plan_id,
            "scenario": plan.scenario,
            "oracle": ",".join(plan.oracles),
            "contract": plan.contract_status,
            "status": "SKIPPED",
            "oracle_status": "",
            "reason": "",
        }
        try:
            require_out = plan.scenario == "shared_mutable_output"
            case_dir = cfg.outputs_dir / "cases" / plan.plan_id
            case = None
            if "O1" in plan.oracles:
                case = synthesize_invalid_case(artifact.api_name, case_dir)
                if case is not None:
                    result_row["case_source"] = "synthetic_invalid"
            elif "O6" in plan.oracles:
                grad_cases = synthesize_grad_mode_cases(artifact.api_name, case_dir)
                if grad_cases:
                    plan.case_paths = [s.case_path for s in grad_cases]
                    plan.thread_case_map = [0, 1]
                    plan.thread_roles = ["grad_true", "grad_false"]
                    plan.context_probe = True
                    plan.context_mode = "grad"
                    plan.recovery_case_index = None
                    plan.execution["baseline_case_index"] = 0
                    case = grad_cases[0]
                    result_row["case_source"] = "synthetic_context"
                else:
                    fork_case = synthesize_fork_rng_case(artifact.api_name, case_dir)
                    if fork_case is not None:
                        plan.case_paths = [fork_case.case_path]
                        plan.context_probe = True
                        plan.context_mode = "fork_rng"
                        plan.recovery_case_index = None
                        plan.execution["baseline_case_index"] = 0
                        case = fork_case
                        result_row["case_source"] = "synthetic_fork_rng"
            elif "O7" in plan.oracles:
                built = synthesize_trace_case(artifact.api_name, case_dir)
                if built is not None:
                    trace_case, trace_mode = built
                    plan.case_paths = [trace_case.case_path]
                    plan.context_mode = trace_mode
                    plan.recovery_case_index = 0
                    plan.execution["baseline_case_index"] = 0
                    case = trace_case
                    result_row["case_source"] = f"synthetic_trace_{trace_mode}"
            elif "O5" in plan.oracles:
                invalid = synthesize_invalid_case(artifact.api_name, case_dir)
                if invalid is not None:
                    valid = capture_case(
                        Path(artifact.harness_path), artifact.api_name, case_dir,
                        require_out=False,
                        max_seeds=64,
                        seed=cfg.seed,
                        timeout=max(30, cfg.timeout_seconds),
                    )
                    plan.case_paths = [invalid.case_path, valid.case_path]
                    plan.thread_case_map = [0, 1]
                    plan.thread_roles = ["invalid", "valid"]
                    plan.recovery_case_index = 1
                    plan.execution["baseline_case_index"] = 1
                    case = valid
                    result_row["case_source"] = "mixed_invalid_valid"
            if case is None:
                case = capture_case(
                    Path(artifact.harness_path), artifact.api_name, case_dir,
                    require_out=require_out,
                    max_seeds=64,
                    seed=cfg.seed,
                    timeout=max(30, cfg.timeout_seconds),
                )
                result_row["case_source"] = "captured"
            plan.case_path = case.case_path
            plan.save(plan_path)
            gen = transform_harness(
                artifact, plan, Path(case.case_path),
                cfg.outputs_dir / "generated_harnesses",
            )
            validation = validate_harness(Path(gen.concurrent_harness), gen)
            write_json(cfg.path("reports", f"{plan.plan_id}.validation.json"), validation)
            result_row["generated_harness"] = gen.concurrent_harness
            result_row["validation"] = validation.get("status")
            if validation.get("status") != "PASS":
                result_row["status"] = "VALIDATION_FAILED"
                result_row["reason"] = "; ".join(validation.get("errors", [])[:3])
                plan_results.append(result_row)
                continue

            if not execute:
                result_row["status"] = "GENERATED"
                result_row["reason"] = "execution disabled (generate-only)"
                plan_results.append(result_row)
                continue

            observations = run_plan(Path(gen.concurrent_harness), plan, cfg)
            result = evaluate_oracles(plan, observations, registry)
            result_path = cfg.path("oracle_results", f"{plan.plan_id}.json")
            save_oracle_result(result, result_path)
            result_row["status"] = "EXECUTED"
            result_row["oracle_status"] = ",".join(f"{c.oracle_id}:{c.status}" for c in result.oracle_results)
            result_row["reason"] = result.summary
            result_row["oracle_result"] = str(result_path)
            result_row["execution"] = result.execution_status
        except Exception as exc:
            result_row["status"] = "ERROR"
            result_row["reason"] = f"{type(exc).__name__}: {exc}"
        plan_results.append(result_row)

    summary_rows = [
        {k: r.get(k, "") for k in (
            "plan_id", "scenario", "oracle", "contract",
            "status", "oracle_status", "execution", "reason")}
        for r in plan_results
    ]
    summary_path = write_summary(
        cfg.path("reports", f"{artifact.artifact_id}.summary.md"),
        f"Concurrent Oracle: {artifact.api_name}",
        summary_rows,
    )
    summary_json = write_json(
        cfg.path("reports", f"{artifact.artifact_id}.summary.json"),
        {
            "artifact": artifact.to_dict(),
            "suitability": report.to_dict(),
            "plans": [p.to_dict() for p in plans],
            "results": plan_results,
        },
    )
    return {
        "artifact": artifact.to_dict(),
        "suitability": report.to_dict(),
        "plans": [p.to_dict() for p in plans],
        "results": plan_results,
        "summary_md": str(summary_path),
        "summary_json": str(summary_json),
    }
