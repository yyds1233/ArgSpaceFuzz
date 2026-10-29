"""Generate ConcurrentTestPlan objects from a SuitabilityReport."""
from __future__ import annotations

from typing import List

from .config import EngineConfig
from .contract_registry import ContractRegistry
from .schemas import (
    ConcurrentTestPlan,
    SuitabilityReport,
    SUPPORTED,
    UNKNOWN,
    new_id,
)


def _base_execution(cfg: EngineConfig, *, threads: List[int], internal: List[int],
                    trials: int, shared: bool) -> dict:
    return {
        "external_threads": threads,
        "internal_threads": internal,
        "trials": trials,
        "shared": shared,
        "include_baseline": True,
    }


def generate_plans(
    report: SuitabilityReport,
    registry: ContractRegistry,
    cfg: EngineConfig,
    *,
    target_api: str,
    has_out: bool,
) -> List[ConcurrentTestPlan]:
    plans: List[ConcurrentTestPlan] = []
    contract = registry.get(target_api)

    for oracle in report.candidate_oracles:
        status = registry.status(target_api, oracle)
        expected = registry.expected(target_api, oracle)
        if oracle == "O1":
            scenario = "invalid_input_safety"
            execution = _base_execution(cfg, threads=[1], internal=[1], trials=1, shared=False)
            state_model = "invalid_input"
            relation = expected or "raises_expected_exception"
            share_keys: List[str] = []
        elif oracle == "O2":
            scenario = "shared_mutable_output" if has_out else "shared_inplace_state"
            execution = _base_execution(
                cfg, threads=list(cfg.external_threads), internal=[1],
                trials=cfg.trials, shared=True,
            )
            state_model = "shared_mutable_tensor"
            relation = expected or "requires_contract_validation"
            share_keys = ["out"] if has_out else ["args:0"]
        elif oracle == "O3":
            scenario = "independent_calls"
            execution = _base_execution(
                cfg, threads=list(cfg.external_threads),
                internal=list(cfg.internal_threads),
                trials=cfg.trials, shared=False,
            )
            state_model = "independent_copies"
            relation = expected or "same_as_serial"
            share_keys = []
        elif oracle == "O4":
            scenario = "read_only_state"
            execution = _base_execution(
                cfg, threads=list(cfg.external_threads), internal=[1],
                trials=cfg.trials, shared=True,
            )
            state_model = "shared_readonly"
            relation = expected or "inputs_unchanged"
            share_keys = ["args"]
        elif oracle == "O5":
            scenario = "exception_isolation"
            execution = _base_execution(
                cfg, threads=list(cfg.external_threads), internal=[1],
                trials=cfg.trials, shared=False,
            )
            execution["mixed_cases"] = True
            state_model = "mixed_exception_tasks"
            relation = expected or "no_cross_task_pollution"
            share_keys = []
        elif oracle == "O6":
            scenario = "context_isolation"
            execution = _base_execution(
                cfg, threads=list(cfg.external_threads), internal=[1],
                trials=cfg.trials, shared=False,
            )
            execution["context_probe"] = True
            state_model = "thread_local_context"
            relation = expected or "thread_local_state"
            share_keys = []
        elif oracle == "O7":
            scenario = "tracing_lifecycle"
            execution = _base_execution(
                cfg, threads=list(cfg.external_threads), internal=[1],
                trials=cfg.trials, shared=False,
            )
            state_model = "tracing_state"
            relation = expected or "no_trace_cross"
            share_keys = []
        else:
            continue

        plan = ConcurrentTestPlan(
            plan_id=new_id("plan"),
            artifact_id=report.artifact_id,
            scenario=scenario,
            execution=execution,
            state_model=state_model,
            oracles=[oracle],
            expected_relation=relation,
            timeout_seconds=cfg.timeout_seconds,
            contract_status=status,
            share_keys=share_keys,
            target_api=target_api,
            context_probe=bool(execution.get("context_probe", False)),
            notes=[],
        )
        if status == UNKNOWN and oracle in ("O2", "O6", "O7"):
            plan.notes.append("contract UNKNOWN: oracle may only return INCONCLUSIVE")
        if status == SUPPORTED:
            plan.notes.append("contract SUPPORTED: violation can be reported if evidence is complete")
        plans.append(plan)
    return plans
