"""Deterministic pre-screening rules for concurrency-test suitability."""
from __future__ import annotations

from typing import Dict, List, Set

from .contract_registry import ContractRegistry
from .schemas import (
    CONDITIONAL,
    INSUFFICIENT_INFO,
    LOW_PRIORITY,
    READY,
    SUPPORTED,
    UNKNOWN,
    SuitabilityReport,
)

O1_APIS = {
    "torch.repeat_interleave", "torch.arange", "torch.sparse_coo_tensor",
    "torch.nn.quantized.functional.conv2d", "torch.tensordot",
}
O5_APIS = {
    "torch.gather", "torch.cat", "torch.index_select", "torch.arange",
    "torch.sparse_coo_tensor",
}
O6_APIS = {
    "torch.set_grad_enabled", "torch.random.fork_rng",
    "torch.autograd.grad", "torch.autograd.backward",
}
O7_APIS = {"torch.jit.trace", "torch.jit.load", "torch.jit.save"}
LOW_PRIORITY_APIS = {"torch.jit.load", "torch.jit.save"}


def _candidate_oracles(artifact, analysis, registry: ContractRegistry) -> List[str]:
    api = artifact.api_name
    contract = registry.get(api).get("oracles", {})
    out: List[str] = []

    if "O1" in contract or api in O1_APIS:
        if registry.status(api, "O1") != "UNSUPPORTED":
            out.append("O1")
    if analysis.has_out_kwarg or analysis.has_inplace_call:
        out.append("O2")
    if "O3" in contract and registry.status(api, "O3") != "UNSUPPORTED":
        out.append("O3")
    if "O4" in contract or api in {"torch.sort", "torch.max", "torch.argsort", "torch.index_select", "torch.topk"}:
        if registry.status(api, "O4") != "UNSUPPORTED":
            out.append("O4")
    if "O5" in contract or api in O5_APIS:
        if registry.status(api, "O5") != "UNSUPPORTED":
            out.append("O5")
    if "O6" in contract or api in O6_APIS or analysis.context_managers:
        if registry.status(api, "O6") != "UNSUPPORTED":
            out.append("O6")
    if "O7" in contract or api in O7_APIS or analysis.tracing_calls:
        if registry.status(api, "O7") != "UNSUPPORTED":
            out.append("O7")
    return sorted(set(out))


def build_rule_report(
    artifact,
    analysis,
    registry: ContractRegistry,
) -> SuitabilityReport:
    api = artifact.api_name
    candidates = _candidate_oracles(artifact, analysis, registry)
    statuses = {o: registry.status(api, o) for o in candidates}

    risk: List[str] = []
    if analysis.has_out_kwarg:
        risk.append("shared_mutable_output")
    if analysis.has_inplace_call:
        risk.append("shared_mutable_state")
    if analysis.context_managers:
        risk.append("thread_context_state")
    if analysis.tracing_calls:
        risk.append("tracing_state")
    if "O1" in candidates:
        risk.append("invalid_input_contract")
    if "O4" in candidates:
        risk.append("read_only_state")
    if "O5" in candidates:
        risk.append("exception_isolation")

    scenarios: List[str] = []
    if "O1" in candidates:
        scenarios.append("invalid_input_safety")
    if "O2" in candidates:
        scenarios.append("shared_mutable_output" if analysis.has_out_kwarg else "shared_inplace_state")
    if "O3" in candidates:
        scenarios.append("independent_calls")
    if "O4" in candidates:
        scenarios.append("read_only_state")
    if "O5" in candidates:
        scenarios.append("exception_isolation")
    if "O6" in candidates:
        scenarios.append("context_isolation")
    if "O7" in candidates:
        scenarios.append("tracing_lifecycle")

    required: List[str] = []
    for o in candidates:
        if statuses[o] == UNKNOWN:
            if o == "O2":
                required.append("API contract for shared out=/in-place writes")
            elif o == "O1":
                required.append("invalid-input rejection contract")
            elif o == "O6":
                required.append("thread-local context contract")
            elif o == "O7":
                required.append("tracing lifecycle/thread-safety contract")
            else:
                required.append(f"contract evidence for {o}")

    if analysis.errors:
        decision = INSUFFICIENT_INFO
        rationale = "Harness analysis failed: " + "; ".join(analysis.errors)
        confidence = 0.2
    elif api in LOW_PRIORITY_APIS and not candidates:
        decision = LOW_PRIORITY
        rationale = "No high-value concurrency scenario identified for this API."
        confidence = 0.6
    elif any(o in candidates and statuses[o] == SUPPORTED for o in ("O1", "O3", "O4", "O6")):
        decision = READY
        rationale = (
            f"{api} has deterministic scenarios with supported contracts: "
            + ", ".join(o for o in candidates if statuses[o] == SUPPORTED)
        )
        confidence = 0.8
    elif candidates:
        decision = CONDITIONAL
        rationale = (
            f"{api} has candidate scenarios but needs contract evidence: "
            + ", ".join(required)
        )
        confidence = 0.6
    else:
        decision = LOW_PRIORITY
        rationale = "No candidate concurrency oracle identified by the rules."
        confidence = 0.5

    aggregate = registry.aggregate_status(api, candidates) if candidates else UNKNOWN
    return SuitabilityReport(
        artifact_id=artifact.artifact_id,
        decision=decision,
        risk_categories=sorted(set(risk)),
        candidate_scenarios=scenarios,
        candidate_oracles=candidates,
        contract_status=aggregate,
        required_evidence=required,
        rationale=rationale,
        confidence=confidence,
        rule_candidates=candidates,
        analysis=analysis.to_dict(),
        llm_used=False,
    )


def rule_summary(report: SuitabilityReport) -> Dict:
    return {
        "decision": report.decision,
        "candidate_oracles": report.candidate_oracles,
        "candidate_scenarios": report.candidate_scenarios,
        "contract_status": report.contract_status,
        "required_evidence": report.required_evidence,
    }
