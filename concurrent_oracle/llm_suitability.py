"""Optional LLM-assisted suitability analysis.

The LLM never executes tests and never decides a violation.  It returns a
*report* that is merged with deterministic contract status.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

from .schemas import (
    CONDITIONAL,
    INSUFFICIENT_INFO,
    LOW_PRIORITY,
    READY,
    UNKNOWN,
    SuitabilityReport,
)

PROMPT_PATH = Path(__file__).resolve().parent / "prompt_templates" / "suitability.md"
DEFAULT_PROMPT = """You are analyzing whether a PyTorch Atheris harness is worth
converting into a concurrency test. Return ONLY a JSON object with keys:
decision (READY|CONDITIONAL|LOW_PRIORITY|INSUFFICIENT_INFO),
risk_categories (list), candidate_scenarios (list), candidate_oracles (list of O1..O7),
contract_status (SUPPORTED|UNSUPPORTED|UNKNOWN), required_evidence (list),
rationale (string), confidence (0..1), evidence (list of strings).
Do not claim a bug.  Do not claim thread safety without evidence.
The YAML/contract status supplied by the caller is authoritative for
SUPPORTED/UNSUPPORTED; use UNKNOWN when evidence is missing.
"""


def _extract_json(text: str) -> Dict:
    text = (text or "").strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)
    l, r = text.find("{"), text.rfind("}")
    if l != -1 and r > l:
        return json.loads(text[l:r + 1])
    raise ValueError("no JSON object in LLM output")


def _valid_decision(x: str) -> bool:
    return x in (READY, CONDITIONAL, LOW_PRIORITY, INSUFFICIENT_INFO)


def enhance_with_llm(
    artifact,
    analysis,
    rule_report: SuitabilityReport,
    registry,
    *,
    model: str,
    base_url: Optional[str],
    temperature: float = 0.0,
    max_tokens: int = 1500,
) -> SuitabilityReport:
    if not os.environ.get("OPENAI_API_KEY"):
        rule_report.warnings.append("OPENAI_API_KEY missing; using rule report")
        return rule_report
    try:
        from openai import OpenAI
    except Exception as exc:
        rule_report.warnings.append(f"openai unavailable: {exc}")
        return rule_report

    harness_text = Path(artifact.harness_path).read_text(encoding="utf-8", errors="ignore")
    prompt = DEFAULT_PROMPT
    if PROMPT_PATH.exists():
        prompt = PROMPT_PATH.read_text(encoding="utf-8")
    user = (
        f"{prompt}\n\n"
        f"=== TARGET API ===\n{artifact.api_name}\n\n"
        f"=== STATIC ANALYSIS ===\n{json.dumps(analysis.to_dict(), ensure_ascii=False, indent=2)}\n\n"
        f"=== CONTRACT (authoritative statuses) ===\n"
        f"{json.dumps(registry.get(artifact.api_name), ensure_ascii=False, indent=2)}\n\n"
        f"=== RULE REPORT ===\n{json.dumps(rule_report.to_dict(), ensure_ascii=False, indent=2)}\n\n"
        f"=== HARNESS (truncated) ===\n{harness_text[:30000]}\n"
    )
    client_kwargs = {"api_key": os.environ["OPENAI_API_KEY"]}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You produce compact JSON only."},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    raw = resp.choices[0].message.content or ""
    data = _extract_json(raw)

    decision = str(data.get("decision", rule_report.decision))
    if not _valid_decision(decision):
        decision = rule_report.decision
    oracles = data.get("candidate_oracles") or rule_report.candidate_oracles
    if not isinstance(oracles, list):
        oracles = rule_report.candidate_oracles
    oracles = [str(x) for x in oracles if str(x) in ("O1", "O2", "O3", "O4", "O5", "O6", "O7")]

    # Deterministic contract status overrides the LLM.
    aggregate = registry.aggregate_status(artifact.api_name, oracles) if oracles else rule_report.contract_status
    contract_from_llm = str(data.get("contract_status", UNKNOWN))
    if aggregate == UNKNOWN and contract_from_llm in ("SUPPORTED", "UNSUPPORTED"):
        # Do not let the model invent a contract; keep UNKNOWN unless the
        # registry has an explicit status.
        aggregate = UNKNOWN

    return SuitabilityReport(
        artifact_id=rule_report.artifact_id,
        decision=decision,
        risk_categories=list(data.get("risk_categories") or rule_report.risk_categories),
        candidate_scenarios=list(data.get("candidate_scenarios") or rule_report.candidate_scenarios),
        candidate_oracles=oracles,
        contract_status=aggregate,
        required_evidence=list(data.get("required_evidence") or rule_report.required_evidence),
        rationale=str(data.get("rationale") or rule_report.rationale),
        confidence=float(data.get("confidence", rule_report.confidence) or 0.0),
        analysis=rule_report.analysis,
        llm_used=True,
        warnings=list(rule_report.warnings),
    )
