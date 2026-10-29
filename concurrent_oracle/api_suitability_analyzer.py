"""Combine static analysis, rules, and optional LLM suitability analysis."""
from __future__ import annotations

from typing import Tuple

from .config import EngineConfig
from .contract_registry import ContractRegistry
from .harness_analyzer import analyze_harness
from .llm_suitability import enhance_with_llm
from .schemas import HarnessAnalysis, SuitabilityReport
from .suitability_rules import build_rule_report


def analyze_artifact(
    artifact,
    registry: ContractRegistry,
    cfg: EngineConfig,
) -> Tuple[HarnessAnalysis, SuitabilityReport]:
    analysis = analyze_harness(artifact)
    report = build_rule_report(artifact, analysis, registry)
    if cfg.use_llm:
        try:
            report = enhance_with_llm(
                artifact, analysis, report, registry,
                model=cfg.model, base_url=cfg.base_url,
                temperature=cfg.temperature, max_tokens=cfg.max_tokens,
            )
        except Exception as exc:
            report.warnings.append(f"LLM suitability failed: {type(exc).__name__}: {exc}")
    return analysis, report
