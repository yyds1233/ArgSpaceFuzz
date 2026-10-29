"""Deterministic concurrent-harness generation from a captured case.

The generated harness is a standalone template; this keeps the first version
auditable and avoids asking an LLM to rewrite arbitrary fuzz harnesses.  The
optional LLM transformer can be added on top for cases the template cannot
represent.
"""
from __future__ import annotations

import json
from pathlib import Path

from .schemas import ConcurrentHarnessArtifact, ConcurrentTestPlan

TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "concurrent_harness.py"


def transform_harness(
    artifact,
    plan: ConcurrentTestPlan,
    case_path: Path,
    out_dir: Path,
    *,
    generation_attempt: int = 1,
) -> ConcurrentHarnessArtifact:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    code = (
        template
        .replace("__TARGET_API__", plan.target_api)
        .replace("__CASE_PATHS__", repr(list(plan.case_paths)))
        .replace("__THREAD_CASE_MAP__", repr(list(plan.thread_case_map)))
        .replace("__THREAD_ROLES__", repr(list(plan.thread_roles)))
        .replace("__RECOVERY_CASE_INDEX__", repr(plan.recovery_case_index))
        .replace("__CONTEXT_PROBE__", repr(bool(plan.context_probe)))
        .replace("__CONTEXT_MODE__", str(plan.context_mode or "call"))
        .replace("__CASE_PATH__", str(Path(case_path).resolve()))
        .replace("__PLAN_ID__", plan.plan_id)
        .replace("__SHARE_KEYS__", repr(list(plan.share_keys)))
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{artifact.artifact_id}__{plan.plan_id}.py"
    out_path.write_text(code, encoding="utf-8")
    return ConcurrentHarnessArtifact(
        artifact_id=artifact.artifact_id,
        source_harness=artifact.harness_path,
        concurrent_harness=str(out_path),
        plan_id=plan.plan_id,
        generation_status="GENERATED",
        validation_status="PENDING",
        generation_attempt=generation_attempt,
        oracle_bindings=list(plan.oracles),
    )
