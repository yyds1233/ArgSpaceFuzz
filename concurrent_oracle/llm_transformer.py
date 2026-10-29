"""Optional LLM-assisted concurrent-harness transformation.

The deterministic template in ``transformer.py`` is the default P0 path.  This
module is kept for harnesses the template cannot represent; it still returns a
``ConcurrentHarnessArtifact`` and must pass ``harness_validator`` before use.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

from .schemas import ConcurrentHarnessArtifact, ConcurrentTestPlan, new_id
from .transformer import transform_harness as template_transform

PROMPT_PATH = Path(__file__).resolve().parent / "prompt_templates" / "transform_harness.md"
CODE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_code(text: str) -> str:
    m = CODE_RE.search(text or "")
    if m:
        return m.group(1).strip() + "\n"
    return (text or "").strip() + "\n"


def transform_with_llm(
    artifact,
    plan: ConcurrentTestPlan,
    case_path: Path,
    out_dir: Path,
    *,
    model: str,
    base_url: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4000,
    generation_attempt: int = 1,
) -> ConcurrentHarnessArtifact:
    if not os.environ.get("OPENAI_API_KEY"):
        return template_transform(artifact, plan, case_path, out_dir,
                                  generation_attempt=generation_attempt)
    try:
        from openai import OpenAI
        client_kwargs = {"api_key": os.environ["OPENAI_API_KEY"]}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)
        harness_text = Path(artifact.harness_path).read_text(encoding="utf-8", errors="ignore")
        prompt = PROMPT_PATH.read_text(encoding="utf-8") if PROMPT_PATH.exists() else ""
        user = (
            f"{prompt}\n\n=== PLAN ===\n{json.dumps(plan.to_dict(), ensure_ascii=False, indent=2)}\n\n"
            f"=== CASE PATH ===\n{case_path}\n\n"
            f"=== ORIGINAL HARNESS ===\n{harness_text[:50000]}\n"
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Return one Python code block only."},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        code = _extract_code(resp.choices[0].message.content or "")
        if "TARGET_API" not in code:
            code = f'TARGET_API = "{plan.target_api}"\n' + code
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{artifact.artifact_id}__{plan.plan_id}__llm.py"
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
            warnings=["LLM transformer output must pass harness_validator"],
        )
    except Exception as exc:
        result = template_transform(artifact, plan, case_path, out_dir,
                                    generation_attempt=generation_attempt)
        result.warnings.append(f"LLM transform failed; used template: {type(exc).__name__}: {exc}")
        return result
