"""Static validation of generated concurrent harnesses."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List

from .schemas import ConcurrentHarnessArtifact


def validate_harness(path: Path, artifact: ConcurrentHarnessArtifact | None = None) -> Dict:
    path = Path(path)
    errors: List[str] = []
    warnings: List[str] = []
    if not path.exists():
        return {"status": "FAIL", "errors": [f"file not found: {path}"], "warnings": []}
    text = path.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        return {"status": "FAIL", "errors": ["empty generated harness"], "warnings": []}
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        return {"status": "FAIL", "errors": [f"SyntaxError: {exc}"], "warnings": []}

    imports = set()
    atheris_attr = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imports.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "atheris":
                atheris_attr = True
    if "atheris" in imports or atheris_attr:
        errors.append("generated harness must not import or call atheris")
    for marker in ("TARGET_API", "CASE_PATH", "PLAN_ID", "SHARE_KEYS"):
        if marker not in text:
            errors.append(f"missing template marker: {marker}")
    for marker in ("threading.Thread", "threading.Barrier", ".join(",
                   "torch.set_num_threads", "exceptions.append"):
        if marker not in text:
            errors.append(f"missing required concurrency structure: {marker}")
    if "except:" in text:
        errors.append("bare except is not allowed; exceptions must be collected")
    if "TestOneInput" in text:
        warnings.append("generated harness mentions TestOneInput; ensure it is not the only entry point")

    funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for fn in ("main", "worker", "_build_case", "_resolve_target"):
        if fn not in funcs:
            errors.append(f"missing function: {fn}")

    if artifact is not None:
        artifact.validation_status = "PASS" if not errors else "FAIL"
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "warnings": warnings,
        "path": str(path),
    }
