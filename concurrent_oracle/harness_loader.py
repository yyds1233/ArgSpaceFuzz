"""Load original fuzz_api_one harnesses and their metadata."""
from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path
from typing import Optional

from .schemas import HarnessArtifact

BANNER_RE = re.compile(r"^#\s*([A-Za-z_][\w]*)\s*=\s*(.*)$")
TARGET_RE = re.compile(r"^#\s*target_api\s*=\s*(\S+)\s*$", re.MULTILINE)


class HarnessLoadError(RuntimeError):
    pass


def _safe_name(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s.strip())
    return s.strip("._-") or "harness"


def parse_banner(text: str) -> dict:
    meta = {}
    for line in text.splitlines()[:40]:
        m = BANNER_RE.match(line.strip())
        if m:
            meta[m.group(1)] = m.group(2).strip()
    return meta


def infer_api_name(path: Path, text: str) -> Optional[str]:
    m = TARGET_RE.search(text)
    if m:
        return m.group(1).strip()
    name = path.name
    if name.startswith("llm.") and name.endswith(".py"):
        return name[len("llm."):-len(".py")]
    return None


def load_harness(
    harness_path: Path,
    *,
    psr_path: Optional[Path] = None,
    framework: str = "pytorch",
    metadata_path: Optional[Path] = None,
) -> HarnessArtifact:
    harness_path = Path(harness_path).resolve()
    if not harness_path.exists():
        raise HarnessLoadError(f"harness not found: {harness_path}")
    if harness_path.suffix != ".py":
        raise HarnessLoadError(f"harness is not a .py file: {harness_path}")
    text = harness_path.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        raise HarnessLoadError(f"empty harness: {harness_path}")

    try:
        tree = ast.parse(text, filename=str(harness_path))
    except SyntaxError as exc:
        raise HarnessLoadError(f"syntax error in {harness_path}: {exc}") from exc

    funcs = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if "TestOneInput" not in funcs:
        raise HarnessLoadError(f"TestOneInput not found in {harness_path}")

    api_name = infer_api_name(harness_path, text)
    if not api_name:
        raise HarnessLoadError(f"cannot infer target_api from {harness_path}")

    meta = parse_banner(text)
    meta["line_count"] = len(text.splitlines())
    meta["sha256"] = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
    meta["has_test_one_input"] = True

    if psr_path is not None:
        psr_path = Path(psr_path).resolve()
        if not psr_path.exists():
            raise HarnessLoadError(f"psr not found: {psr_path}")
    if metadata_path is not None:
        metadata_path = Path(metadata_path).resolve()
        if not metadata_path.exists():
            raise HarnessLoadError(f"metadata not found: {metadata_path}")

    stem = _safe_name(api_name)
    short = meta["sha256"][:8]
    artifact_id = f"{stem}_{short}"
    return HarnessArtifact(
        artifact_id=artifact_id,
        framework=framework,
        api_name=api_name,
        harness_path=str(harness_path),
        psr_path=str(psr_path) if psr_path else None,
        metadata_path=str(metadata_path) if metadata_path else None,
        metadata=meta,
    )
