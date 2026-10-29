"""Static analysis of original Atheris harnesses.

The analyzer is intentionally conservative: it records call sites and
structural clues, and never executes the harness.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Optional

from .schemas import HarnessAnalysis

TRACING_APIS = (
    "torch.jit.trace",
    "torch.fx.symbolic_trace",
    "torch.fx.experimental.proxy_tensor.make_fx",
    "torch.compile",
)
CONTEXT_APIS = (
    "torch.no_grad",
    "torch.enable_grad",
    "torch.set_grad_enabled",
    "torch.inference_mode",
    "torch.random.fork_rng",
    "torch.autocast",
)


def _dotted(node: ast.AST) -> Optional[str]:
    parts: List[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


def _collect_aliases(tree: ast.AST) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.asname:
                    aliases[a.asname] = a.name
                else:
                    aliases[a.name.split(".")[0]] = a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for a in node.names:
                if a.name == "*":
                    continue
                target = f"{mod}.{a.name}" if mod else a.name
                aliases[a.asname or a.name] = target
    return aliases


def _resolve(node: ast.AST, aliases: Dict[str, str]) -> Optional[str]:
    dotted = _dotted(node)
    if not dotted:
        return None
    head, _, rest = dotted.partition(".")
    if head in aliases:
        base = aliases[head]
        return f"{base}.{rest}" if rest else base
    return dotted


def is_instance_method_api(api: str) -> bool:
    parts = (api or "").split(".")
    return len(parts) >= 3 and parts[-2] in ("Tensor", "Module")


def _same_api(resolved: Optional[str], target: str) -> bool:
    if not resolved or not target:
        return False
    if resolved == target:
        return True
    rt = resolved.split(".")
    tt = target.split(".")
    if not rt or not tt or rt[-1] != tt[-1]:
        return False
    # Instance methods: target torch.Tensor.add_, call x.add_.
    if is_instance_method_api(target):
        if not resolved.startswith("torch."):
            return True
    # Functional aliases: torch.nn.linear vs torch.nn.functional.linear, etc.
    rmod = ".".join(rt[:-1])
    tmod = ".".join(tt[:-1])
    return rmod.startswith(tmod) or tmod.startswith(rmod)


def analyze_harness(artifact, *, max_call_sites: int = 100) -> HarnessAnalysis:
    path = Path(artifact.harness_path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    tree = ast.parse(text, filename=str(path))
    aliases = _collect_aliases(tree)
    target = artifact.api_name

    analysis = HarnessAnalysis(artifact_id=artifact.artifact_id, target_api=target)
    analysis.has_fuzzed_data_provider = "FuzzedDataProvider" in text
    analysis.has_atheris_setup = "atheris.Setup" in text
    analysis.has_atheris_fuzz = "atheris.Fuzz" in text

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            resolved = _resolve(node.func, aliases)
            if _same_api(resolved, target):
                kw_names = [k.arg for k in node.keywords if k.arg]
                func_name = resolved or _dotted(node.func) or "<call>"
                instance_method = bool(
                    is_instance_method_api(target)
                    and resolved
                    and not resolved.startswith("torch.")
                    and resolved.split(".")[-1] == target.split(".")[-1]
                )
                site = {
                    "line": getattr(node, "lineno", None),
                    "col": getattr(node, "col_offset", None),
                    "function": func_name,
                    "resolved": resolved,
                    "arg_count": len(node.args),
                    "keywords": kw_names,
                    "has_out": "out" in kw_names,
                    "inplace": func_name.split(".")[-1].endswith("_"),
                    "instance_method": instance_method,
                    "receiver": resolved.rsplit(".", 1)[0] if instance_method else None,
                    "instance_method_uncertain": instance_method,
                }
                analysis.call_sites.append(site)
                analysis.call_count += 1
                analysis.has_out_kwarg = analysis.has_out_kwarg or site["has_out"]
                analysis.has_inplace_call = analysis.has_inplace_call or site["inplace"]
                analysis.has_instance_method_call = (
                    analysis.has_instance_method_call or instance_method
                )
                analysis.instance_method_uncertain = (
                    analysis.instance_method_uncertain or site["instance_method_uncertain"]
                )
                if len(analysis.call_sites) >= max_call_sites:
                    break
            # Context managers / tracing calls are recorded regardless of target.
            if resolved and any(resolved == c or resolved.startswith(c + ".") for c in CONTEXT_APIS):
                analysis.context_managers.append(resolved)
            if resolved and any(resolved == t or resolved.startswith(t + ".") for t in TRACING_APIS):
                analysis.tracing_calls.append(resolved)
        elif isinstance(node, ast.ExceptHandler):
            if node.type is None or (isinstance(node.type, ast.Name) and node.type.id == "Exception"):
                analysis.has_blanket_except = True

    if analysis.has_out_kwarg:
        analysis.shared_state_clues.append("out= shared output tensor")
    if analysis.has_inplace_call:
        analysis.shared_state_clues.append("in-place operation")
    if analysis.has_instance_method_call:
        analysis.shared_state_clues.append(
            "instance method call; receiver type is statically uncertain"
        )
    if "requires_grad" in text:
        analysis.shared_state_clues.append("requires_grad / autograd state")
    if "Parameter" in text:
        analysis.shared_state_clues.append("Parameter / module state")
    if "generator" in text:
        analysis.shared_state_clues.append("RNG generator")
    if "global " in text:
        analysis.shared_state_clues.append("global state")

    if not analysis.call_sites:
        analysis.errors.append(f"target call not found for {target!r}")
    return analysis
