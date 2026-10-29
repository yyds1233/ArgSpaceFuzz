"""JSON/Markdown report helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_summary(path: Path, title: str, rows: List[Dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    if not rows:
        lines.append("No results.")
    else:
        keys = list(rows[0].keys())
        lines.append("| " + " | ".join(keys) + " |")
        lines.append("|" + "|".join("---" for _ in keys) + "|")
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def status_counts(oracle_results: List[Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in oracle_results:
        for c in getattr(r, "oracle_results", []) or []:
            counts[c.status] = counts.get(c.status, 0) + 1
    return counts
