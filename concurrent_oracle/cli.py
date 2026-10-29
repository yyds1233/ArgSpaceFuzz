"""Independent CLI for concurrent_oracle."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .api_suitability_analyzer import analyze_artifact
from .config import default_config
from .contract_registry import ContractRegistry
from .harness_loader import load_harness
from .oracle_engine import evaluate_oracles, save_oracle_result
from .pipeline import run_pipeline
from .reducer import reduce_configuration
from .reporter import write_json
from .reproducer import reproduce_plan
from .runner import run_plan
from .schemas import ConcurrentTestPlan, OracleResult


def _parse_ints(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--outputs-dir", default=None)
    parser.add_argument("--contract-dir", default=None)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--transformer", choices=["template", "llm"], default="template")
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--threads", default="2,4")
    parser.add_argument("--internal-threads", default="1,4")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--oracle", default=None, help="comma-separated oracle filter, e.g. O3,O5")


def _cfg(args):
    return default_config(
        outputs_dir=Path(args.outputs_dir) if args.outputs_dir else None,
        contract_dir=Path(args.contract_dir) if args.contract_dir else None,
        transformer_mode=args.transformer,
        use_llm=bool(args.use_llm),
        model=args.model,
        temperature=args.temperature,
        external_threads=_parse_ints(args.threads),
        internal_threads=_parse_ints(args.internal_threads),
        trials=args.trials,
        timeout_seconds=args.timeout,
        seed=args.seed,
    )


def cmd_analyze(args) -> int:
    cfg = _cfg(args)
    artifact = load_harness(Path(args.harness), psr_path=Path(args.psr) if args.psr else None)
    registry = ContractRegistry(cfg.contract_dir)
    analysis, report = analyze_artifact(artifact, registry, cfg)
    artifact_path = cfg.path("artifacts", f"{artifact.artifact_id}.json")
    artifact.save(artifact_path)
    report_path = cfg.path("reports", f"{artifact.artifact_id}.suitability.json")
    write_json(report_path, report)
    print(json.dumps({
        "artifact": artifact.to_dict(),
        "analysis": analysis.to_dict(),
        "suitability": report.to_dict(),
        "artifact_path": str(artifact_path),
        "report_path": str(report_path),
    }, indent=2, ensure_ascii=False))
    return 0


def cmd_generate(args) -> int:
    cfg = _cfg(args)
    out = run_pipeline(
        Path(args.harness), cfg,
        psr_path=Path(args.psr) if args.psr else None,
        max_plans=args.max_plans,
        execute=False,
        oracle_filter=args.oracle.split(",") if args.oracle else None,
    )
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


def cmd_run(args) -> int:
    cfg = _cfg(args)
    plan = ConcurrentTestPlan.from_dict(json.loads(Path(args.plan).read_text(encoding="utf-8")))
    registry = ContractRegistry(cfg.contract_dir)
    observations = run_plan(Path(args.generated_harness), plan, cfg)
    result = evaluate_oracles(plan, observations, registry)
    path = save_oracle_result(result, cfg.path("oracle_results", f"{plan.plan_id}.json"))
    print(json.dumps({"result": result.to_dict(), "path": str(path)}, indent=2, ensure_ascii=False))
    return 0


def cmd_reproduce(args) -> int:
    cfg = _cfg(args)
    plan = ConcurrentTestPlan.from_dict(json.loads(Path(args.plan).read_text(encoding="utf-8")))
    out = reproduce_plan(
        plan, Path(args.generated_harness), cfg,
        repeats=args.repeats,
        confirmed=args.confirmed,
    )
    path = cfg.path("reproducers", f"{plan.plan_id}.reproduction.json")
    write_json(path, out)
    print(json.dumps({"reproduction": out, "path": str(path)}, indent=2, ensure_ascii=False))
    return 0


def cmd_reduce(args) -> int:
    cfg = _cfg(args)
    plan = ConcurrentTestPlan.from_dict(json.loads(Path(args.plan).read_text(encoding="utf-8")))
    out = reduce_configuration(
        plan, Path(args.generated_harness), cfg,
        oracle_ids=args.oracle.split(",") if args.oracle else None,
        trigger_statuses=[x.strip().upper() for x in args.trigger_status.split(",") if x.strip()],
        max_variants=args.max_variants,
    )
    path = cfg.path("reproducers", f"{plan.plan_id}.reduction.json")
    write_json(path, out)
    print(json.dumps({"reduction": out, "path": str(path)}, indent=2, ensure_ascii=False))
    return 0


def cmd_pipeline(args) -> int:
    cfg = _cfg(args)
    out = run_pipeline(
        Path(args.harness), cfg,
        psr_path=Path(args.psr) if args.psr else None,
        max_plans=args.max_plans,
        execute=True,
        oracle_filter=args.oracle.split(",") if args.oracle else None,
    )
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Concurrent oracle for fuzz_api_one harnesses")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("analyze", help="analyze one harness and write a suitability report")
    _common(p)
    p.add_argument("--harness", required=True)
    p.add_argument("--psr", default=None)
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("generate", help="analyze, plan, capture, transform, validate (no execution)")
    _common(p)
    p.add_argument("--harness", required=True)
    p.add_argument("--psr", default=None)
    p.add_argument("--max-plans", type=int, default=None)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("run", help="run one plan against a generated harness")
    _common(p)
    p.add_argument("--plan", required=True)
    p.add_argument("--generated-harness", required=True)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("reproduce", help="repeat a plan and report reproduction rate")
    _common(p)
    p.add_argument("--plan", required=True)
    p.add_argument("--generated-harness", required=True)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--confirmed", type=int, default=None,
                   help="optional number of repeats manually confirmed as a defect")
    p.set_defaults(func=cmd_reproduce)

    p = sub.add_parser("reduce", help="search smaller configurations that still trigger")
    _common(p)
    p.add_argument("--plan", required=True)
    p.add_argument("--generated-harness", required=True)
    p.add_argument("--trigger-status", default="VIOLATION")
    p.add_argument("--max-variants", type=int, default=8)
    p.set_defaults(func=cmd_reduce)

    p = sub.add_parser("pipeline", help="full pipeline for one harness")
    _common(p)
    p.add_argument("--harness", required=True)
    p.add_argument("--psr", default=None)
    p.add_argument("--max-plans", type=int, default=None)
    p.set_defaults(func=cmd_pipeline)
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
