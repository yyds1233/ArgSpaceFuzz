"""Shared data models for the concurrent-oracle pipeline.

The classes mirror the JSON contracts in ``guidance.md`` section 8.  They are
plain dataclasses so the module has no mandatory third-party dependency.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Decision values (guidance §3.3)
READY = "READY"
CONDITIONAL = "CONDITIONAL"
LOW_PRIORITY = "LOW_PRIORITY"
INSUFFICIENT_INFO = "INSUFFICIENT_INFO"
DECISIONS = (READY, CONDITIONAL, LOW_PRIORITY, INSUFFICIENT_INFO)

# Contract status (guidance §5.1)
SUPPORTED = "SUPPORTED"
UNSUPPORTED = "UNSUPPORTED"
UNKNOWN = "UNKNOWN"
CONTRACT_STATUSES = (SUPPORTED, UNSUPPORTED, UNKNOWN)

# Oracle status (guidance §8.5)
PASS = "PASS"
VIOLATION = "VIOLATION"
INCONCLUSIVE = "INCONCLUSIVE"
NOT_APPLICABLE = "NOT_APPLICABLE"
ORACLE_STATUSES = (PASS, VIOLATION, INCONCLUSIVE, NOT_APPLICABLE)

# Execution status.
COMPLETED = "COMPLETED"
TIMEOUT = "TIMEOUT"
CRASH = "CRASH"
ERROR = "ERROR"
EXECUTION_STATUSES = (COMPLETED, TIMEOUT, CRASH, ERROR)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def dumps(value: Any, indent: int = 2) -> str:
    return json.dumps(_jsonable(value), indent=indent, ensure_ascii=False)


@dataclass
class HarnessArtifact:
    artifact_id: str
    framework: str
    api_name: str
    harness_path: str
    psr_path: Optional[str] = None
    metadata_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HarnessArtifact":
        return cls(**{k: data.get(k) for k in (
            "artifact_id", "framework", "api_name", "harness_path",
            "psr_path", "metadata_path", "metadata") if k in data})

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dumps(self), encoding="utf-8")
        return path


@dataclass
class HarnessAnalysis:
    artifact_id: str
    target_api: str
    call_sites: List[Dict[str, Any]] = field(default_factory=list)
    has_out_kwarg: bool = False
    has_inplace_call: bool = False
    has_instance_method_call: bool = False
    instance_method_uncertain: bool = False
    has_fuzzed_data_provider: bool = False
    has_atheris_setup: bool = False
    has_atheris_fuzz: bool = False
    has_blanket_except: bool = False
    context_managers: List[str] = field(default_factory=list)
    tracing_calls: List[str] = field(default_factory=list)
    shared_state_clues: List[str] = field(default_factory=list)
    call_count: int = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HarnessAnalysis":
        return cls(**{k: data.get(k) for k in cls.__dataclass_fields__ if k in data})


@dataclass
class SuitabilityReport:
    artifact_id: str
    decision: str
    risk_categories: List[str] = field(default_factory=list)
    candidate_scenarios: List[str] = field(default_factory=list)
    candidate_oracles: List[str] = field(default_factory=list)
    contract_status: str = UNKNOWN
    required_evidence: List[str] = field(default_factory=list)
    rationale: str = ""
    confidence: float = 0.0
    rule_candidates: List[str] = field(default_factory=list)
    llm_candidates: List[str] = field(default_factory=list)
    llm_evidence: List[str] = field(default_factory=list)
    analysis: Dict[str, Any] = field(default_factory=dict)
    llm_used: bool = False
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SuitabilityReport":
        return cls(**{k: data.get(k) for k in cls.__dataclass_fields__ if k in data})


@dataclass
class ConcurrentTestPlan:
    plan_id: str
    artifact_id: str
    scenario: str
    execution: Dict[str, Any] = field(default_factory=dict)
    state_model: str = "independent_copies"
    oracles: List[str] = field(default_factory=list)
    expected_relation: str = "requires_contract_validation"
    timeout_seconds: int = 60
    contract_status: str = UNKNOWN
    share_keys: List[str] = field(default_factory=list)
    target_api: str = ""
    case_path: Optional[str] = None
    case_paths: List[str] = field(default_factory=list)
    thread_case_map: List[int] = field(default_factory=list)
    thread_roles: List[str] = field(default_factory=list)
    recovery_case_index: Optional[int] = None
    context_probe: bool = False
    context_mode: str = "call"
    expected_exception_types: List[str] = field(default_factory=list)
    expected_behavior: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConcurrentTestPlan":
        return cls(**{k: data.get(k) for k in cls.__dataclass_fields__ if k in data})

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dumps(self), encoding="utf-8")
        return path


@dataclass
class CaseSpec:
    target_api: str
    case_path: str
    seed_hex: str = ""
    call_line: Optional[int] = None
    has_out: bool = False
    kwargs_keys: List[str] = field(default_factory=list)
    args_kinds: List[str] = field(default_factory=list)
    captured_exception: Optional[str] = None
    require_out: bool = False
    expected_exception_types: List[str] = field(default_factory=list)
    expected_behavior: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CaseSpec":
        return cls(**{k: data.get(k) for k in cls.__dataclass_fields__ if k in data})


@dataclass
class ConcurrentHarnessArtifact:
    artifact_id: str
    source_harness: str
    concurrent_harness: str
    plan_id: str
    generation_status: str = "GENERATED"
    validation_status: str = "PENDING"
    generation_attempt: int = 1
    oracle_bindings: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConcurrentHarnessArtifact":
        return cls(**{k: data.get(k) for k in cls.__dataclass_fields__ if k in data})


@dataclass
class Observation:
    plan_id: str
    execution_status: str = COMPLETED
    config_id: str = ""
    role: str = ""
    completeness: str = "complete"
    calls: int = 0
    successful_calls: int = 0
    exceptions: List[Dict[str, Any]] = field(default_factory=list)
    timeouts: int = 0
    elapsed_seconds: float = 0.0
    thread_count: int = 1
    internal_threads: int = 1
    trials: int = 0
    result_digests: Dict[str, Any] = field(default_factory=dict)
    shared_digest: Any = None
    shared_path: Optional[str] = None
    input_digests_before: Dict[str, Any] = field(default_factory=dict)
    input_digests_after: Dict[str, Any] = field(default_factory=dict)
    thread_roles: List[str] = field(default_factory=list)
    case_map: List[int] = field(default_factory=list)
    context_probe: Dict[str, Any] = field(default_factory=dict)
    main_grad_enabled: Optional[bool] = None
    main_rng_restored: Optional[bool] = None
    recovery: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)
    stderr: str = ""
    returncode: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Observation":
        return cls(**{k: data.get(k) for k in cls.__dataclass_fields__ if k in data})


@dataclass
class OracleCheckResult:
    oracle_id: str
    status: str
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OracleCheckResult":
        return cls(**{k: data.get(k) for k in cls.__dataclass_fields__ if k in data})


@dataclass
class OracleResult:
    plan_id: str
    scenario: str
    execution_status: str
    observations: Dict[str, Any] = field(default_factory=dict)
    oracle_results: List[OracleCheckResult] = field(default_factory=list)
    reproduction_rate: Optional[float] = None
    failure_artifacts: List[str] = field(default_factory=list)
    config_results: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OracleResult":
        oracles = [OracleCheckResult.from_dict(x) for x in data.get("oracle_results", [])]
        return cls(
            plan_id=data.get("plan_id", ""),
            scenario=data.get("scenario", ""),
            execution_status=data.get("execution_status", ""),
            observations=data.get("observations", {}),
            oracle_results=oracles,
            reproduction_rate=data.get("reproduction_rate"),
            failure_artifacts=data.get("failure_artifacts", []),
            config_results=data.get("config_results", []),
            summary=data.get("summary", ""),
        )


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
