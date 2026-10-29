"""API contract registry.

Contracts encode *what is known* about concurrency support.  An unknown
contract is represented as ``UNKNOWN`` rather than assumed thread-safe.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from .schemas import SUPPORTED, UNKNOWN, UNSUPPORTED

# Built-in conservative defaults.  JSON files under contracts/pytorch/ can
# override individual APIs without changing code.
DEFAULT_CONTRACTS: Dict[str, Dict] = {
    "torch.repeat_interleave": {
        "oracles": {
            "O1": {"status": SUPPORTED, "expected": "invalid repeats/output_size raises RuntimeError or ValueError"},
            "O3": {"status": SUPPORTED},
            "O4": {"status": SUPPORTED},
        },
        "notes": "Invalid repeats/output_size should be rejected without a native crash.",
    },
    "torch.arange": {
        "oracles": {
            "O1": {"status": UNKNOWN, "expected": "invalid start/end/step should raise; a native crash is a candidate"},
            "O3": {"status": SUPPORTED},
        },
        "notes": "Historical run contains an FPE crash on torch.arange; do not assume arbitrary exceptions are safe.",
    },
    "torch.sparse_coo_tensor": {
        "oracles": {"O1": {"status": UNKNOWN}, "O3": {"status": SUPPORTED}},
        "notes": "Invalid index/value combinations need contract evidence.",
    },
    "torch.nn.quantized.functional.conv2d": {
        "oracles": {"O1": {"status": UNKNOWN}, "O3": {"status": SUPPORTED}},
        "notes": "Quantized kernel has historical native crashes; classify crashes separately.",
    },
    "torch.tensordot": {
        "oracles": {"O1": {"status": UNKNOWN}, "O3": {"status": SUPPORTED}},
        "notes": "OOM during audit replay is an execution failure, not automatically a correctness violation.",
    },
    "torch.addmm": {
        "oracles": {
            "O2": {"status": UNKNOWN, "expected": "shared out= writes are not documented as thread-safe"},
            "O3": {"status": SUPPORTED},
            "O4": {"status": SUPPORTED},
        },
        "notes": "Independent calls are comparable; shared out= requires contract validation.",
    },
    "torch.cat": {
        "oracles": {
            "O2": {"status": UNKNOWN, "expected": "shared out= writes are not documented as thread-safe"},
            "O3": {"status": SUPPORTED},
            "O4": {"status": SUPPORTED},
            "O5": {"status": SUPPORTED, "expected": "invalid shapes raise RuntimeError"},
        },
        "notes": "Independent calls and read-only inputs are comparable.",
    },
    "torch.gather": {
        "oracles": {
            "O2": {"status": UNKNOWN, "expected": "shared out= writes are not documented as thread-safe"},
            "O3": {"status": SUPPORTED},
            "O4": {"status": SUPPORTED},
            "O5": {"status": SUPPORTED, "expected": "invalid indices raise RuntimeError/IndexError"},
        },
        "notes": "Out-of-range indices are expected to raise.",
    },
    "torch.Tensor.add_": {
        "oracles": {
            "O2": {"status": UNKNOWN, "expected": "shared in-place writes are not documented as thread-safe"},
            "O3": {"status": SUPPORTED},
            "O4": {"status": UNKNOWN, "expected": "the API mutates its receiver by design"},
        },
        "notes": "Instance method; capture is synthetic because Python cannot patch torch.Tensor.add_ safely.",
    },
    "torch.nn.utils.clip_grad_value_": {
        "oracles": {"O2": {"status": UNKNOWN}, "O3": {"status": SUPPORTED}, "O4": {"status": UNKNOWN}},
        "notes": "The API mutates gradients by design; O4 is not a read-only scenario.",
    },
    "torch.nn.init.normal_": {
        "oracles": {"O2": {"status": UNKNOWN}, "O3": {"status": UNKNOWN}, "O4": {"status": UNKNOWN}},
        "notes": "In-place random initialization consumes RNG; deterministic comparison needs a fixed generator/seed.",
    },
    "torch.matmul": {
        "oracles": {"O3": {"status": SUPPORTED}, "O4": {"status": SUPPORTED}},
        "notes": "Pure function with read-only inputs.",
    },
    "torch.bmm": {
        "oracles": {"O3": {"status": SUPPORTED}, "O4": {"status": SUPPORTED}},
        "notes": "Pure function with read-only inputs.",
    },
    "torch.nn.functional.conv2d": {
        "oracles": {"O1": {"status": UNKNOWN}, "O3": {"status": SUPPORTED}, "O4": {"status": SUPPORTED}},
        "notes": "Pure convolution; independent calls are comparable.",
    },
    "torch.nn.linear": {
        "oracles": {"O3": {"status": SUPPORTED}, "O4": {"status": SUPPORTED}},
        "notes": "Pure affine transform; independent calls are comparable.",
    },
    "torch.nn.functional.linear": {
        "oracles": {"O3": {"status": SUPPORTED}, "O4": {"status": SUPPORTED}},
        "notes": "Pure affine transform; independent calls are comparable.",
    },
    "torch.sort": {
        "oracles": {"O3": {"status": SUPPORTED}, "O4": {"status": SUPPORTED}},
        "notes": "Input is read-only; output is new tensors.",
    },
    "torch.max": {
        "oracles": {"O3": {"status": SUPPORTED}, "O4": {"status": SUPPORTED}},
        "notes": "Input is read-only; output is new tensors.",
    },
    "torch.argsort": {
        "oracles": {"O3": {"status": SUPPORTED}, "O4": {"status": SUPPORTED}},
        "notes": "Input is read-only; output is a new tensor.",
    },
    "torch.index_select": {
        "oracles": {"O3": {"status": SUPPORTED}, "O4": {"status": SUPPORTED}, "O5": {"status": SUPPORTED}},
        "notes": "Out-of-range indices are expected to raise.",
    },
    "torch.topk": {
        "oracles": {"O3": {"status": SUPPORTED}, "O4": {"status": SUPPORTED}},
        "notes": "Input is read-only; output is new tensors.",
    },
    "torch.set_grad_enabled": {
        "oracles": {"O6": {"status": SUPPORTED, "expected": "grad mode is thread-local and restored on context exit"}},
        "notes": "Context manager/function with explicit thread-local semantics.",
    },
    "torch.random.fork_rng": {
        "oracles": {"O6": {"status": UNKNOWN, "expected": "RNG state is forked and restored; concurrent thread safety is not documented"}},
        "notes": "Context manager; concurrent fork/restore semantics need contract evidence.",
    },
    "torch.autograd.grad": {
        "oracles": {"O3": {"status": UNKNOWN}, "O6": {"status": UNKNOWN}},
        "notes": "Autograd has shared graph/context state; need contract evidence.",
    },
    "torch.autograd.backward": {
        "oracles": {"O3": {"status": UNKNOWN}, "O6": {"status": UNKNOWN}},
        "notes": "Autograd has shared graph/context state; need contract evidence.",
    },
    "torch.jit.trace": {
        "oracles": {"O7": {"status": UNKNOWN}, "O3": {"status": UNKNOWN}},
        "notes": "Tracing lifecycle and thread safety are not documented as a supported concurrency contract.",
    },
    "torch.fx.symbolic_trace": {
        "oracles": {"O7": {"status": UNKNOWN}, "O3": {"status": UNKNOWN}},
        "notes": "FX tracing lifecycle and thread safety are not confirmed.",
    },
    "torch.fx.experimental.proxy_tensor.make_fx": {
        "oracles": {"O7": {"status": UNKNOWN}, "O3": {"status": UNKNOWN}},
        "notes": "make_fx tracing lifecycle and thread safety are not confirmed.",
    },
    "torch.jit.load": {
        "oracles": {"O7": {"status": UNKNOWN}},
        "notes": "Load/save lifecycle; thread safety unknown.",
    },
    "torch.jit.save": {
        "oracles": {"O7": {"status": UNKNOWN}},
        "notes": "Load/save lifecycle; thread safety unknown.",
    },
}


class ContractRegistry:
    def __init__(self, contract_dir: Optional[Path] = None):
        self.contracts = {k: dict(v) for k, v in DEFAULT_CONTRACTS.items()}
        if contract_dir:
            self._load_dir(Path(contract_dir))

    def _load_dir(self, root: Path) -> None:
        if not root.exists():
            return
        for path in sorted(root.rglob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            api = data.get("api_name")
            if not api:
                continue
            self.contracts[api] = data

    def get(self, api_name: str) -> Dict:
        if api_name in self.contracts:
            return self.contracts[api_name]
        # Unique tail fallback (e.g. torch.nn.linear vs torch.nn.functional.linear).
        tail = api_name.split(".")[-1]
        hits = [v for k, v in self.contracts.items() if k.split(".")[-1] == tail]
        if len(hits) == 1:
            return hits[0]
        return {"api_name": api_name, "oracles": {}, "notes": "no contract registered; status UNKNOWN"}

    def status(self, api_name: str, oracle_id: str) -> str:
        return str(self.get(api_name).get("oracles", {}).get(oracle_id, {}).get("status", UNKNOWN))

    def expected(self, api_name: str, oracle_id: str) -> str:
        return str(self.get(api_name).get("oracles", {}).get(oracle_id, {}).get("expected", ""))

    def aggregate_status(self, api_name: str, oracle_ids: list[str]) -> str:
        statuses = [self.status(api_name, o) for o in oracle_ids]
        if any(s == SUPPORTED for s in statuses):
            return SUPPORTED
        if statuses and all(s == UNSUPPORTED for s in statuses):
            return UNSUPPORTED
        return UNKNOWN
