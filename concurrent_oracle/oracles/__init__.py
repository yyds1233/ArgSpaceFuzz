"""Oracle registry."""
from __future__ import annotations

from . import consistency, context, independence, recovery, safety, state, tracing

ORACLES = {
    "O1": safety,
    "O2": consistency,
    "O3": independence,
    "O4": state,
    "O5": recovery,
    "O6": context,
    "O7": tracing,
}


def get_oracle(oracle_id: str):
    return ORACLES.get(oracle_id)
