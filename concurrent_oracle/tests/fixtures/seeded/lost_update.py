"""Synthetic O2 positive control: a deterministic lost-update target.

The target performs a Python-level read-modify-write.  With a non-zero delay
and a shared tensor, two threads can read the same value before either writes,
so the final value is less than the serialized total.
"""
from __future__ import annotations

import time

import torch


def update_(x: torch.Tensor, delay: float = 0.0) -> torch.Tensor:
    value = x.clone()
    if delay > 0:
        time.sleep(float(delay))
    x.copy_(value + 1)
    return x
