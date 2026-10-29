"""Concurrent Oracle: independent concurrency-testing pipeline for ArgSpaceFuzz.

The module intentionally does not import or modify ``screen``.  It consumes
harnesses produced by ``fuzz_api_one`` and optional PSR metadata.
"""

__version__ = "0.1.0"
