You are analyzing a PyTorch Atheris harness for **concurrency-test suitability**.
You are not executing the harness and you are not deciding that a bug exists.

Return ONLY a JSON object with these keys:

```
{
  "decision": "READY | CONDITIONAL | LOW_PRIORITY | INSUFFICIENT_INFO",
  "risk_categories": ["shared_mutable_state", "..."],
  "candidate_scenarios": ["shared_mutable_output", "independent_calls", "..."],
  "candidate_oracles": ["O1", "O2", "O3", "O4", "O5", "O6", "O7"],
  "contract_status": "SUPPORTED | UNSUPPORTED | UNKNOWN",
  "required_evidence": ["..."],
  "rationale": "short evidence-based explanation",
  "confidence": 0.0,
  "evidence": ["file/line or doc clue"]
}
```

Rules:
- The caller supplies an authoritative contract registry. Do not upgrade UNKNOWN to SUPPORTED.
- Do not output a permanent UNSUITABLE decision. If a scenario is unsupported,
  put it in required_evidence / rationale; the plan layer will mark
  UNSUPPORTED_SCENARIO.
- Only propose an oracle if the harness actually exercises the relevant
  behavior (shared tensor, out=, context, tracing, invalid input, ...).
- `confidence` is your analysis confidence, not a vulnerability probability.
- When evidence is missing, use INSUFFICIENT_INFO or CONDITIONAL rather than
  guessing.
