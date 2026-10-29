import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from concurrent_oracle.harness_validator import validate_harness
from concurrent_oracle.llm_transformer import transform_with_llm
from concurrent_oracle.schemas import ConcurrentTestPlan, HarnessArtifact


class TestLLMTransformerOptional(unittest.TestCase):
    def test_no_key_falls_back_to_template(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            case = root / "case.pt"
            torch.save({
                "target_api": "torch.matmul",
                "args": (torch.randn(2, 2), torch.randn(2, 2)),
                "kwargs": {},
            }, case)
            artifact = HarnessArtifact("id", "pytorch", "torch.matmul", "/tmp/h.py")
            plan = ConcurrentTestPlan(
                "p", "id", "independent_calls",
                execution={"external_threads": [2], "internal_threads": [1],
                           "trials": 1, "shared": False},
                oracles=["O3"], contract_status="SUPPORTED",
                target_api="torch.matmul",
            )
            with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
                gen = transform_with_llm(
                    artifact, plan, case, root / "gen",
                    model="gpt-5.4-mini", base_url="https://api.gpt.ge/v1/",
                )
            self.assertTrue(Path(gen.concurrent_harness).exists())
            result = validate_harness(Path(gen.concurrent_harness))
            self.assertEqual(result["status"], "PASS", result)


if __name__ == "__main__":
    unittest.main()
