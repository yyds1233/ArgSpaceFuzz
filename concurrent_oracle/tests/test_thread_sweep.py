import tempfile
import unittest
from pathlib import Path

import torch

from concurrent_oracle.config import default_config
from concurrent_oracle.oracle_engine import evaluate_oracles
from concurrent_oracle.runner import run_plan
from concurrent_oracle.schemas import ConcurrentTestPlan, HarnessArtifact
from concurrent_oracle.transformer import transform_harness


class TestThreadSweep(unittest.TestCase):
    def test_all_external_threads_execute(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            case_path = root / "case.pt"
            torch.save({
                "target_api": "torch.matmul",
                "args": (torch.randn(2, 2), torch.randn(2, 2)),
                "kwargs": {},
            }, case_path)
            artifact = HarnessArtifact("id", "pytorch", "torch.matmul", "/tmp/h.py")
            plan = ConcurrentTestPlan(
                "p", "id", "independent_calls",
                execution={"external_threads": [2, 4], "internal_threads": [1],
                           "trials": 1, "shared": False},
                oracles=["O3"], contract_status="SUPPORTED",
                target_api="torch.matmul",
            )
            gen = transform_harness(artifact, plan, case_path, root / "gen")
            cfg = default_config(outputs_dir=root / "out")
            observations = run_plan(Path(gen.concurrent_harness), plan, cfg)
            configs = {o.config_id for o in observations if o.config_id}
            self.assertEqual(configs, {"n2", "n4"})
            for cid in ("n2", "n4"):
                roles = [o.role for o in observations if o.config_id == cid]
                self.assertIn("baseline", roles)
                self.assertIn("concurrent", roles)
            result = evaluate_oracles(plan, observations, None)
            self.assertEqual(len(result.config_results), 2)
            self.assertEqual(result.oracle_results[0].status, "PASS")


if __name__ == "__main__":
    unittest.main()
