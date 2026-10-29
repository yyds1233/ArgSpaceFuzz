import tempfile
import unittest
from pathlib import Path

from concurrent_oracle.harness_validator import validate_harness
from concurrent_oracle.schemas import ConcurrentTestPlan, HarnessArtifact
from concurrent_oracle.transformer import transform_harness

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures" / "harnesses"


class TestHarnessValidator(unittest.TestCase):
    def _generate(self, td: str):
        artifact = HarnessArtifact("id", "pytorch", "torch.addmm", str(FIXTURES / "llm.torch.addmm.py"))
        plan = ConcurrentTestPlan(
            "plan_x", "id", "independent_calls",
            execution={"external_threads": [2], "internal_threads": [1], "trials": 2},
            oracles=["O3"], target_api="torch.addmm", share_keys=[],
        )
        gen = transform_harness(artifact, plan, Path(td) / "case.pt", Path(td))
        return Path(gen.concurrent_harness)

    def test_valid_template(self):
        with tempfile.TemporaryDirectory() as td:
            p = self._generate(td)
            result = validate_harness(p)
            self.assertEqual(result["status"], "PASS", result)

    def test_rejects_atheris(self):
        with tempfile.TemporaryDirectory() as td:
            p = self._generate(td)
            text = p.read_text(encoding="utf-8")
            p.write_text("import atheris\n" + text, encoding="utf-8")
            result = validate_harness(p)
            self.assertEqual(result["status"], "FAIL")
            self.assertTrue(any("atheris" in e for e in result["errors"]))


if __name__ == "__main__":
    unittest.main()
