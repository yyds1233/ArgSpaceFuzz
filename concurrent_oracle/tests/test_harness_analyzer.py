import unittest
from pathlib import Path

from concurrent_oracle.harness_analyzer import analyze_harness
from concurrent_oracle.harness_loader import load_harness

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures" / "harnesses"


class TestHarnessAnalyzer(unittest.TestCase):
    def test_addmm(self):
        a = load_harness(FIXTURES / "llm.torch.addmm.py")
        info = analyze_harness(a)
        self.assertGreater(info.call_count, 0)
        self.assertTrue(info.has_out_kwarg)
        self.assertTrue(info.has_fuzzed_data_provider)
        self.assertTrue(info.has_atheris_setup)
        self.assertTrue(info.has_atheris_fuzz)

    def test_matmul(self):
        a = load_harness(FIXTURES / "llm.torch.matmul.py")
        info = analyze_harness(a)
        self.assertGreater(info.call_count, 0)
        self.assertFalse(info.errors)

    def test_context(self):
        a = load_harness(FIXTURES / "llm.torch.set_grad_enabled.py")
        info = analyze_harness(a)
        self.assertGreater(info.call_count, 0)
        self.assertTrue(info.context_managers)


if __name__ == "__main__":
    unittest.main()
