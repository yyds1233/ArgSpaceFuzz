import tempfile
import unittest
from pathlib import Path

from concurrent_oracle.harness_loader import HarnessLoadError, load_harness

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures" / "harnesses"


class TestHarnessLoader(unittest.TestCase):
    def test_load_addmm(self):
        a = load_harness(FIXTURES / "llm.torch.addmm.py")
        self.assertEqual(a.api_name, "torch.addmm")
        self.assertEqual(a.framework, "pytorch")
        self.assertTrue(a.artifact_id)
        self.assertTrue(Path(a.harness_path).exists())

    def test_missing_file(self):
        with self.assertRaises(HarnessLoadError):
            load_harness(FIXTURES / "does_not_exist.py")

    def test_no_test_one_input(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "llm.torch.foo.py"
            p.write_text("# target_api=torch.foo\nimport torch\n", encoding="utf-8")
            with self.assertRaises(HarnessLoadError):
                load_harness(p)


if __name__ == "__main__":
    unittest.main()
