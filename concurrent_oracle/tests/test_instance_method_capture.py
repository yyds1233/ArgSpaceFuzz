import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch

from concurrent_oracle.case_capture import CaptureError, capture_in_process
from concurrent_oracle.harness_analyzer import analyze_harness
from concurrent_oracle.harness_loader import load_harness

HERE = Path(__file__).resolve().parent
SEEDED = HERE / "fixtures" / "seeded"
TEMPLATE = HERE.parent / "templates" / "concurrent_harness.py"


def load_template_module():
    code = TEMPLATE.read_text(encoding="utf-8")
    for key, value in {
        "__CASE_PATHS__": "[]",
        "__THREAD_CASE_MAP__": "[]",
        "__THREAD_ROLES__": "[]",
        "__RECOVERY_CASE_INDEX__": "None",
        "__CONTEXT_PROBE__": "False",
        "__SHARE_KEYS__": "[]",
        "__CONTEXT_MODE__": "call",
        "__TARGET_API__": "torch.addmm",
        "__CASE_PATH__": "/tmp/unused-case.pt",
        "__PLAN_ID__": "test",
    }.items():
        code = code.replace(key, value)
    tmp = Path(tempfile.mkdtemp()) / "template.py"
    tmp.write_text(code, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("co_concurrent_template_inst_test", tmp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestInstanceMethodCapture(unittest.TestCase):
    def test_analyzer_detects_instance_method(self):
        a = load_harness(SEEDED / "instance_method_harness.py")
        info = analyze_harness(a)
        self.assertTrue(info.has_instance_method_call)
        self.assertTrue(info.instance_method_uncertain)
        self.assertTrue(info.call_sites)
        self.assertTrue(info.call_sites[0]["instance_method"])

    def test_alias_capture(self):
        case, meta = capture_in_process(
            SEEDED / "alias_harness.py", "torch.addmm", max_seeds=8
        )
        self.assertIsNotNone(case, meta)
        self.assertGreater(meta["call_count"], 0)

    def test_instance_method_patch_is_diagnostic(self):
        with self.assertRaises(CaptureError) as ctx:
            capture_in_process(
                SEEDED / "instance_method_harness.py", "torch.Tensor.add_", max_seeds=4
            )
        msg = str(ctx.exception)
        self.assertIn("target_patch_unsupported", msg)
        self.assertIn("instance_method=True", msg)

    def test_no_call_is_distinguished_from_patch_failure(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "no_call.py"
            p.write_text(
                "# target_api=torch.addmm\n"
                "import torch\n"
                "def TestOneInput(data):\n    return None\n",
                encoding="utf-8",
            )
            case, meta = capture_in_process(p, "torch.addmm", max_seeds=2)
            self.assertIsNone(case)
            self.assertEqual(meta.get("error"), "no target call captured")

    def test_template_resolves_instance_method(self):
        mod = load_template_module()
        mod.TARGET_API = "torch.Tensor.add_"
        target = mod._resolve_target()
        x = torch.zeros(2, 2)
        y = torch.ones(2, 2)
        target(x, y)
        self.assertTrue(torch.equal(x, y))


if __name__ == "__main__":
    unittest.main()
