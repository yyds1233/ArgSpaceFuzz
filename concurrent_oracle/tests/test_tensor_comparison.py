import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch

from concurrent_oracle.digest import digests_equal, tensor_digest

TEMPLATE = Path(__file__).resolve().parent.parent / "templates" / "concurrent_harness.py"


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
    spec = importlib.util.spec_from_file_location("co_concurrent_template_test", tmp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestTensorComparison(unittest.TestCase):
    def test_sum_collision_is_detected(self):
        a = tensor_digest(torch.tensor([1, 2]))
        b = tensor_digest(torch.tensor([0, 3]))
        self.assertEqual(a["sum"], b["sum"])
        self.assertNotEqual(a["content_hash"], b["content_hash"])
        self.assertFalse(digests_equal(a, b))

    def test_identical_content(self):
        a = tensor_digest(torch.tensor([1.0, 2.0]))
        b = tensor_digest(torch.tensor([1.0, 2.0]))
        self.assertTrue(digests_equal(a, b))
        self.assertTrue(digests_equal(a, b, rel_tol=1e-6, abs_tol=1e-8))

    def test_tolerance_mode(self):
        a = tensor_digest(torch.tensor([1.0], dtype=torch.float64))
        b = tensor_digest(torch.tensor([1.0 + 1e-9], dtype=torch.float64))
        self.assertFalse(digests_equal(a, b))
        self.assertTrue(digests_equal(a, b, rel_tol=1e-6, abs_tol=1e-8))

    def test_nan_inf_complex_empty(self):
        nan_a = tensor_digest(torch.tensor([float("nan")]))
        nan_b = tensor_digest(torch.tensor([float("nan")]))
        self.assertTrue(digests_equal(nan_a, nan_b, rel_tol=1e-6, abs_tol=1e-8))
        inf_a = tensor_digest(torch.tensor([float("inf")]))
        inf_b = tensor_digest(torch.tensor([float("inf")]))
        self.assertTrue(digests_equal(inf_a, inf_b, rel_tol=1e-6, abs_tol=1e-8))
        cplx = tensor_digest(torch.tensor([1 + 2j, 3 - 4j]))
        self.assertTrue(cplx["is_complex"])
        self.assertTrue(digests_equal(cplx, cplx, rel_tol=1e-6, abs_tol=1e-8))
        empty = tensor_digest(torch.empty(0))
        self.assertTrue(digests_equal(empty, empty))

    def test_sparse_tensor(self):
        t = torch.sparse_coo_tensor(torch.tensor([[0], [1]]), torch.tensor([3.0]), (2, 2))
        a = tensor_digest(t)
        b = tensor_digest(t.clone())
        self.assertEqual(a["layout"], "torch.sparse_coo")
        self.assertTrue(digests_equal(a, b))

    def test_template_digest_matches_shared_digest(self):
        mod = load_template_module()
        t = torch.tensor([1.0, 2.0])
        d1 = tensor_digest(t)
        d2 = mod._digest(t)
        self.assertEqual(d1["content_hash"], d2["content_hash"])
        self.assertEqual(d1["dtype"], d2["dtype"])
        self.assertEqual(d1["shape"], d2["shape"])


if __name__ == "__main__":
    unittest.main()
