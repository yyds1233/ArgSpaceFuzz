import unittest

from concurrent_oracle.reducer import make_variants
from concurrent_oracle.schemas import ConcurrentTestPlan


class TestReducer(unittest.TestCase):
    def test_variants_are_smaller(self):
        p = ConcurrentTestPlan(
            "p", "id", "shared_inplace_state",
            execution={"external_threads": [2, 4], "trials": 4, "internal_threads": [1, 4]},
            oracles=["O2"], contract_status="SUPPORTED", target_api="torch.example",
            share_keys=["args:0"],
        )
        variants = make_variants(p)
        self.assertTrue(variants)
        self.assertLessEqual(max(variants[0].execution["external_threads"]), 4)
        self.assertLessEqual(variants[0].execution["trials"], 4)
        self.assertTrue(all(v.plan_id != p.plan_id for v in variants))
        self.assertEqual(variants[0].share_keys, p.share_keys)


if __name__ == "__main__":
    unittest.main()
