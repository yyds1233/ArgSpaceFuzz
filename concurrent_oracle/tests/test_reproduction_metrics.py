import unittest

from concurrent_oracle.reproducer import summarize_statuses


class TestReproductionMetrics(unittest.TestCase):
    def test_inconclusive_is_not_a_confirmed_defect(self):
        out = summarize_statuses([["INCONCLUSIVE"]] * 5)
        self.assertEqual(out["violation_reproduction_rate"], 0.0)
        self.assertEqual(out["candidate_reproduction_rate"], 1.0)
        self.assertEqual(out["inconclusive_rate"], 1.0)
        self.assertEqual(out["reproduction_rate"], 0.0)
        self.assertIsNone(out["confirmed_reproduction_rate"])

    def test_mixed_rates_and_confirmed(self):
        statuses = [["VIOLATION"], ["VIOLATION"], ["INCONCLUSIVE"], ["PASS"], ["VIOLATION"]]
        out = summarize_statuses(statuses, confirmed=2)
        self.assertEqual(out["violation_hits"], 3)
        self.assertEqual(out["candidate_hits"], 4)
        self.assertEqual(out["inconclusive_hits"], 1)
        self.assertAlmostEqual(out["violation_reproduction_rate"], 0.6)
        self.assertAlmostEqual(out["candidate_reproduction_rate"], 0.8)
        self.assertAlmostEqual(out["confirmed_reproduction_rate"], 0.4)


if __name__ == "__main__":
    unittest.main()
