import unittest

from concurrent_oracle.schemas import (
    CaseSpec, ConcurrentHarnessArtifact, ConcurrentTestPlan, HarnessAnalysis,
    HarnessArtifact, Observation, OracleCheckResult, OracleResult,
    SuitabilityReport,
)


class TestSchemas(unittest.TestCase):
    def test_harness_artifact_roundtrip(self):
        a = HarnessArtifact("id", "pytorch", "torch.addmm", "/tmp/h.py", metadata={"x": 1})
        b = HarnessArtifact.from_dict(a.to_dict())
        self.assertEqual(a.artifact_id, b.artifact_id)
        self.assertEqual(a.metadata, b.metadata)

    def test_report_plan_roundtrip(self):
        r = SuitabilityReport("id", "READY", candidate_oracles=["O3"], contract_status="SUPPORTED")
        self.assertEqual(SuitabilityReport.from_dict(r.to_dict()).decision, "READY")
        p = ConcurrentTestPlan("p", "id", "independent_calls", execution={"external_threads": [2]},
                               oracles=["O3"], contract_status="SUPPORTED")
        self.assertEqual(ConcurrentTestPlan.from_dict(p.to_dict()).oracles, ["O3"])

    def test_observation_oracle_roundtrip(self):
        o = Observation("p", calls=2, result_digests={"0": []})
        self.assertEqual(Observation.from_dict(o.to_dict()).calls, 2)
        c = OracleCheckResult("O3", "PASS", "ok")
        self.assertEqual(OracleCheckResult.from_dict(c.to_dict()).status, "PASS")
        result = OracleResult("p", "independent_calls", "COMPLETED", oracle_results=[c])
        self.assertEqual(OracleResult.from_dict(result.to_dict()).oracle_results[0].oracle_id, "O3")

    def test_analysis_and_case(self):
        a = HarnessAnalysis("id", "torch.addmm", call_count=1)
        self.assertEqual(HarnessAnalysis.from_dict(a.to_dict()).call_count, 1)
        c = CaseSpec("torch.addmm", "/tmp/case.pt", kwargs_keys=["out"])
        self.assertEqual(CaseSpec.from_dict(c.to_dict()).kwargs_keys, ["out"])
        g = ConcurrentHarnessArtifact("id", "/tmp/a.py", "/tmp/b.py", "p")
        self.assertEqual(ConcurrentHarnessArtifact.from_dict(g.to_dict()).plan_id, "p")


if __name__ == "__main__":
    unittest.main()
