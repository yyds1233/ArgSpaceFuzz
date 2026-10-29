import os
import unittest
from pathlib import Path

from concurrent_oracle.api_suitability_analyzer import analyze_artifact
from concurrent_oracle.config import default_config
from concurrent_oracle.contract_registry import ContractRegistry
from concurrent_oracle.harness_loader import load_harness
from concurrent_oracle.scenario_generator import generate_plans

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures" / "harnesses"


class TestPipelineSmoke(unittest.TestCase):
    def test_analyze_addmm(self):
        cfg = default_config(outputs_dir=Path("/tmp/co_unittest"))
        a = load_harness(FIXTURES / "llm.torch.addmm.py")
        analysis, report = analyze_artifact(a, ContractRegistry(cfg.contract_dir), cfg)
        self.assertEqual(report.decision, "READY")
        self.assertIn("O2", report.candidate_oracles)
        self.assertIn("O3", report.candidate_oracles)
        self.assertEqual(report.rule_candidates, report.candidate_oracles)
        self.assertTrue(analysis.has_out_kwarg)
        plans = generate_plans(report, ContractRegistry(cfg.contract_dir), cfg,
                               target_api=a.api_name, has_out=analysis.has_out_kwarg)
        self.assertTrue(plans)
        self.assertTrue(any(p.scenario == "shared_mutable_output" for p in plans))

    @unittest.skipUnless(os.environ.get("RUN_CONCURRENT_SMOKE") == "1",
                         "set RUN_CONCURRENT_SMOKE=1 for the slow end-to-end smoke")
    def test_end_to_end_matmul_o3(self):
        from concurrent_oracle.pipeline import run_pipeline
        cfg = default_config(outputs_dir=Path("/tmp/co_unittest_e2e"),
                             external_threads=[2], trials=2, timeout_seconds=30)
        out = run_pipeline(FIXTURES / "llm.torch.matmul.py", cfg, max_plans=3)
        self.assertTrue(any(r.get("status") == "EXECUTED" for r in out["results"]))

    @unittest.skipUnless(os.environ.get("RUN_CONCURRENT_SMOKE") == "1",
                         "set RUN_CONCURRENT_SMOKE=1 for the slow end-to-end smoke")
    def test_end_to_end_o1_repeat_interleave(self):
        from concurrent_oracle.pipeline import run_pipeline
        cfg = default_config(outputs_dir=Path("/tmp/co_unittest_o1"),
                             external_threads=[1], trials=1, timeout_seconds=30)
        out = run_pipeline(FIXTURES / "llm.torch.repeat_interleave.py", cfg,
                           max_plans=1, oracle_filter=["O1"])
        self.assertTrue(any("O1:PASS" in str(r.get("oracle_status", ""))
                            for r in out["results"]), out["results"])

    @unittest.skipUnless(os.environ.get("RUN_CONCURRENT_SMOKE") == "1",
                         "set RUN_CONCURRENT_SMOKE=1 for the slow end-to-end smoke")
    def test_end_to_end_instance_method(self):
        from concurrent_oracle.pipeline import run_pipeline
        seeded = HERE / "fixtures" / "seeded"
        cfg = default_config(outputs_dir=Path("/tmp/co_unittest_inst"),
                             external_threads=[2], trials=1, timeout_seconds=30)
        out = run_pipeline(seeded / "instance_method_harness.py", cfg,
                           max_plans=1, oracle_filter=["O3"])
        self.assertTrue(any("O3:PASS" in str(r.get("oracle_status", ""))
                            for r in out["results"]), out["results"])

    @unittest.skipUnless(os.environ.get("RUN_CONCURRENT_SMOKE") == "1",
                         "set RUN_CONCURRENT_SMOKE=1 for the slow end-to-end smoke")
    def test_end_to_end_o5_cat(self):
        from concurrent_oracle.pipeline import run_pipeline
        cfg = default_config(outputs_dir=Path("/tmp/co_unittest_o5"),
                             external_threads=[2], trials=2, timeout_seconds=30)
        out = run_pipeline(FIXTURES / "llm.torch.cat.py", cfg, max_plans=1,
                           oracle_filter=["O5"])
        self.assertTrue(any("O5:PASS" in str(r.get("oracle_status", ""))
                            for r in out["results"]), out["results"])

    @unittest.skipUnless(os.environ.get("RUN_CONCURRENT_SMOKE") == "1",
                         "set RUN_CONCURRENT_SMOKE=1 for the slow end-to-end smoke")
    def test_end_to_end_o6_grad(self):
        from concurrent_oracle.pipeline import run_pipeline
        cfg = default_config(outputs_dir=Path("/tmp/co_unittest_o6"),
                             external_threads=[2], trials=2, timeout_seconds=30)
        out = run_pipeline(FIXTURES / "llm.torch.set_grad_enabled.py", cfg,
                           max_plans=1, oracle_filter=["O6"])
        self.assertTrue(any("O6:PASS" in str(r.get("oracle_status", ""))
                            for r in out["results"]), out["results"])

    @unittest.skipUnless(os.environ.get("RUN_CONCURRENT_SMOKE") == "1",
                         "set RUN_CONCURRENT_SMOKE=1 for the slow end-to-end smoke")
    def test_end_to_end_o7_trace(self):
        from concurrent_oracle.pipeline import run_pipeline
        cfg = default_config(outputs_dir=Path("/tmp/co_unittest_o7"),
                             external_threads=[2], trials=1, timeout_seconds=60)
        out = run_pipeline(FIXTURES / "llm.torch.jit.trace.py", cfg,
                           max_plans=1, oracle_filter=["O7"])
        self.assertTrue(any("O7:" in str(r.get("oracle_status", ""))
                            for r in out["results"]), out["results"])

    @unittest.skipUnless(os.environ.get("RUN_CONCURRENT_SMOKE") == "1",
                         "set RUN_CONCURRENT_SMOKE=1 for the slow end-to-end smoke")
    def test_end_to_end_o6_fork_rng(self):
        from concurrent_oracle.pipeline import run_pipeline
        cfg = default_config(outputs_dir=Path("/tmp/co_unittest_fork"),
                             external_threads=[2], trials=2, timeout_seconds=30)
        out = run_pipeline(FIXTURES / "llm.torch.random.fork_rng.py", cfg,
                           max_plans=1, oracle_filter=["O6"])
        self.assertTrue(any("O6:INCONCLUSIVE" in str(r.get("oracle_status", ""))
                            or "O6:PASS" in str(r.get("oracle_status", ""))
                            for r in out["results"]), out["results"])

    @unittest.skipUnless(os.environ.get("RUN_CONCURRENT_SMOKE") == "1",
                         "set RUN_CONCURRENT_SMOKE=1 for the slow end-to-end smoke")
    def test_end_to_end_o7_symbolic_trace(self):
        from concurrent_oracle.pipeline import run_pipeline
        seeded = HERE / "fixtures" / "seeded"
        cfg = default_config(outputs_dir=Path("/tmp/co_unittest_sym"),
                             external_threads=[2], trials=1, timeout_seconds=60)
        out = run_pipeline(seeded / "symbolic_trace_harness.py", cfg,
                           max_plans=1, oracle_filter=["O7"])
        self.assertTrue(any("O7:" in str(r.get("oracle_status", ""))
                            for r in out["results"]), out["results"])

    @unittest.skipUnless(os.environ.get("RUN_CONCURRENT_SMOKE") == "1",
                         "set RUN_CONCURRENT_SMOKE=1 for the slow end-to-end smoke")
    def test_end_to_end_o7_make_fx(self):
        from concurrent_oracle.pipeline import run_pipeline
        seeded = HERE / "fixtures" / "seeded"
        cfg = default_config(outputs_dir=Path("/tmp/co_unittest_mfx"),
                             external_threads=[2], trials=1, timeout_seconds=60)
        out = run_pipeline(seeded / "make_fx_harness.py", cfg,
                           max_plans=1, oracle_filter=["O7"])
        self.assertTrue(any("O7:" in str(r.get("oracle_status", ""))
                            for r in out["results"]), out["results"])

    @unittest.skipUnless(os.environ.get("RUN_CONCURRENT_SMOKE") == "1",
                         "set RUN_CONCURRENT_SMOKE=1 for the slow end-to-end smoke")
    def test_reducer_seeded_o2(self):
        from concurrent_oracle.pipeline import run_pipeline
        from concurrent_oracle.reducer import reduce_configuration
        from concurrent_oracle.schemas import ConcurrentTestPlan
        seeded = HERE / "fixtures" / "seeded"
        cfg = default_config(outputs_dir=Path("/tmp/co_unittest_reduce"),
                             external_threads=[2], trials=3, timeout_seconds=30,
                             contract_dir=seeded / "contracts")
        out = run_pipeline(seeded / "lost_update_harness.py", cfg, max_plans=1)
        self.assertTrue(out["results"])
        plan = ConcurrentTestPlan.from_dict(out["plans"][0])
        gen = out["results"][0]["generated_harness"]
        reduced = reduce_configuration(plan, Path(gen), cfg,
                                       oracle_ids=["O2"],
                                       trigger_statuses=["VIOLATION"],
                                       max_variants=4)
        self.assertIsNotNone(reduced["best"], reduced)

    @unittest.skipUnless(os.environ.get("RUN_CONCURRENT_SMOKE") == "1",
                         "set RUN_CONCURRENT_SMOKE=1 for the slow end-to-end smoke")
    def test_seeded_o2_positive_control(self):
        from concurrent_oracle.pipeline import run_pipeline
        seeded = HERE / "fixtures" / "seeded"
        cfg = default_config(outputs_dir=Path("/tmp/co_unittest_seeded"),
                             external_threads=[2], trials=3,
                             timeout_seconds=30,
                             contract_dir=seeded / "contracts")
        out = run_pipeline(seeded / "lost_update_harness.py", cfg, max_plans=1)
        self.assertTrue(any(
            "VIOLATION" in str(r.get("oracle_status", ""))
            for r in out["results"]
        ), out["results"])


if __name__ == "__main__":
    unittest.main()
