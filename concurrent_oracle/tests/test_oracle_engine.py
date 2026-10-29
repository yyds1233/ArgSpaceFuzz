import unittest

from concurrent_oracle.oracle_engine import evaluate_oracles
from concurrent_oracle.schemas import (
    COMPLETED, CRASH, ERROR, TIMEOUT, ConcurrentTestPlan, Observation,
    OracleCheckResult,
)


def tensor_digest(total: float, shape=(2,)):
    return {"kind": "tensor", "dtype": "torch.float32", "shape": list(shape),
            "numel": shape[0], "sum": total}


def obs(plan_id="p", digests=None, shared=None, exceptions=None,
        status=COMPLETED, before=None, after=None):
    return Observation(
        plan_id=plan_id,
        execution_status=status,
        calls=2,
        exceptions=exceptions or [],
        result_digests=digests or {"0": [tensor_digest(3.0)]},
        shared_digest=shared,
        input_digests_before=before or {"args": [tensor_digest(1.0)]},
        input_digests_after=after or {"args": [tensor_digest(1.0)]},
    )


def plan(oracle, contract="SUPPORTED", expected=None):
    p = ConcurrentTestPlan(
        "p", "id", "test", execution={}, oracles=[oracle],
        contract_status=contract, target_api="torch.example",
    )
    if expected is not None:
        p.expected_behavior = "raise"
        p.expected_exception_types = list(expected)
    return p


class TestOracleEngine(unittest.TestCase):
    def test_o3_pass(self):
        result = evaluate_oracles(plan("O3"), [obs(), obs()], None)
        self.assertEqual(result.oracle_results[0].status, "PASS")

    def test_o3_violation(self):
        concurrent = obs(digests={"0": [tensor_digest(4.0)]})
        result = evaluate_oracles(plan("O3"), [obs(), concurrent], None)
        self.assertEqual(result.oracle_results[0].status, "VIOLATION")

    def test_o3_unknown_is_inconclusive(self):
        concurrent = obs(digests={"0": [tensor_digest(4.0)]})
        result = evaluate_oracles(plan("O3", contract="UNKNOWN"), [obs(), concurrent], None)
        self.assertEqual(result.oracle_results[0].status, "INCONCLUSIVE")

    def test_o2_unknown_and_supported(self):
        baseline = obs(shared=tensor_digest(1.0))
        concurrent = obs(shared=tensor_digest(2.0))
        r = evaluate_oracles(plan("O2", contract="UNKNOWN"), [baseline, concurrent], None)
        self.assertEqual(r.oracle_results[0].status, "INCONCLUSIVE")
        r = evaluate_oracles(plan("O2", contract="SUPPORTED"), [baseline, concurrent], None)
        self.assertEqual(r.oracle_results[0].status, "VIOLATION")

    def test_o1_pass_and_crash(self):
        exc = [{"thread": 0, "trial": 0, "type": "RuntimeError", "message": "bad"}]
        p = plan("O1", expected=["RuntimeError"])
        r = evaluate_oracles(p, [obs(exceptions=exc), obs(exceptions=exc)], None)
        self.assertEqual(r.oracle_results[0].status, "PASS")
        r = evaluate_oracles(p, [obs(status=CRASH), obs(status=CRASH)], None)
        self.assertEqual(r.oracle_results[0].status, "VIOLATION")

    def test_o1_partial_failure_is_not_pass(self):
        exc = [{"thread": 0, "trial": 0, "type": "RuntimeError", "message": "bad"}]
        rejected = Observation("p", calls=1, successful_calls=0, exceptions=exc)
        accepted = Observation("p", calls=1, successful_calls=1, exceptions=[])
        p = plan("O1", expected=["RuntimeError"])
        r = evaluate_oracles(p, [rejected, accepted], None)
        self.assertNotEqual(r.oracle_results[0].status, "PASS")
        self.assertEqual(r.oracle_results[0].status, "VIOLATION")

    def test_o1_unexpected_and_missing_are_inconclusive(self):
        p = plan("O1", expected=["RuntimeError"])
        wrong = Observation("p", calls=1, successful_calls=0,
                            exceptions=[{"thread": 0, "type": "KeyError", "message": "x"}])
        r = evaluate_oracles(p, [wrong], None)
        self.assertEqual(r.oracle_results[0].status, "INCONCLUSIVE")
        missing = Observation("p", calls=0, successful_calls=0)
        r = evaluate_oracles(p, [missing], None)
        self.assertEqual(r.oracle_results[0].status, "INCONCLUSIVE")

    def test_o1_undefined_behavior_is_not_applicable(self):
        r = evaluate_oracles(plan("O1"), [obs(), obs()], None)
        self.assertEqual(r.oracle_results[0].status, "NOT_APPLICABLE")

    def test_o4_pass_and_violation(self):
        r = evaluate_oracles(plan("O4"), [obs(), obs()], None)
        self.assertEqual(r.oracle_results[0].status, "PASS")
        changed = obs(after={"args": [tensor_digest(9.0)]})
        r = evaluate_oracles(plan("O4"), [obs(), changed], None)
        self.assertEqual(r.oracle_results[0].status, "VIOLATION")

    def test_o5_pass_and_pollution(self):
        exc = [{"thread": 0, "role": "invalid", "trial": 0, "type": "RuntimeError", "message": "bad"}]
        good = Observation(
            "p", calls=4, successful_calls=3, exceptions=exc,
            thread_roles=["invalid", "valid"], recovery={"attempted": True, "success": True},
            result_digests={"0": [], "1": [tensor_digest(1.0)]},
        )
        r = evaluate_oracles(plan("O5"), [obs(), good], None)
        self.assertEqual(r.oracle_results[0].status, "PASS")
        polluted = Observation(
            "p", calls=4, successful_calls=2,
            exceptions=exc + [{"thread": 1, "role": "valid", "trial": 0,
                               "type": "RuntimeError", "message": "polluted"}],
            thread_roles=["invalid", "valid"], recovery={"attempted": True, "success": True},
            result_digests={"0": [], "1": []},
        )
        r = evaluate_oracles(plan("O5"), [obs(), polluted], None)
        self.assertEqual(r.oracle_results[0].status, "VIOLATION")

    def test_o6_pass_and_leak(self):
        good = Observation(
            "p", calls=2, successful_calls=2,
            thread_roles=["grad_true", "grad_false"],
            context_probe={"0": [{"grad_enabled": True}], "1": [{"grad_enabled": False}]},
            main_grad_enabled=True,
        )
        r = evaluate_oracles(plan("O6"), [obs(), good], None)
        self.assertEqual(r.oracle_results[0].status, "PASS")
        leak = Observation(
            "p", calls=2, successful_calls=2,
            thread_roles=["grad_true", "grad_false"],
            context_probe={"0": [{"grad_enabled": True}], "1": [{"grad_enabled": True}]},
            main_grad_enabled=False,
        )
        r = evaluate_oracles(plan("O6"), [obs(), leak], None)
        self.assertEqual(r.oracle_results[0].status, "VIOLATION")

    def test_o6_fork_rng_restore(self):
        good = Observation(
            "p", calls=2, successful_calls=2,
            context_probe={"0": [{"rng_restored": True}], "1": [{"rng_restored": True}]},
            main_rng_restored=True,
        )
        r = evaluate_oracles(plan("O6"), [obs(), good], None)
        self.assertEqual(r.oracle_results[0].status, "PASS")
        bad = Observation(
            "p", calls=2, successful_calls=2,
            context_probe={"0": [{"rng_restored": False}], "1": [{"rng_restored": True}]},
            main_rng_restored=False,
        )
        r = evaluate_oracles(plan("O6"), [obs(), bad], None)
        self.assertEqual(r.oracle_results[0].status, "VIOLATION")

    def test_o6_grad_lifecycle(self):
        good = Observation(
            "p", calls=2, successful_calls=2,
            thread_roles=["grad_true", "grad_false"],
            context_probe={
                "0": [{"grad_before": True, "grad_inside": True, "grad_restored": True}],
                "1": [{"grad_before": True, "grad_inside": False, "grad_restored": True}],
            },
            main_grad_enabled=True,
        )
        r = evaluate_oracles(plan("O6"), [obs(), good], None)
        self.assertEqual(r.oracle_results[0].status, "PASS")
        bad = Observation(
            "p", calls=2, successful_calls=2,
            thread_roles=["grad_true", "grad_false"],
            context_probe={
                "0": [{"grad_before": True, "grad_inside": True, "grad_restored": False}],
                "1": [{"grad_before": True, "grad_inside": False, "grad_restored": True}],
            },
            main_grad_enabled=True,
        )
        r = evaluate_oracles(plan("O6"), [obs(), bad], None)
        self.assertEqual(r.oracle_results[0].status, "VIOLATION")

    def test_incomplete_observations_are_not_semantic_violations(self):
        b = obs(shared=tensor_digest(1.0))
        c = obs(shared=tensor_digest(2.0))
        c.execution_status = TIMEOUT
        c.completeness = "timeout"
        r = evaluate_oracles(plan("O2"), [b, c], None)
        self.assertEqual(r.oracle_results[0].status, "INCONCLUSIVE")

        b3 = obs()
        c3 = obs(digests={"0": [tensor_digest(9.0)]})
        c3.execution_status = CRASH
        c3.completeness = "crash"
        r = evaluate_oracles(plan("O3"), [b3, c3], None)
        self.assertEqual(r.oracle_results[0].status, "INCONCLUSIVE")

        c4 = obs(after={"args": [tensor_digest(9.0)]})
        c4.execution_status = ERROR
        c4.completeness = "error"
        r = evaluate_oracles(plan("O4"), [obs(), c4], None)
        self.assertEqual(r.oracle_results[0].status, "INCONCLUSIVE")

    def test_o7_unknown_and_supported(self):
        ok = Observation("p", calls=1, successful_calls=1,
                         recovery={"attempted": True, "success": True})
        r = evaluate_oracles(plan("O7", contract="UNKNOWN"), [obs(), ok], None)
        self.assertEqual(r.oracle_results[0].status, "INCONCLUSIVE")
        bad = Observation("p", calls=1, successful_calls=0,
                          exceptions=[{"thread": 0, "type": "RuntimeError", "message": "trace"}],
                          recovery={"attempted": True, "success": False, "type": "RuntimeError"})
        r = evaluate_oracles(plan("O7", contract="SUPPORTED"), [obs(), bad], None)
        self.assertEqual(r.oracle_results[0].status, "VIOLATION")


if __name__ == "__main__":
    unittest.main()
