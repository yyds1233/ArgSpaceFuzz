import unittest

from concurrent_oracle.oracle_engine import evaluate_oracles
from concurrent_oracle.schemas import (
    COMPLETED, CRASH, ConcurrentTestPlan, Observation, OracleCheckResult,
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


def plan(oracle, contract="SUPPORTED"):
    return ConcurrentTestPlan(
        "p", "id", "test", execution={}, oracles=[oracle],
        contract_status=contract, target_api="torch.example",
    )


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
        r = evaluate_oracles(plan("O1"), [obs(exceptions=exc), obs(exceptions=exc)], None)
        self.assertEqual(r.oracle_results[0].status, "PASS")
        r = evaluate_oracles(plan("O1"), [obs(status=CRASH), obs(status=CRASH)], None)
        self.assertEqual(r.oracle_results[0].status, "VIOLATION")

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
