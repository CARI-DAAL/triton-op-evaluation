from triton_evaluation.evaluators.base import EvaluatorResult
from triton_evaluation.evaluators.evaluation import EvaluationEvaluator
from triton_evaluation.models import JobSpec, TaskStatus


def _spec(checks, **kw):
    base = dict(op_name="softmax", reference_code="R", kernel_code="K", checks=checks)
    base.update(kw)
    return JobSpec(**base)


class _StubVerify:
    """Returns a canned correctness result. mode: pass | fail | decoy | timeout."""
    def __init__(self, mode):
        self.mode = mode

    def run(self, spec, verify_dir, device_id):
        if self.mode == "timeout":
            return EvaluatorResult(TaskStatus.TIMEOUT.value, error="verify timed out")
        if self.mode == "decoy":
            return EvaluatorResult(TaskStatus.SUCCEEDED.value,
                                   result={"decoy": True, "valid": False, "regression_type": 1})
        passed, total = (2, 2) if self.mode == "pass" else (1, 2)
        failures = [] if self.mode == "pass" else [
            {"case_idx": 2, "kind": "constant_offset", "sample": []}
        ]
        return EvaluatorResult(
            TaskStatus.SUCCEEDED.value,
            result={"op_name": spec.op_name, "total_cases": total, "passed_cases": passed,
                    "failed_cases": total - passed, "failures": failures},
        )


class _StubBench:
    def __init__(self, speedup=2.5):
        self.speedup = speedup
        self.called = False

    def run(self, spec, verify_dir, device_id):
        self.called = True
        return EvaluatorResult(TaskStatus.SUCCEEDED.value,
                               result={"speedup_vs_torch": self.speedup, "per_shape_results": []})


class _StubProfile:
    def __init__(self, state="complete"):
        self.state = state
        self.called = False

    def run(self, spec, verify_dir, device_id):
        self.called = True
        return EvaluatorResult(TaskStatus.SUCCEEDED.value,
                               result={"profile_mode": "op", "selected_cases": 2,
                                       "completed_cases": 2 if self.state == "complete" else 1,
                                       "failed_cases": 0, "partial_cases": 0})


class _Harness(EvaluationEvaluator):
    def __init__(self, verify_mode="pass", speedup=2.5, profile_state="complete"):
        super().__init__("unused")
        self._verify = _StubVerify(verify_mode)
        self._bench = _StubBench(speedup)
        self._profile = _StubProfile(profile_state)

    def _verify_evaluator(self): return self._verify
    def _benchmark_evaluator(self): return self._bench
    def _profile_evaluator(self): return self._profile


def test_correctness_only_returns_payload_with_correctness(tmp_path):
    h = _Harness()
    res = h.run(_spec(["correctness"]), str(tmp_path), device_id=0)
    assert res.status == TaskStatus.SUCCEEDED.value
    assert res.result["correctness"]["passed"] is True
    assert res.result["performance"] is None
    assert res.result["profile"] is None
    assert h._bench.called is False  # performance not requested -> not run


def test_performance_runs_only_when_correctness_passes(tmp_path):
    h = _Harness(verify_mode="fail")
    res = h.run(_spec(["correctness", "performance"], warmup=2, repeats=5),
                str(tmp_path), device_id=0)
    assert res.status == TaskStatus.SUCCEEDED.value
    assert res.result["correctness"]["passed"] is False
    assert res.result["performance"] is None  # combination rule: correctness failed
    assert h._bench.called is False


def test_performance_runs_when_correctness_passes(tmp_path):
    h = _Harness(verify_mode="pass")
    res = h.run(_spec(["correctness", "performance"], warmup=2, repeats=5),
                str(tmp_path), device_id=0)
    assert res.result["performance"]["speedup_vs_torch"] == 2.5
    assert h._bench.called is True


def test_profile_runs_independently_of_correctness(tmp_path):
    h = _Harness(verify_mode="fail")
    res = h.run(_spec(["correctness", "profile"], profile_mode="op"), str(tmp_path), device_id=1)
    assert res.result["profile"]["profile_state"] in ("complete", "partial")
    assert h._profile.called is True  # profile runs even though correctness failed


def test_profile_only_has_no_correctness(tmp_path):
    res = _Harness().run(_spec(["profile"], profile_mode="op"), str(tmp_path), device_id=0)
    assert res.result["correctness"] is None
    assert res.result["profile"]["profile_state"] == "complete"


def test_verify_timeout_propagates(tmp_path):
    res = _Harness(verify_mode="timeout").run(_spec(["correctness"]), str(tmp_path), device_id=0)
    assert res.status == TaskStatus.TIMEOUT.value
    assert res.result is None or res.result == {}


def test_decoy_short_circuits_to_decoy_correctness(tmp_path):
    h = _Harness(verify_mode="decoy")
    res = h.run(_spec(["correctness", "performance"], warmup=2, repeats=5),
                str(tmp_path), device_id=0)
    assert res.status == TaskStatus.SUCCEEDED.value
    assert res.result["correctness"]["decoy"] is True
    assert res.result["performance"] is None  # no benchmark on a decoy
    assert h._bench.called is False
