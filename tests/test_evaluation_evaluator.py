from triton_evaluation.evaluators.base import EvaluatorResult
from triton_evaluation.evaluators.evaluation import EvaluationEvaluator
from triton_evaluation.models import JobSpec, TaskStatus


def _spec(checks, **overrides):
    values = dict(op_name="softmax", reference_code="R", kernel_code="K", checks=checks)
    values.update(overrides)
    return JobSpec(**values)


class _StubVerify:
    def __init__(self, mode):
        self.mode = mode

    def run(self, spec, verify_dir, device_id):
        if self.mode == "timeout":
            return EvaluatorResult(TaskStatus.TIMEOUT.value, error="verify timed out")
        if self.mode == "decoy":
            return EvaluatorResult(
                TaskStatus.SUCCEEDED.value,
                result={"decoy": True, "valid": False, "regression_type": 1},
            )
        passed, total = (2, 2) if self.mode == "pass" else (1, 2)
        failures = []
        if self.mode != "pass":
            failures = [
                {
                    "case_idx": 2,
                    "error_type": "AccuracyError",
                    "error_msg": "classified traceback",
                    "kind": "constant_offset",
                    "sample": [],
                }
            ]
        return EvaluatorResult(
            TaskStatus.SUCCEEDED.value,
            result={
                "op_name": spec.op_name,
                "total_cases": total,
                "passed_cases": passed,
                "failed_cases": total - passed,
                "failures": failures,
            },
        )


class _StubBench:
    def __init__(self, mode="success"):
        self.mode = mode
        self.called = False

    def run(self, spec, verify_dir, device_id):
        self.called = True
        if self.mode == "timeout":
            return EvaluatorResult(TaskStatus.TIMEOUT.value, error="benchmark timed out")
        result = {
            "speedup_vs_torch": 2.5 if self.mode == "success" else None,
            "failed_cases": 0,
            "per_shape_results": [
                {
                    "case_idx": 1,
                    "input_desc": [{"type": "tensor", "shape": [16], "dtype": "float32"}],
                    "status": "pass",
                    "implementation": {"avg_latency_ms": 0.4, "peak_memory_mb": 1.0},
                    "speedup_vs_torch": 2.5,
                }
            ],
        }
        return EvaluatorResult(TaskStatus.SUCCEEDED.value, result=result)


class _StubProfile:
    def __init__(self, status="SUCCEEDED", state="complete"):
        self.status = status
        self.state = state
        self.called = False

    def run(self, spec, verify_dir, device_id):
        self.called = True
        complete = self.state == "complete"
        result = {
            "profile_mode": "op",
            "selected_cases": 1,
            "completed_cases": 1 if complete else 0,
            "partial_cases": 0 if complete else 1,
            "failed_cases": 0,
            "timed_out": self.status == TaskStatus.TIMEOUT.value,
            "per_case_results": [
                {
                    "case_idx": 0,
                    "status": "complete" if complete else "partial",
                    "kernels": [{"name": "softmax_kernel", "duration_us": 400.0}],
                }
            ],
        }
        error = "profile timed out" if self.status == TaskStatus.TIMEOUT.value else None
        return EvaluatorResult(self.status, result=result, error=error)


class _Harness(EvaluationEvaluator):
    def __init__(
        self,
        verify_mode="pass",
        bench_mode="success",
        profile_status="SUCCEEDED",
        profile_state="complete",
    ):
        super().__init__("unused")
        self.verify = _StubVerify(verify_mode)
        self.bench = _StubBench(bench_mode)
        self.profile = _StubProfile(profile_status, profile_state)

    def _verify_evaluator(self):
        return self.verify

    def _benchmark_evaluator(self):
        return self.bench

    def _profile_evaluator(self):
        return self.profile


def test_correctness_only_returns_typed_payload(tmp_path):
    result = _Harness().run(_spec(["correctness"]), str(tmp_path), device_id=0)

    assert result.status == TaskStatus.SUCCEEDED.value
    assert result.result["correctness"]["passed"] is True
    assert result.result["performance"] is None
    assert result.result["profile"] is None


def test_wrong_kernel_skips_performance_but_keeps_profile(tmp_path):
    harness = _Harness(verify_mode="fail")
    result = harness.run(
        _spec(["correctness", "performance", "profile"]),
        str(tmp_path),
        device_id=0,
    )

    assert result.status == TaskStatus.SUCCEEDED.value
    assert result.result["correctness"]["passed"] is False
    assert result.result["performance"] is None
    assert result.result["profile"]["profile_state"] == "complete"
    assert harness.bench.called is False


def test_performance_runs_when_correctness_passes(tmp_path):
    result = _Harness().run(
        _spec(["correctness", "performance"]),
        str(tmp_path),
        device_id=0,
    )

    assert result.status == TaskStatus.SUCCEEDED.value
    assert result.result["performance"]["speedup_vs_torch"] == 2.5
    assert result.result["performance"]["per_shape"][0]["shape"] == "[16]"


def test_decoy_still_returns_profile(tmp_path):
    result = _Harness(verify_mode="decoy").run(
        _spec(["correctness", "performance", "profile"]),
        str(tmp_path),
        device_id=0,
    )

    assert result.result["correctness"]["decoy"] is True
    assert result.result["performance"] is None
    assert result.result["profile"]["profile_state"] == "complete"


def test_profile_timeout_returns_timeout_with_partial_profile(tmp_path):
    result = _Harness(profile_status="TIMEOUT", profile_state="partial").run(
        _spec(["profile"]),
        str(tmp_path),
        device_id=0,
    )

    assert result.status == TaskStatus.TIMEOUT.value
    assert result.error == "profile timed out"
    assert result.result["profile"]["profile_state"] == "partial"


def test_verify_timeout_still_runs_independent_profile(tmp_path):
    result = _Harness(verify_mode="timeout").run(
        _spec(["correctness", "profile"]),
        str(tmp_path),
        device_id=0,
    )

    assert result.status == TaskStatus.TIMEOUT.value
    assert result.result["correctness"] is None
    assert result.result["profile"]["profile_state"] == "complete"


def test_malformed_benchmark_fails_but_retains_other_results(tmp_path):
    result = _Harness(bench_mode="malformed").run(
        _spec(["correctness", "performance", "profile"]),
        str(tmp_path),
        device_id=0,
    )

    assert result.status == TaskStatus.FAILED.value
    assert "benchmark result is incomplete" in result.error
    assert result.result["correctness"]["passed"] is True
    assert result.result["performance"] is None
    assert result.result["profile"] is not None


class _Raises:
    def __init__(self, message):
        self.message = message
        self.called = False

    def run(self, *args):
        self.called = True
        raise RuntimeError(self.message)


class _MissingResult:
    def run(self, *args):
        return EvaluatorResult(TaskStatus.SUCCEEDED.value, result=None)


def test_verify_exception_becomes_failure_and_profile_still_runs(tmp_path):
    harness = _Harness()
    harness.verify = _Raises("verify infrastructure broke")

    result = harness.run(
        _spec(["correctness", "performance", "profile"]),
        str(tmp_path),
        device_id=0,
    )

    assert result.status == TaskStatus.FAILED.value
    assert "verify infrastructure broke" in result.error
    assert result.result["correctness"] is None
    assert result.result["performance"] is None
    assert result.result["profile"]["profile_state"] == "complete"
    assert harness.bench.called is False


def test_benchmark_exception_retains_correctness_and_profile(tmp_path):
    harness = _Harness()
    harness.bench = _Raises("benchmark infrastructure broke")

    result = harness.run(
        _spec(["correctness", "performance", "profile"]),
        str(tmp_path),
        device_id=0,
    )

    assert result.status == TaskStatus.FAILED.value
    assert "benchmark infrastructure broke" in result.error
    assert result.result["correctness"]["passed"] is True
    assert result.result["performance"] is None
    assert result.result["profile"]["profile_state"] == "complete"


def test_profile_exception_retains_correctness(tmp_path):
    harness = _Harness()
    harness.profile = _Raises("profile infrastructure broke")

    result = harness.run(
        _spec(["correctness", "profile"]),
        str(tmp_path),
        device_id=0,
    )

    assert result.status == TaskStatus.FAILED.value
    assert "profile infrastructure broke" in result.error
    assert result.result["correctness"]["passed"] is True
    assert result.result["profile"] is None


def test_successful_profile_without_result_is_failed(tmp_path):
    harness = _Harness()
    harness.profile = _MissingResult()

    result = harness.run(
        _spec(["correctness", "profile"]),
        str(tmp_path),
        device_id=0,
    )

    assert result.status == TaskStatus.FAILED.value
    assert "profile result is missing" in result.error
    assert result.result["correctness"]["passed"] is True
    assert result.result["profile"] is None


def test_malformed_correctness_result_still_runs_profile(tmp_path):
    class _MalformedCorrectness:
        def run(self, *args):
            return EvaluatorResult(TaskStatus.SUCCEEDED.value, result={})

    harness = _Harness()
    harness.verify = _MalformedCorrectness()

    result = harness.run(
        _spec(["correctness", "profile"]),
        str(tmp_path),
        device_id=0,
    )

    assert result.status == TaskStatus.FAILED.value
    assert "correctness result is invalid" in result.error
    assert result.result["correctness"] is None
    assert result.result["profile"]["profile_state"] == "complete"
