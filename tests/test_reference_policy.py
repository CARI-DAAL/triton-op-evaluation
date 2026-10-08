"""Fixed reference selection must never depend on candidate acceptance."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from conftest import make_request

from triton_evaluation.config import Settings
from triton_evaluation.engine import verify
from triton_evaluation.evaluators.base import ScriptOutcome
from triton_evaluation.evaluators.evaluation import EvaluationEvaluator


@pytest.mark.parametrize(
    "error",
    [None, verify.AccuracyError("mismatch", {}), RuntimeError("reference unavailable")],
)
def test_strict_npu_never_tries_cpu_or_retries_after_a_verdict(monkeypatch, error):
    devices = []

    def attempt(_modules, _context, _inputs, _device, golden_device, _non_compute):
        devices.append(golden_device)
        if error is not None:
            raise error

    monkeypatch.setattr(verify, "_attempt_case", attempt)
    monkeypatch.setattr(verify, "describe_input", lambda _: [])
    passed, _failure = verify._try_run_case(
        {}, verify.CaseContext(1, 1), [], "npu:0", reference_device_policy="npu-strict"
    )
    assert devices == ["npu:0"]
    assert passed is (error is None)


def test_default_cpu_first_only_retries_reference_execution_error(monkeypatch):
    devices = []

    def attempt(_modules, _context, _inputs, _device, golden_device, _non_compute):
        devices.append(golden_device)
        if golden_device == "cpu":
            raise RuntimeError("CPU operation unavailable")

    monkeypatch.setattr(verify, "_attempt_case", attempt)
    assert verify._try_run_case({}, verify.CaseContext(1, 1), [], "npu:0") == (True, None)
    assert devices == ["cpu", "npu:0"]


def test_settings_reaches_engine_command(tmp_path, monkeypatch):
    import triton_evaluation.evaluators.verify as evaluator_module

    seen = []

    def run(cmd, *_args, **_kwargs):
        seen.extend(cmd)
        return ScriptOutcome(0, "", "", False, "exited")

    monkeypatch.setattr(evaluator_module, "run_script", run)
    monkeypatch.setattr(evaluator_module, "read_json", lambda _, **_kwargs: {})
    settings = Settings(reference_device_policy="npu-strict")
    evaluator = EvaluationEvaluator("unused", settings)._verify_evaluator()
    evaluator._run_verify(
        make_request(),
        tmp_path,
        3,
        datetime.now(timezone.utc) + timedelta(seconds=120),
        lambda: True,
    )
    assert seen[seen.index("--reference-device-policy") + 1] == "npu-strict"


def test_unknown_reference_policy_is_rejected():
    with pytest.raises(ValueError, match="reference_device_policy"):
        Settings(reference_device_policy="switch-until-passing")


def test_policy_binds_admission_and_worker_compatibility(
    settings,
    store,
    release_catalog,
    clock,
):
    from conftest import make_caller

    from triton_evaluation.instance import EvaluationInstance
    from triton_evaluation.worker import Worker

    strict = settings.model_copy(update={"reference_device_policy": "npu-strict"})
    old_worker = Worker(store, None, device_id=3, settings=settings, clock=clock)
    new_worker = Worker(store, None, device_id=3, settings=strict, clock=clock)
    store.heartbeat(new_worker.identity)
    instance = EvaluationInstance(
        settings=strict,
        store=store,
        catalog=release_catalog,
    )
    created = instance.submit(
        make_request(),
        make_caller(),
        session_id="strict",
        idempotency_key="strict",
    )
    assert store.claim_next(old_worker.identity, card=3) is None
    claimed = store.claim_next(new_worker.identity, card=3)
    assert claimed.evaluation_id == created.resource.evaluation_id
    assert claimed.pinned.algorithms.correctness == "correctness-v1:npu-strict-v1"
    assert settings.effective_correctness_algorithm_version == "correctness-v1"


def test_policy_env_is_read_and_effective_version_is_bounded(monkeypatch):
    monkeypatch.setenv("TRITON_EVAL_REFERENCE_DEVICE_POLICY", "npu-strict")
    assert Settings.from_env().effective_correctness_algorithm_version == (
        "correctness-v1:npu-strict-v1"
    )
    with pytest.raises(ValueError, match="exceeds 128"):
        Settings(correctness_algorithm_version="a" * 128)


@pytest.mark.parametrize("fails", [False, True])
def test_actual_reference_output_device_is_distinct_from_selection(monkeypatch, fails):
    import contextlib
    import sys

    class Tensor:
        def __init__(self, device):
            self.device = device

    torch = SimpleNamespace(Tensor=Tensor, no_grad=contextlib.nullcontext)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(verify, "_infer_input_type", lambda _: ("float", None))
    monkeypatch.setattr(verify, "_instantiate_models", lambda *_a, **_kw: (object(), object()))
    monkeypatch.setattr(verify, "cleanup_npu_memory", lambda: None)
    monkeypatch.setattr(verify, "_run_candidate_forward", lambda *_a: 1)

    def reference(*_args):
        if fails:
            raise RuntimeError("reference unavailable")
        return Tensor("npu:0")

    monkeypatch.setattr(verify, "_invoke_model_shared", reference)
    trace = []
    context = verify.CaseContext(1, 1, reference_trace=trace)
    modules = dict(framework_cls=None, impl_cls=None, get_init_inputs=None)
    if fails:
        with pytest.raises(RuntimeError, match="reference unavailable"):
            verify._attempt_case(modules, context, [], "npu", "npu", False)
    else:
        verify._attempt_case(modules, context, [], "npu", "npu", False)
    assert trace == [
        {
            "case_idx": 1,
            "selected_reference_device": "npu",
            "reference_completed": not fails,
            "output_devices": [] if fails else [{"output_index": 0, "device": "npu:0"}],
            "reference_source": "fresh",
        }
    ]


def test_watchdog_forwards_fixed_policy_to_engine_subprocess(tmp_path, monkeypatch):
    import runpy
    import sys

    seen = []

    class Process:
        returncode = 0

        def __init__(self, cmd, **_kwargs):
            seen.extend(cmd)

        def communicate(self, timeout):
            return b"", b""

    monkeypatch.setattr("subprocess.Popen", Process)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            verify.__file__,
            "--op_name",
            "fixed",
            "--verify_dir",
            str(tmp_path),
            "--reference-device-policy",
            "npu-strict",
        ],
    )
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(verify.__file__, run_name="__main__")
    assert stopped.value.code == 0
    assert seen[seen.index("--reference-device-policy") + 1] == "npu-strict"
    assert "--subprocess" in seen
