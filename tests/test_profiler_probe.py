"""Controlled feedback-loop checks; these never establish NPU profiler health."""

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/probe_profiler_latency.py"
spec = importlib.util.spec_from_file_location("profiler_probe_test", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError("Cannot import tracked feedback loop")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.mark.parametrize("value", [None, True, False, 0, -1, float("nan"), float("inf"), "1"])
def test_latency_none_and_nonfinite_are_sharp_failures(value):
    assert not probe.valid_latency(value)


def test_accepts_actual_numeric_profiler_scalar():
    import numpy as np

    assert probe.valid_latency(np.float64(0.001))


def test_numpy_profiler_latency_produces_native_boolean_and_json_report(tmp_path):
    import numpy as np

    latency = np.float64(0.00471)
    accepted = probe.valid_latency(latency)
    assert type(accepted) is bool
    output = tmp_path / "report.json"
    probe.write_report(output, {"samples": [{"latency_ms": float(latency), "passed": accepted}]})
    saved = json.loads(output.read_bytes())
    assert saved["samples"][0]["passed"] is True


@pytest.mark.parametrize("kind", ["nan", "inf", "zero", "negative", "bool"])
def test_numpy_invalid_latencies_remain_native_false(kind):
    import numpy as np

    value = {
        "nan": np.float64("nan"),
        "inf": np.float64("inf"),
        "zero": np.float64(0),
        "negative": np.float64(-1),
        "bool": np.bool_(True),
    }[kind]
    assert probe.valid_latency(value) is False


def test_report_rejects_nonfinite_and_unknown_values(tmp_path):
    output = tmp_path / "invalid-report.json"
    with pytest.raises(ValueError, match="JSON compliant"):
        probe.write_report(output, {"latency_ms": float("nan")})
    with pytest.raises(TypeError, match="not JSON serializable"):
        probe.write_report(output, {"unknown": object()})
    assert not output.exists()


def test_eight_actual_numpy_timings_are_written_without_a_scalar_serializer(
    tmp_path, monkeypatch, controlled_measurement
):
    import numpy as np

    monkeypatch.setattr(
        controlled_measurement,
        "measure_single",
        lambda *_args, **_kwargs: (
            {"aclnnMatmul": np.float64(4.94)},
            np.float64(0.00471),
            np.float64(1.0),
        ),
    )
    assert probe.probe(tmp_path / "numpy-timings", 8) == 0
    saved = json.loads((tmp_path / "numpy-timings/report.json").read_bytes())
    assert saved["state"] == "passed" and len(saved["samples"]) == 8
    assert all(row["passed"] is True for row in saved["samples"])


def test_wrong_mask_cannot_create_device_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,8")
    output = tmp_path / "new"
    with pytest.raises(ValueError, match="exactly"):
        probe.probe(output, 8)
    assert not output.exists()


def test_previous_attempt_is_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "8")
    previous = tmp_path / "report.json"
    previous.write_text("old failed attempt\n")
    with pytest.raises(FileExistsError):
        probe.probe(tmp_path, 8)
    assert previous.read_text() == "old failed attempt\n"


@pytest.fixture
def controlled_measurement(monkeypatch):
    from triton_evaluation.engine import benchmark

    class Tensor:
        device = "npu:0"
        shape = [256, 256]
        dtype = "torch.float32"

        def __matmul__(self, _other):
            return self

    class Model:
        def eval(self):
            return self

        def __call__(self, *args):
            return self.forward(*args)

    class NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    npu = SimpleNamespace(
        set_device=lambda *_: None,
        current_device=lambda: 0,
        device_count=lambda: 1,
        get_device_name=lambda *_: "controlled",
        manual_seed=lambda *_: None,
        synchronize=lambda: None,
    )
    fake_torch = SimpleNamespace(
        __version__="controlled",
        npu=npu,
        nn=SimpleNamespace(Module=Model),
        set_default_device=lambda *_: None,
        manual_seed=lambda *_: None,
        get_default_device=lambda: "npu:0",
        randn=lambda *_: Tensor(),
        device=lambda value: value,
        no_grad=NoGrad,
        isfinite=lambda *_: SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: True)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(__version__="controlled"))
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "8")
    return benchmark


def test_real_latency_none_symptom_is_retained_and_exits_one(
    tmp_path, monkeypatch, controlled_measurement
):
    benchmark = controlled_measurement
    original_cleanup = benchmark.cleanup_profile_path
    calls = []

    def fail(ctx, **kwargs):
        calls.append((ctx, kwargs))
        artifact = Path.cwd() / "sample-01_raw"
        artifact.mkdir()
        (artifact / "raw.txt").write_text("controlled failed capture")
        benchmark.cleanup_profile_path(str(artifact))
        return None, None, 1.0

    monkeypatch.setattr(benchmark, "measure_single", fail)
    output = tmp_path / "red"
    assert probe.probe(output, 8) == 1
    report = json.loads((output / "report.json").read_bytes())
    assert report["state"] == "failed"
    assert report["samples"][0]["raw_latency"] == "None"
    assert "finite and positive" in report["error"]
    assert len(calls) == 1  # Stop on the actual symptom, do not repeat eight failures.
    assert (output / "sample-01_raw/raw.txt").is_file()
    assert benchmark.cleanup_profile_path is original_cleanup


def test_green_requires_all_eight_original_profiler_contexts_and_median(
    tmp_path, monkeypatch, controlled_measurement
):
    benchmark = controlled_measurement
    calls = []

    def measure(ctx, **kwargs):
        assert (ctx.method, ctx.warmup, ctx.repeats) == ("profiler", 5, 20)
        assert kwargs == {"clear_l2_cache": False, "keep_res": True}
        calls.append(ctx)
        return {"matrix_product": 1.0}, len(calls) * 0.01, 1.0

    monkeypatch.setattr(benchmark, "measure_single", measure)
    output = tmp_path / "green"
    assert probe.probe(output, 8) == 0
    report = json.loads((output / "report.json").read_bytes())
    assert len(calls) == len(report["samples"]) == 8
    assert report["median_ms"] == pytest.approx(0.045)
    assert report["is_independent_sample"] is False
    assert report["admission_evidence"] is False


@pytest.mark.parametrize("trace_kind", ["valid", "missing", "wrong-consumer", "wrong-cann-env"])
def test_configured_probe_requires_actual_pinned_callback_not_only_environment(
    tmp_path, monkeypatch, controlled_measurement, trace_kind
):
    from triton_evaluation import profiler_runtime

    expected = {
        "enabled": True,
        "manifest_path": str(tmp_path / "runtime/manifest.json"),
        "manifest_sha256": "a" * 64,
        "base_ld_sha256": "b" * 64,
        "cann_runtime": {"cann_set_env_sha256": "e" * 64, "effective_environment_sha256": "f" * 64},
    }
    for name in ("alias", "library", "consumer", "msprof"):
        expected[name + "_path"] = str(tmp_path / "runtime" / name)
        expected[name + "_sha256"] = "c" * 64
    monkeypatch.setattr(profiler_runtime, "describe_runtime", lambda: expected)
    calls = []

    def measure(ctx, **_kwargs):
        calls.append(ctx)
        if trace_kind != "missing":
            directory = Path.cwd() / f"{ctx.profile_name}_controlled"
            directory.mkdir()
            trace = {
                **expected,
                "callback_scope_completed": True,
                "base_ld_restored": True,
                "main_thread": True,
                "process_id": os.getpid(),
                "helper_sha256": hashlib.sha256(
                    (probe.ROOT / "triton_evaluation/profiler_runtime.py").read_bytes()
                ).hexdigest(),
                "helper_path": str(probe.ROOT / "triton_evaluation/profiler_runtime.py"),
                "service_package_path": str(probe.ROOT / "triton_evaluation"),
                "physical_device": "8",
                "visible_devices": "8",
                "reference_device_policy": os.environ.get(
                    "TRITON_EVAL_REFERENCE_DEVICE_POLICY", "cpu-first"
                ),
                "effective_ld_prefix": str(Path(expected["alias_path"]).parent),
                "previous_ld_sha256": expected["base_ld_sha256"],
            }
            if trace_kind == "wrong-consumer":
                trace["consumer_sha256"] = "d" * 64
            if trace_kind == "wrong-cann-env":
                trace["cann_runtime"] = {
                    **expected["cann_runtime"],
                    "effective_environment_sha256": "0" * 64,
                }
            (directory / "profiler-runtime.json").write_text(json.dumps(trace))
        return {"matrix_product": 20.0}, 0.02, 1.0

    monkeypatch.setattr(controlled_measurement, "measure_single", measure)
    output = tmp_path / trace_kind
    assert probe.probe(output, 8) == (0 if trace_kind == "valid" else 1)
    report = json.loads((output / "report.json").read_bytes())
    assert len(report["samples"]) == (8 if trace_kind == "valid" else 1)
    # A completed measurement is retained even when proof invalidates the run.
    assert report["samples"][0]["latency_ms"] == 0.02
    assert report["samples"][0]["runtime_evidence_valid"] is (trace_kind == "valid")
    assert report["state"] == ("passed" if trace_kind == "valid" else "failed")


@pytest.mark.parametrize("valid", [True, False])
def test_isolated_wrapper_uses_actual_boundary_and_rejects_missing_measurements(
    tmp_path, monkeypatch, valid
):
    from triton_evaluation.evaluators import base, sandbox

    for key in (
        sandbox.CONFIG_ENV,
        sandbox.DIGEST_ENV,
        sandbox.OVERLAY_CONFIG_ENV,
        sandbox.OVERLAY_DIGEST_ENV,
    ):
        monkeypatch.setenv(key, "controlled-not-an-actual-device-pin")
    calls = []

    def execute(command, cwd, device_id, *, budget_seconds):
        calls.append((command, cwd, device_id, budget_seconds))
        native = cwd / "native"
        native.mkdir()
        (native / "report.json").write_text(
            json.dumps(
                {
                    "state": "passed",
                    "physical_device": 8,
                    "process_id": 2,
                    "profiler_runtime": {"enabled": True},
                    "median_ms": 0.02,
                    "samples": [
                        {
                            "passed": True,
                            "runtime_evidence_valid": True,
                            "latency_ms": 0.02 if valid else None,
                        }
                    ]
                    * 8,
                    "native_tensor_proof": {
                        "input_device": "npu:0",
                        "output_device": "npu:0",
                        "finite": True,
                    },
                }
            )
        )
        return base.ScriptOutcome(0, "controlled stdout", "", False)

    monkeypatch.setattr(base, "run_script", execute)
    output = tmp_path / "wrapper"
    assert probe.isolated_probe(output, 8) == (0 if valid else 1)
    assert calls[0][2:] == (8, 180)
    assert calls[0][0][1] == str(SCRIPT)
    observed = json.loads((output / "sandbox-probe.json").read_bytes())
    assert observed["admission_evidence"] is observed["is_independent_sample"] is False
    assert (
        observed["native_report_sha256"]
        == hashlib.sha256((output / "native/report.json").read_bytes()).hexdigest()
    )


def test_cli_sandbox_option_selects_boundary_callback(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(
        sys, "argv", [str(SCRIPT), "--output", str(tmp_path), "--device", "8", "--sandbox"]
    )
    monkeypatch.setattr(probe, "isolated_probe", lambda *args: called.append(args) or 7)
    assert probe.main() == 7
    assert called == [(tmp_path, 8)]
