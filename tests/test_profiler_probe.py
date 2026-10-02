"""Controlled feedback-loop checks; these never establish NPU profiler health."""

import importlib.util
import json
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
