"""Real engine entry points retain completed timing when their process is killed."""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from triton_evaluation.engine import benchmark
from triton_evaluation.engine import joint_measurement as joint
from triton_evaluation.engine import single_benchmark as single

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("comparison", "single", "joint-comparison", "joint-single")


def _controlled_engine(variant, output, marker, outcome):
    """Replace device measurements, retaining the CLI, gate and case loops."""
    output, marker = Path(output), Path(marker)

    class Model:
        def __init__(self, *_):
            pass

        def to(self, _):
            return self

    torch = SimpleNamespace(
        Tensor=type("Tensor", (), {}), device=lambda _: "cpu", manual_seed=lambda _: None,
        npu=SimpleNamespace(manual_seed=lambda _: None),
    )
    sys.modules["torch"] = torch
    sys.modules["torch_npu"] = SimpleNamespace()
    groups = [[], [], [], []]
    reference = SimpleNamespace(Model=Model, get_init_inputs=lambda: [],
                                get_input_groups=lambda: groups)
    benchmark._enforce_conv_fp32 = lambda: None
    benchmark._ensure_npu_ambient_shared = lambda *_a, **_k: True
    benchmark.cleanup_npu_memory = lambda: None
    benchmark.resolve_inputs = lambda *_: groups
    benchmark._load_benchmark_modules = lambda *_: (Model, Model, lambda: [])
    benchmark.describe_input = lambda _: []
    single.load_module = lambda name, _: (
        reference if name.endswith("_torch") else SimpleNamespace(ModelNew=Model)
    )
    joint._prepare = lambda *_: (reference, Model, joint.CleanInputGroups(groups), "cpu")
    joint._rng_state = lambda: (None, None)
    joint._restore_rng = lambda _: None

    def before_measure(index):
        if index != 3:
            return
        if outcome == "interrupted":
            marker.write_text("next case entered", encoding="utf-8")
            threading.Event().wait()
        if outcome == "failed":
            raise RuntimeError("controlled measurement failure")

    def compare(_models, _inputs, _config, _device, context):
        before_measure(context.case_idx - 1)
        return (
            benchmark.PerformanceResult(4.0, 2.0, {}),
            benchmark.PerformanceResult(1.0, 1.0, {}, kernel_metrics={
                "kernel": {"latency_ms": 1.0, "aic_cube_ratio": .4},
            }),
            4.0,
        )

    def measure(context):
        if context.profile_name.startswith("single_"):
            before_measure(int(context.profile_name.rsplit("_", 1)[-1]))
        return benchmark.Measurement(
            {"kernel": 1000.0}, 1.0, 1.0,
            {"kernel": {"latency_ms": 1.0, "aic_cube_ratio": .4}},
        )

    benchmark.run_single_benchmark = compare
    benchmark.measure_single = measure
    argv = [
        "engine", "--op_name", "task", "--verify_dir", str(output.parent),
        "--output", str(output), "--case-indices", "2,4", "--warmup", "1",
        "--repeats", "2", "--measurement-method", "event",
    ]
    if variant == "single":
        sys.argv = [*argv, "--implementation", "triton",
                    "--triton_impl_name", "triton_ascend_impl"]
        single.main()
    elif variant.startswith("joint-"):
        sys.argv = [*argv, "--disable-reference-cache", "--max_retries", "1",
                    "--implementation", "triton" if variant == "joint-single" else "comparison",
                    "--trace-output", str(output.parent / "trace.json"),
                    "--trace-case-indices", "2,4"]
        raise SystemExit(joint.main())
    else:
        sys.argv = [*argv, "--disable-reference-cache", "--max_retries", "1"]
        benchmark.main()


def _child_command(variant, output, marker, outcome):
    return [
        sys.executable, "-c",
        "import runpy; runpy.run_path('tests/test_benchmark_checkpoints.py', run_name='__main__')",
        variant, str(output), str(marker), outcome,
    ]


def _gate(root, passed=True):
    (root / "verify_result.json").write_text(
        json.dumps({"total_cases": 4, "passed_cases": 4 if passed else 0, "failures": []}),
        encoding="utf-8",
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_killed_next_case_retains_prior_numbers_and_full_sparse_selection(tmp_path, variant):
    _gate(tmp_path)
    output, marker = tmp_path / "performance.json", tmp_path / "in_next_case"
    process = subprocess.Popen(
        _child_command(variant, output, marker, "interrupted"), cwd=ROOT,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 15
        while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(.02)
        assert marker.exists(), "engine did not enter the controlled next-case measurement"
        saved = output.read_bytes()
        raw = json.loads(saved)
        assert raw["progress"] == {
            "selected_case_indices": [1, 3], "completed_case_indices": [1], "complete": False,
        }
        if "comparison" in variant:
            assert raw["total_cases"] == 2
            assert raw["passed_cases"] == 1 and raw["failed_cases"] == 0
            assert raw["case_indices"] == [2, 4]
            assert len(raw["per_shape_results"]) == 1
            row = raw["per_shape_results"][0]
            assert row["case_idx"] == 2 and row["speedup_vs_torch"] == 4
            assert row["implementation"]["avg_latency_ms"] == 1
            assert row["implementation"]["kernel_metrics"]["kernel"]["aic_cube_ratio"] == .4
        else:
            assert raw["correctness_verified"] is False
            assert raw["selected_cases"] == 2 and raw["measured_cases"] == 1
            assert raw["coverage"] == .5 and raw["unmeasured_cases"] == []
            assert len(raw["per_case"]) == 1
            assert raw["per_case"][0]["case_idx"] == 1
            assert raw["per_case"][0]["latency_ms"] == 1
        process.kill()
        process.communicate(timeout=10)
        assert output.read_bytes() == saved
        assert not (tmp_path / "trace.json").exists()
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=10)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("outcome", ["complete", "failed"])
def test_final_checkpoint_marks_executed_failures_complete_without_pending_rows(
    tmp_path, variant, outcome,
):
    _gate(tmp_path)
    output, marker = tmp_path / "performance.json", tmp_path / "unused"
    result = subprocess.run(
        _child_command(variant, output, marker, outcome), cwd=ROOT, capture_output=True,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"}, timeout=15,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    raw = json.loads(output.read_bytes())
    assert raw["progress"] == {
        "selected_case_indices": [1, 3], "completed_case_indices": [1, 3], "complete": True,
    }
    if "comparison" in variant:
        assert [row["case_idx"] for row in raw["per_shape_results"]] == [2, 4]
        assert raw["failed_cases"] == (1 if outcome == "failed" else 0)
    else:
        assert raw["measured_cases"] == (1 if outcome == "failed" else 2)
        assert [row["case_idx"] for row in raw["unmeasured_cases"]] == (
            [3] if outcome == "failed" else []
        )


@pytest.mark.parametrize("variant", ["comparison", "joint-comparison"])
def test_correctness_gate_failure_never_creates_performance_checkpoint(tmp_path, variant):
    _gate(tmp_path, passed=False)
    output, marker = tmp_path / "performance.json", tmp_path / "unused"
    result = subprocess.run(
        _child_command(variant, output, marker, "complete"), cwd=ROOT, capture_output=True,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"}, timeout=15,
    )
    assert result.returncode == benchmark.VERIFY_GATE_EXIT_CODE
    assert not output.exists() and not marker.exists()


@pytest.mark.parametrize("failure", ["serialize", "replace"])
def test_atomic_write_failure_preserves_previous_complete_checkpoint(
    tmp_path, monkeypatch, failure,
):
    checkpoint = sys.modules[benchmark.write_checkpoint.__module__]
    output = tmp_path / "performance.json"
    benchmark.write_checkpoint(output, {"measured": [1]})
    saved = output.read_bytes()

    def interrupted_dump(_result, stream, **_kwargs):
        stream.write('{"measured": [')
        raise OSError("interrupted checkpoint write")

    def interrupted_replace(*_):
        raise OSError("interrupted checkpoint replace")

    if failure == "serialize":
        monkeypatch.setattr(checkpoint.json, "dump", interrupted_dump)
    else:
        monkeypatch.setattr(checkpoint.os, "replace", interrupted_replace)
    with pytest.raises(OSError, match="interrupted checkpoint"):
        benchmark.write_checkpoint(output, {"measured": [1, 2]})
    assert output.read_bytes() == saved
    assert list(tmp_path.iterdir()) == [output]


if __name__ == "__main__":
    _controlled_engine(*sys.argv[1:])
