"""Native Torch and Triton profiling select real implementations without inventing a counterpart."""

import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from triton_evaluation.engine import msprof_deep, msprof_op, profile_runner


@pytest.mark.parametrize("implementation", ["torch", "triton", "comparison"])
def test_runner_executes_only_selected_model_with_cpu_torch(tmp_path, monkeypatch, implementation):
    torch = pytest.importorskip("torch")
    marker = tmp_path / "executed"
    factory = (
        "import torch\ndef get_init_inputs(): return []\n"
        "def get_input_groups(): return [[torch.tensor([2.])]]\n"
    )
    model = (
        "class {name}(torch.nn.Module):\n"
        "    def forward(self, x):\n"
        "        assert x.item() == 2\n"
        "        open({marker!r}, 'w').write('selected')\n"
        "        return x * 2\n"
    )
    (tmp_path / "sample_torch.py").write_text(
        factory
        + (model.format(name="Model", marker=str(marker)) if implementation == "torch" else "")
    )
    (tmp_path / "sample_triton_ascend_impl.py").write_text(
        "raise AssertionError('missing Triton counterpart imported')\n"
        if implementation == "torch"
        else "import torch\n" + model.format(name="ModelNew", marker=str(marker))
    )
    monkeypatch.setitem(
        sys.modules,
        "_common_utils",
        SimpleNamespace(
            describe_input=lambda group: [],
            enforce_conv_fp32=lambda: None,
            ensure_npu_ambient_device=lambda torch: None,
            invoke_model=lambda model, values: model(*values),
            prepare_candidate_inputs=lambda model, values: values,
        ),
    )
    original_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "to", lambda self, device: original_to(self, "cpu"))
    monkeypatch.setattr(torch.nn.Module, "to", lambda self, device: self)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--engine-dir",
            str(tmp_path),
            "--verify-dir",
            str(tmp_path),
            "--op-name",
            "sample",
            "--triton-impl-name",
            "triton_ascend_impl",
            "--implementation",
            implementation,
            "--case-idx",
            "0",
        ],
    )
    namespace = {"__name__": "runner_test"}
    exec(compile(profile_runner.build_runner_source(), "runner", "exec"), namespace)
    namespace["main"]()
    assert marker.read_text() == "selected"


@pytest.mark.parametrize("mode", ["op", "deep"])
def test_torch_orchestrator_does_not_read_missing_triton_source(tmp_path, mode):
    config = profile_runner.ProfileConfig(
        op_name="sample",
        verify_dir=str(tmp_path),
        triton_impl_name="missing",
        profile_mode=mode,
        case_idx=None,
        launch_count=None,
        timeout=30,
        output=str(tmp_path / "result.json"),
        engine_dir=str(tmp_path),
        python_bin=sys.executable,
        implementation="torch",
    )
    seen = []

    def backend(**kwargs):
        seen.append(kwargs)
        return {"status": "partial", "warnings": ["native capture"], "error": None}

    result, status = profile_runner.run_profile(
        config,
        discover=lambda **kw: [{"case_idx": 0, "input_desc": []}],
        op_backend=backend,
        deep_backend=backend,
    )
    assert status == 0 and seen[0]["kernel_names"] is None
    assert seen[0]["app_cmd"][seen[0]["app_cmd"].index("--implementation") + 1] == "torch"
    assert result["per_case_results"][0]["warnings"] == ["native capture"]


@pytest.mark.parametrize("mode", ["op", "deep"])
def test_native_capture_uses_supported_msprof_cli_and_warns_about_attribution(
    tmp_path,
    monkeypatch,
    mode,
):
    fake = Path(__file__).parent / "fixtures/fake_msprof.py"
    log = tmp_path / "calls.log"
    monkeypatch.setenv("MSPROF_BIN", str(fake))
    monkeypatch.setenv("FAKE_MSPROF_LOG", str(log))
    kwargs = dict(
        app_cmd=[sys.executable, "-c", "pass"],
        kernel_names=None,
        output_dir=str(tmp_path / "capture"),
        cwd=str(tmp_path),
        env=dict(os.environ),
        deadline=time.monotonic() + 30,
    )
    result = (
        msprof_op.profile_op_case(**kwargs, launch_count=None)
        if mode == "op"
        else msprof_deep.profile_deep_case(**kwargs)
    )
    assert result["status"] == "partial"
    assert result["target_kernel"] == "add_kernel_hash"  # Actual emitted name, never rewritten.
    assert any("initialization" in warning or "setup" in warning for warning in result["warnings"])
    if mode == "op":
        command = log.read_text()
        assert "--launch-count=100" in command
        assert "--kernel-name=a|b|c" in command and ".*" not in command


def test_native_parser_accepts_real_operator_names_and_empty_triton_filter_rejects(tmp_path):
    from test_profile_op import _write_op_dir

    _write_op_dir(tmp_path, "aclnnAdd", 10)
    assert msprof_op.parse_op_output(str(tmp_path), None)["target_kernel"] == "aclnnAdd"
    assert msprof_op.parse_op_output(str(tmp_path), [])["status"] == "failed"
