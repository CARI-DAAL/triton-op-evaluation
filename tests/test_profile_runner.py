import os
import sys
from pathlib import Path

import pytest

from triton_evaluation.engine import profile


def _write_fake_runtime(root: Path):
    (root / "torch.py").write_text(
        "class Tensor:\n"
        "    def __init__(self, value=None): self.value = value; self.device = None\n"
        "    def to(self, device): self.device = device; return self\n"
        "def manual_seed(value): pass\n"
        "class _Npu:\n"
        "    def manual_seed_all(self, value): pass\n"
        "    def synchronize(self): pass\n"
        "npu = _Npu()\n"
        "class no_grad:\n"
        "    def __enter__(self): return self\n"
        "    def __exit__(self, *args): return False\n",
        encoding="utf-8",
    )
    (root / "_common_utils.py").write_text(
        "import os\n"
        "def enforce_conv_fp32():\n"
        "    marker = os.environ.get('HF32_MARKER')\n"
        "    if marker:\n"
        "        open(marker, 'w').write('off')\n"
        "def describe_input(inputs):\n"
        "    def safe(value):\n"
        "        return value if isinstance(value, (int, float, str, bool)) else repr(value)\n"
        "    return [{'type': 'scalar', 'value': safe(value)} for value in inputs]\n",
        encoding="utf-8",
    )


def _write_task(root: Path):
    (root / "sample_torch.py").write_text(
        "def get_init_inputs(): return []\n"
        "def get_input_groups():\n"
        "    yield [11]\n"
        "    yield [22]\n",
        encoding="utf-8",
    )
    (root / "sample_triton_ascend_impl.py").write_text(
        "import os\n"
        "class ModelNew:\n"
        "    def __init__(self): pass\n"
        "    def to(self, device): return self\n"
        "    def __call__(self, value):\n"
        "        marker = os.environ['KERNEL_MARKER']\n"
        "        assert open(os.environ['HF32_MARKER']).read() == 'off'\n"
        "        open(marker, 'w').write(str(value))\n",
        encoding="utf-8",
    )


def _discover(tmp_path: Path, runner: str):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tmp_path)
    return profile.discover_cases(
        runner_path=runner,
        engine_dir=str(tmp_path),
        verify_dir=str(tmp_path),
        op_name="sample",
        impl_name="triton_ascend_impl",
        python_bin=sys.executable,
        env=env,
        deadline=profile.time.monotonic() + 30,
    )


def test_runner_discovers_generator_cases_without_running_kernel(tmp_path):
    _write_fake_runtime(tmp_path)
    _write_task(tmp_path)
    runner = profile.write_runner(str(tmp_path))
    cases = _discover(tmp_path, runner)
    assert [case["case_idx"] for case in cases] == [0, 1]
    assert cases[1]["input_desc"][0]["value"] == 22


def test_runner_falls_back_to_single_get_inputs_case(tmp_path):
    _write_fake_runtime(tmp_path)
    _write_task(tmp_path)
    (tmp_path / "sample_torch.py").write_text(
        "def get_init_inputs(): return []\n"
        "def get_inputs(): return [33]\n",
        encoding="utf-8",
    )
    runner = profile.write_runner(str(tmp_path))
    cases = _discover(tmp_path, runner)
    assert cases == [
        {"case_idx": 0, "input_desc": [{"type": "scalar", "value": 33}]}
    ]


def test_runner_accepts_tuple_case_groups(tmp_path):
    _write_fake_runtime(tmp_path)
    _write_task(tmp_path)
    (tmp_path / "sample_torch.py").write_text(
        "def get_init_inputs(): return []\n"
        "def get_input_groups(): return ([11], (22,))\n",
        encoding="utf-8",
    )
    runner = profile.write_runner(str(tmp_path))
    cases = _discover(tmp_path, runner)
    assert [case["case_idx"] for case in cases] == [0, 1]


def test_runner_rejects_empty_input_groups(tmp_path):
    _write_fake_runtime(tmp_path)
    _write_task(tmp_path)
    (tmp_path / "sample_torch.py").write_text(
        "def get_init_inputs(): return []\n"
        "def get_input_groups(): return iter(())\n",
        encoding="utf-8",
    )
    runner = profile.write_runner(str(tmp_path))
    with pytest.raises(RuntimeError, match="case discovery failed"):
        _discover(tmp_path, runner)


def test_runner_moves_nested_tensor_inputs_to_local_npu(tmp_path):
    _write_fake_runtime(tmp_path)
    (tmp_path / "sample_torch.py").write_text(
        "import torch\n"
        "def get_init_inputs(): return []\n"
        "def get_input_groups():\n"
        "    return [[{'x': torch.Tensor(7)}, (torch.Tensor(8),)]]\n",
        encoding="utf-8",
    )
    (tmp_path / "sample_triton_ascend_impl.py").write_text(
        "import os\n"
        "class ModelNew:\n"
        "    def __init__(self): pass\n"
        "    def to(self, device): return self\n"
        "    def __call__(self, mapping, pair):\n"
        "        assert mapping['x'].device == 'npu:0'\n"
        "        assert pair[0].device == 'npu:0'\n"
        "        open(os.environ['KERNEL_MARKER'], 'w').write('nested-ok')\n",
        encoding="utf-8",
    )
    runner = profile.write_runner(str(tmp_path))
    hf32_marker = tmp_path / "hf32.txt"
    kernel_marker = tmp_path / "kernel.txt"
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(tmp_path),
            "HF32_MARKER": str(hf32_marker),
            "KERNEL_MARKER": str(kernel_marker),
        }
    )
    outcome = profile.run_command(
        [
            sys.executable,
            runner,
            "--engine-dir",
            str(tmp_path),
            "--verify-dir",
            str(tmp_path),
            "--op-name",
            "sample",
            "--triton-impl-name",
            "triton_ascend_impl",
            "--case-idx",
            "0",
        ],
        cwd=str(tmp_path),
        env=env,
        timeout=30,
    )
    assert outcome.exit_code == 0, outcome.stderr
    assert kernel_marker.read_text() == "nested-ok"


def test_runner_disables_hf32_before_selected_case_launch(tmp_path):
    _write_fake_runtime(tmp_path)
    _write_task(tmp_path)
    runner = profile.write_runner(str(tmp_path))
    hf32_marker = tmp_path / "hf32.txt"
    kernel_marker = tmp_path / "kernel.txt"
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(tmp_path),
            "HF32_MARKER": str(hf32_marker),
            "KERNEL_MARKER": str(kernel_marker),
        }
    )
    outcome = profile.run_command(
        [
            sys.executable,
            runner,
            "--engine-dir",
            str(tmp_path),
            "--verify-dir",
            str(tmp_path),
            "--op-name",
            "sample",
            "--triton-impl-name",
            "triton_ascend_impl",
            "--case-idx",
            "1",
        ],
        cwd=str(tmp_path),
        env=env,
        timeout=30,
    )
    assert outcome.exit_code == 0, outcome.stderr
    assert hf32_marker.read_text() == "off"
    assert kernel_marker.read_text() == "22"
