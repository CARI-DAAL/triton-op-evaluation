import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from triton_evaluation.engine import profile_runner as profile


def test_engine_directory_does_not_shadow_stdlib_profile():
    engine_dir = Path(profile.__file__).parent
    completed = subprocess.run(
        [sys.executable, "-c", "import cProfile"],
        cwd=engine_dir,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


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
        "    return [{'type': 'scalar', 'value': safe(value)} for value in inputs]\n"
        "def ensure_npu_ambient_device(torch):\n"
        "    return False\n"
        "def prepare_candidate_inputs(reference_model, inputs):\n"
        "    return list(inputs)\n"
        "def invoke_model(model, inputs):\n"
        "    return model(*inputs)\n",
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


def test_generated_operator_discovery_uses_metadata_without_tensor_draws(tmp_path):
    _write_fake_runtime(tmp_path)
    normalized = [{
        "input_order": ["x", "scale", "tensors", "optional", "axes"],
        "inputs": {
            "x": {"type": "random", "shape": [1024, 4096], "dtype": "float32"},
            "scale": {"type": "scalar", "value": 3},
            "tensors": {"type": "tensor_list", "shapes": [[2, 3], []], "dtype": "float16"},
            "optional": {"type": "scalar", "value": None},
            "axes": {"type": "scalar", "value": [1, 2]},
        },
    }]
    (tmp_path / "sample_torch.py").write_text(
        f"_operator_cases = {normalized!r}\n"
        "class Groups:\n"
        "    def __len__(self): return len(_operator_cases)\n"
        "    def __getitem__(self, index):\n"
        "        raise AssertionError('metadata discovery must not materialize input tensors')\n"
        "def get_input_groups(): return Groups()\n"
    )
    cases = _discover(tmp_path, profile.write_runner(str(tmp_path)))
    assert cases == [{"case_idx": 0, "input_desc": [
        {"type": "tensor", "shape": [1024, 4096], "dtype": "torch.float32"},
        {"type": "scalar", "value": 3},
        {"type": "tensor_list", "tensors": [
            {"type": "tensor", "shape": [2, 3], "dtype": "torch.float16"},
            {"type": "tensor", "shape": [], "dtype": "torch.float16"},
        ]},
        {"type": "scalar", "value": "None"},
        {"type": "scalar", "value": "[1, 2]"},
    ]}]


def test_custom_metadata_falls_back_only_for_custom_case(tmp_path):
    _write_fake_runtime(tmp_path)
    cases = [
        {"input_order": ["x"], "inputs": {
            "x": {"type": "random", "shape": [100], "dtype": "float32"},
        }},
        {"input_order": ["x"], "inputs": {"x": {"type": "custom"}}},
    ]
    marker = tmp_path / "materialized"
    (tmp_path / "sample_torch.py").write_text(
        f"_operator_cases = {cases!r}\n"
        "class Groups:\n"
        "    def __len__(self): return len(_operator_cases)\n"
        "    def __getitem__(self, index):\n"
        "        assert index == 1, 'built-in case should not be materialized'\n"
        f"        open({str(marker)!r}, 'a').write(str(index))\n"
        "        return [47]\n"
        "def get_input_groups(): return Groups()\n"
    )
    described = _discover(tmp_path, profile.write_runner(str(tmp_path)))
    assert marker.read_text() == "1"
    assert described[1] == {"case_idx": 1, "input_desc": [{"type": "scalar", "value": 47}]}


def test_runner_falls_back_to_single_get_inputs_case(tmp_path):
    _write_fake_runtime(tmp_path)
    _write_task(tmp_path)
    (tmp_path / "sample_torch.py").write_text(
        "def get_init_inputs(): return []\ndef get_inputs(): return [33]\n",
        encoding="utf-8",
    )
    runner = profile.write_runner(str(tmp_path))
    cases = _discover(tmp_path, runner)
    assert cases == [{"case_idx": 0, "input_desc": [{"type": "scalar", "value": 33}]}]


def test_runner_accepts_tuple_case_groups(tmp_path):
    _write_fake_runtime(tmp_path)
    _write_task(tmp_path)
    (tmp_path / "sample_torch.py").write_text(
        "def get_init_inputs(): return []\ndef get_input_groups(): return ([11], (22,))\n",
        encoding="utf-8",
    )
    runner = profile.write_runner(str(tmp_path))
    cases = _discover(tmp_path, runner)
    assert [case["case_idx"] for case in cases] == [0, 1]


def test_runner_rejects_empty_input_groups(tmp_path):
    _write_fake_runtime(tmp_path)
    _write_task(tmp_path)
    (tmp_path / "sample_torch.py").write_text(
        "def get_init_inputs(): return []\ndef get_input_groups(): return iter(())\n",
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


def test_runner_registers_path_modules_before_execution(tmp_path):
    _write_fake_runtime(tmp_path)
    (tmp_path / "sample_torch.py").write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class InputSpec:\n"
        "    value: int\n"
        "def get_init_inputs(): return []\n"
        "def get_inputs(): return [InputSpec(7).value]\n",
        encoding="utf-8",
    )
    (tmp_path / "sample_triton_ascend_impl.py").write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "import os\n"
        "@dataclass\n"
        "class ModelNew:\n"
        "    value: int = 1\n"
        "    def to(self, device): return self\n"
        "    def __call__(self, value):\n"
        "        open(os.environ['KERNEL_MARKER'], 'w').write(str(value))\n",
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
    assert kernel_marker.read_text() == "7"


def test_profile_child_uses_real_default_preparation_before_candidate_call(tmp_path):
    from triton_evaluation.engine import _common_utils

    _write_fake_runtime(tmp_path)
    # Keep this test's NPU stub but execute the real shared invocation/default
    # functions in the separate profile child, not a mocked preparation route.
    import inspect

    common = tmp_path / "_common_utils.py"
    common.write_text(
        common.read_text()
        + "\n"
        + inspect.getsource(_common_utils.prepare_candidate_inputs)
        + "\n"
        + inspect.getsource(_common_utils.invoke_model)
    )
    (tmp_path / "sample_torch.py").write_text(
        "_ADAPTER_VERSION = 'official-task-v1'\n"
        "_PARAM_ORDER = ('x', 'scale', 'flag')\n"
        "_ABSENT = object()\n"
        "def run(x, scale=7, *, flag=True): return x * scale\n"
        "class Model:\n"
        "    def forward(self, *inputs): pass\n"
        "def get_init_inputs(): return []\n"
        "def get_input_groups(): return [[3, _ABSENT, _ABSENT]]\n"
    )
    (tmp_path / "sample_triton_ascend_impl.py").write_text(
        "import os\n"
        "class ModelNew:\n"
        "    def to(self, device): return self\n"
        "    def forward(self, x, scale=99, *, flag=False):\n"
        "        assert scale == 7 and flag is True\n"
        "        open(os.environ['KERNEL_MARKER'], 'w').write(str(x * scale))\n"
        "    def __call__(self, *args, **kwargs): return self.forward(*args, **kwargs)\n"
    )
    marker = tmp_path / "kernel.txt"
    environment = {**os.environ, "PYTHONPATH": str(tmp_path), "KERNEL_MARKER": str(marker)}
    runner = profile.write_runner(str(tmp_path))
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
        env=environment,
        timeout=30,
    )
    assert outcome.exit_code == 0, outcome.stderr
    assert marker.read_text() == "21"


def test_runner_materializes_only_one_lazy_case_at_a_time(tmp_path, monkeypatch):
    _write_fake_runtime(tmp_path)
    _write_task(tmp_path)
    (tmp_path / "sample_torch.py").write_text(
        "import gc, os, weakref\n"
        "class Payload:\n"
        "    def __init__(self, index): self.index = index\n"
        "    def __str__(self): return str(self.index)\n"
        "_live = []\n"
        "class Groups:\n"
        "    def __len__(self): return 3\n"
        "    def __getitem__(self, index):\n"
        "        gc.collect()\n"
        "        if any(ref() is not None for ref in _live):\n"
        "            raise RuntimeError('previous case retained')\n"
        "        open(os.environ['MATERIALIZED_LOG'], 'a').write(f'{index}\\n')\n"
        "        value = Payload(index)\n"
        "        _live.append(weakref.ref(value))\n"
        "        return [value]\n"
        "def get_init_inputs(): return []\n"
        "def get_input_groups(): return Groups()\n",
        encoding="utf-8",
    )
    log = tmp_path / "materialized.txt"
    monkeypatch.setenv("MATERIALIZED_LOG", str(log))
    runner = profile.write_runner(str(tmp_path))
    assert [case["case_idx"] for case in _discover(tmp_path, runner)] == [0, 1, 2]
    assert log.read_text().split() == ["0", "1", "2"]

    log.unlink()
    kernel_marker = tmp_path / "kernel.txt"
    env = {
        **os.environ,
        "PYTHONPATH": str(tmp_path),
        "HF32_MARKER": str(tmp_path / "hf32.txt"),
        "KERNEL_MARKER": str(kernel_marker),
    }
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
            "2",
        ],
        cwd=str(tmp_path),
        env=env,
        timeout=30,
    )
    assert outcome.exit_code == 0, outcome.stderr
    assert kernel_marker.read_text() == "2"
    assert log.read_text().split() == ["2"]


def test_discovery_caches_cases_across_invocations(tmp_path):
    _write_fake_runtime(tmp_path)
    _write_task(tmp_path)
    runner = profile.write_runner(str(tmp_path))
    first = _discover(tmp_path, runner)
    # Any fresh subprocess now fails loudly; only a cache hit can succeed.
    Path(runner).write_text("raise SystemExit(3)\n", encoding="utf-8")
    (tmp_path / "sample_torch.py").write_text("raise SystemExit(4)\n", encoding="utf-8")
    assert _discover(tmp_path, runner) == first == [
        {"case_idx": 0, "input_desc": [{"type": "scalar", "value": 11}]},
        {"case_idx": 1, "input_desc": [{"type": "scalar", "value": 22}]},
    ]


def test_discovery_cache_holds_the_described_payload(tmp_path):
    _write_fake_runtime(tmp_path)
    _write_task(tmp_path)
    runner = profile.write_runner(str(tmp_path))
    _discover(tmp_path, runner)
    payload = json.loads((tmp_path / "sample_cases.json").read_text(encoding="utf-8"))
    assert payload == {
        "total_cases": 2,
        "cases": [
            {"case_idx": 0, "input_desc": [{"type": "scalar", "value": 11}]},
            {"case_idx": 1, "input_desc": [{"type": "scalar", "value": 22}]},
        ],
    }


def test_discovery_reruns_subprocess_when_cache_is_corrupt(tmp_path):
    _write_fake_runtime(tmp_path)
    _write_task(tmp_path)
    runner = profile.write_runner(str(tmp_path))
    _discover(tmp_path, runner)
    (tmp_path / "sample_cases.json").write_text("{ not json", encoding="utf-8")
    Path(runner).write_text("raise SystemExit(3)\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="case discovery failed"):
        _discover(tmp_path, runner)


def test_discovery_overwrites_corrupt_cache_with_fresh_result(tmp_path):
    _write_fake_runtime(tmp_path)
    _write_task(tmp_path)
    runner = profile.write_runner(str(tmp_path))
    _discover(tmp_path, runner)
    (tmp_path / "sample_cases.json").write_text("{ not json", encoding="utf-8")
    cases = _discover(tmp_path, runner)
    assert [case["case_idx"] for case in cases] == [0, 1]
    payload = json.loads((tmp_path / "sample_cases.json").read_text(encoding="utf-8"))
    assert payload["total_cases"] == 2


def test_discovery_failure_writes_no_cache(tmp_path):
    _write_fake_runtime(tmp_path)
    _write_task(tmp_path)
    (tmp_path / "sample_torch.py").write_text(
        "def get_init_inputs(): return []\ndef get_input_groups(): return iter(())\n",
        encoding="utf-8",
    )
    runner = profile.write_runner(str(tmp_path))
    with pytest.raises(RuntimeError, match="case discovery failed"):
        _discover(tmp_path, runner)
    assert not (tmp_path / "sample_cases.json").exists()


def test_discovery_timeout_defaults_to_900_seconds(monkeypatch):
    monkeypatch.delenv("MSPROF_DISCOVERY_TIMEOUT_SEC", raising=False)
    assert profile.remaining_discovery_timeout(profile.time.monotonic() + 10_000) == 900


def test_discovery_timeout_env_caps_and_deadline_still_binds(monkeypatch):
    monkeypatch.setenv("MSPROF_DISCOVERY_TIMEOUT_SEC", "1200")
    assert profile.remaining_discovery_timeout(profile.time.monotonic() + 10_000) == 1200
    assert 59 <= profile.remaining_discovery_timeout(profile.time.monotonic() + 60) <= 61
    for invalid in ("not-a-number", "0", "-5"):
        monkeypatch.setenv("MSPROF_DISCOVERY_TIMEOUT_SEC", invalid)
        with pytest.raises(ValueError, match="MSPROF_DISCOVERY_TIMEOUT_SEC"):
            profile.remaining_discovery_timeout(profile.time.monotonic() + 60)


def test_discovery_honors_dedicated_timeout_env(tmp_path, monkeypatch):
    _write_fake_runtime(tmp_path)
    _write_task(tmp_path)
    (tmp_path / "sample_torch.py").write_text(
        "import time\n"
        "time.sleep(6)\n"
        "def get_init_inputs(): return []\n"
        "def get_input_groups():\n"
        "    yield [11]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MSPROF_DISCOVERY_TIMEOUT_SEC", "1")
    runner = profile.write_runner(str(tmp_path))
    with pytest.raises(
        RuntimeError, match="case discovery exceeded its per-process timeout"
    ):
        _discover(tmp_path, runner)
    assert not (tmp_path / "sample_cases.json").exists()
