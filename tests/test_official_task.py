"""CPU-only adapter tests; official NPU references are compiled, never executed."""

import hashlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from triton_evaluation import official_task

torch = pytest.importorskip("torch")

OFFICIAL_ROOT = Path(os.environ.get("OFFICIAL_TASK_ROOT", "/nonexistent/official-task-data"))
requires_official = pytest.mark.skipif(
    not OFFICIAL_ROOT.is_dir(), reason="set OFFICIAL_TASK_ROOT to official kernel_generator data"
)

DEFINITION = {
    "name": "Demo",
    "inputs": {"x1": {}, "x2": {}, "scale": {}, "output_dtype": {}},
    "reference": (
        "import torch\n"
        "import torch.nn as nn\n"
        "\n"
        "def run(x1, x2, scale=None, output_dtype=None, group_sizes=None):\n"
        "    return x1 @ x2 + (0 if scale is None else scale.sum())\n"
    ),
}
WORKLOAD = (
    '{"uuid": "d-1", "inputs": {'
    '"x1": {"type": "random", "shape": [4, 8], "dtype": "int8", "range": [-3, 3]},'
    ' "x2": {"type": "random", "shape": [8, 4], "dtype": "int8", "range": [-3, 3]},'
    ' "scale": {"type": "random", "shape": [4], "dtype": "float32"},'
    ' "output_dtype": {"type": "scalar", "value": "float16"}}}\n'
    '{"uuid": "d-2", "inputs": {'
    '"x1": {"type": "random", "shape": [2, 6], "dtype": "int8"},'
    ' "x2": {"type": "random", "shape": [6, 2], "dtype": "int8"}}}\n'
)


@pytest.fixture(autouse=True)
def cpu_device():
    # A scoped factory-device override does not import torch_npu or mutate a
    # caller's global default device. Every structural tensor test runs on CPU.
    with torch.device("cpu"):
        yield


def _compose():
    return official_task.compose_reference_source(
        json.dumps(DEFINITION).encode(), WORKLOAD.encode(), problem_id="demo_task"
    )


def _import_composed(source: str):
    module = types.ModuleType("composed_reference")
    exec(compile(source, "composed_reference.py", "exec"), module.__dict__)
    return module


class TestCompose:
    def test_deterministic_bytes(self):
        assert _compose() == _compose()

    def test_deterministic_across_process_hash_seeds(self, tmp_path):
        definition = tmp_path / "definition.json"
        workload = tmp_path / "workload.jsonl"
        definition.write_text(json.dumps(DEFINITION), encoding="utf-8")
        workload.write_text(WORKLOAD, encoding="utf-8")
        program = (
            "import hashlib, sys; from pathlib import Path; "
            "from triton_evaluation.official_task import compose_reference_source; "
            "s=compose_reference_source(Path(sys.argv[1]).read_bytes(), "
            "Path(sys.argv[2]).read_bytes(), problem_id='demo_task'); "
            "print(hashlib.sha256(s.encode()).hexdigest())"
        )
        hashes = [
            subprocess.check_output(
                [sys.executable, "-c", program, str(definition), str(workload)],
                env={**os.environ, "PYTHONHASHSEED": seed},
                text=True,
            ).strip()
            for seed in ("1", "2", "42", "random")
        ]
        assert len(set(hashes)) == 1
        assert hashes[0] == hashlib.sha256(_compose().encode()).hexdigest()

    def test_header_stamps(self):
        source = _compose()
        assert official_task.OFFICIAL_ADAPTER_VERSION in source
        assert "problem_id: demo_task" in source
        digest = official_task.combined_digest(json.dumps(DEFINITION).encode(), WORKLOAD.encode())
        assert digest in source

    def test_official_code_verbatim(self):
        reference = DEFINITION["reference"] + "  \n\n"
        source = official_task.compose_reference_source(
            json.dumps(dict(DEFINITION, reference=reference)).encode(),
            WORKLOAD.encode(),
            problem_id="demo_task",
        )
        assert reference in source

    def test_compiles(self):
        compile(_compose(), "x", "exec")

    def test_rejects_missing_run(self):
        with pytest.raises(official_task.OfficialTaskError):
            official_task.compose_reference_source(
                json.dumps(dict(DEFINITION, reference="def not_run(): pass")).encode(),
                WORKLOAD.encode(),
                problem_id="bad",
            )

    def test_rejects_undeclared_workload_input(self):
        with pytest.raises(official_task.OfficialTaskError):
            official_task.compose_reference_source(
                json.dumps(DEFINITION).encode(),
                WORKLOAD.replace('"x2"', '"zzz"').encode(),
                problem_id="bad",
            )

    def test_rejects_bad_dtype(self):
        with pytest.raises(official_task.OfficialTaskError):
            official_task.compose_reference_source(
                json.dumps(DEFINITION).encode(),
                WORKLOAD.replace('"int8"', '"complex128"').encode(),
                problem_id="bad",
            )

    def test_rejects_empty_workload(self):
        with pytest.raises(official_task.OfficialTaskError):
            official_task.compose_reference_source(
                json.dumps(DEFINITION).encode(), b"\n", problem_id="bad"
            )

    @pytest.mark.parametrize(
        "workload", [b"\xff", b"[]", b'{"inputs": []}', b'{"inputs": {"x1": 2}}']
    )
    def test_rejects_malformed_workload(self, workload):
        with pytest.raises(official_task.OfficialTaskError):
            official_task.compose_reference_source(
                json.dumps(DEFINITION).encode(), workload, problem_id="bad"
            )


class TestComposedModule:
    def _module(self):
        return _import_composed(_compose())

    def test_structure(self):
        mod = self._module()
        assert hasattr(mod, "Model") and hasattr(mod, "get_input_groups")
        assert hasattr(mod, "get_init_inputs") and callable(mod.run)
        assert mod.get_init_inputs() == []
        assert mod.get_case_count() == 2

    def test_case_count_and_tensor_spec(self):
        groups = self._module().get_input_groups()
        assert len(groups) == 2
        x1, x2, scale, output_dtype = groups[0]
        assert x1.device.type == x2.device.type == scale.device.type == "cpu"
        assert x1.dtype is torch.int8 and tuple(x1.shape) == (4, 8)
        assert x2.dtype is torch.int8 and tuple(x2.shape) == (8, 4)
        assert scale.dtype is torch.float32 and tuple(scale.shape) == (4,)
        assert output_dtype == "float16"

    def test_range_honored(self):
        x1 = self._module().get_input_groups()[0][0]
        assert int(x1.min()) >= -3 and int(x1.max()) <= 3

    def test_single_case_preserves_seed0_full_inventory_rng(self):
        mod = self._module()
        torch.manual_seed(0)
        expected = mod.get_input_groups()
        for index, group in enumerate(expected):
            torch.manual_seed(0)
            actual = mod.get_input_group(index)
            for before, after in zip(group, actual, strict=True):
                if isinstance(before, torch.Tensor):
                    assert torch.equal(before, after)
                else:
                    assert before is after or before == after

    @pytest.mark.parametrize("index", [-1, 2, True, "0"])
    def test_single_case_rejects_invalid_index(self, index):
        with pytest.raises(IndexError):
            self._module().get_input_group(index)

    def test_absent_params_get_sentinel_and_forward_kwargs_only(self):
        mod = self._module()
        case2 = mod.get_input_groups()[1]
        assert any(value is mod._ABSENT for value in case2)
        seen = {}

        def spy(**kwargs):
            seen.update(kwargs)
            return kwargs["x1"]

        mod.run = spy
        mod.Model()(*case2)
        assert set(seen) == {"x1", "x2"}

    def test_explicit_none_is_not_absence(self):
        mod = self._module()
        seen = {}
        mod.run = lambda **kwargs: seen.update(kwargs)
        mod.Model()(1, 2, None, mod._ABSENT)
        assert seen == {"x1": 1, "x2": 2, "scale": None}

    def test_missing_inputs_apply_python_defaults(self):
        mod = self._module()
        case2 = mod.get_input_groups()[1]
        assert torch.equal(mod.Model()(*case2), case2[0] @ case2[1])

    def test_forward_rejects_wrong_input_count(self):
        mod = self._module()
        with pytest.raises(ValueError, match="input count"):
            mod.Model()(1, 2)

    def test_forward_returns_run_result(self):
        mod = self._module()
        out = mod.Model()(*mod.get_input_groups()[0])
        assert isinstance(out, torch.Tensor) and tuple(out.shape) == (4, 4)

    @pytest.mark.parametrize("dtype,low", [("int8", -5), ("int64", -5), ("uint8", 0)])
    def test_integer_no_range_policy(self, dtype, low):
        mod = self._module()
        tensor = mod._materialize({"type": "tensor", "dtype": dtype, "shape": [1024]})
        assert tensor.dtype is getattr(torch, dtype)
        assert tensor.min().item() >= low
        assert tensor.max().item() < 5

    def test_scalar_and_bool_policy(self):
        mod = self._module()
        assert mod._materialize({"type": "attr", "value": None}) is None
        result = mod._materialize({"type": "tensor", "dtype": "bool", "shape": [10]})
        assert result.dtype is torch.bool


@requires_official
class TestRealOfficialTasks:
    @pytest.mark.parametrize(
        "level,name",
        [
            ("level3", "BatchMatmul"),
            ("level3", "GroupedMatmul"),
            ("level4", "QuantMatmul"),
            ("level4", "LightningIndexer"),
            ("level4", "FusionAttention"),
            ("level4", "GroupedMatmulSwigluQuantV2"),
        ],
    )
    def test_compose_and_compile_only(self, level, name):
        task_dir = OFFICIAL_ROOT / level / name
        entry = official_task.load_task(task_dir)
        assert entry["cases"] > 0
        # Official references may import torch_npu or launch NPU setup during
        # import. Structural coverage must not execute that code on a CPU host.
        compile(entry["reference_source"], f"{name}.py", "exec")
        assert (
            official_task.compose_reference_source(
                (task_dir / "definition.json").read_bytes(),
                (task_dir / "workload.jsonl").read_bytes(),
                problem_id=name,
            )
            == entry["reference_source"]
        )
