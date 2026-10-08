"""Tests for triton_evaluation.official_task (adapter composition).

CPU-only structural tests. NPU correctness of composed tasks is validated
by the F-26 audit pipeline on-device, not here.
"""
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from triton_evaluation import official_task  # noqa: E402

OFFICIAL_ROOT = Path(
    "/home/e84381970/open-evo-data/ascendc-kernelgen-data/npu-kernelbench"
    "/data/kernel_generator"
)
requires_official = pytest.mark.skipif(
    not OFFICIAL_ROOT.is_dir(), reason="official npu-kernelbench data not present"
)


def _task_dir(level: str, name: str) -> Path:
    return OFFICIAL_ROOT / level / name


DEFINITION = {
    "name": "Demo",
    "inputs": {"x1": {}, "x2": {}, "scale": {}, "output_dtype": {}},
    "reference": (
        "import torch\n"
        "import torch.nn as nn\n"
        "\n"
        "def run(x1, x2, scale, output_dtype=None, group_sizes=None):\n"
        "    return x1 @ x2 + (0 if scale is None else scale.sum())\n"
    ),
}
WORKLOAD = (
    '{"uuid": "d-1", "inputs": {"x1": {"type": "random", "shape": [4, 8], "dtype": "int8", "range": [-3, 3]},'
    ' "x2": {"type": "random", "shape": [8, 4], "dtype": "int8", "range": [-3, 3]},'
    ' "scale": {"type": "random", "shape": [4], "dtype": "float32"},'
    ' "output_dtype": {"type": "scalar", "value": "float16"}}}\n'
    '{"uuid": "d-2", "inputs": {"x1": {"type": "random", "shape": [2, 6], "dtype": "int8"},'
    ' "x2": {"type": "random", "shape": [6, 2], "dtype": "int8"}}}\n'
)


def _compose():
    return official_task.compose_reference_source(
        json.dumps(DEFINITION).encode(), WORKLOAD.encode(), problem_id="demo_task"
    )


def _import_composed(source: str):
    namespace: dict = {}
    exec(compile(source, "composed_reference.py", "exec"), namespace)
    return namespace


class TestCompose:
    def test_deterministic_bytes(self):
        assert _compose() == _compose()

    def test_header_stamps(self):
        source = _compose()
        assert official_task.OFFICIAL_ADAPTER_VERSION in source
        assert "problem_id: demo_task" in source
        digest = official_task.combined_digest(
            json.dumps(DEFINITION).encode(), WORKLOAD.encode()
        )
        assert digest in source

    def test_official_code_verbatim(self):
        source = _compose()
        assert DEFINITION["reference"].rstrip() in source

    def test_compiles(self):
        compile(_compose(), "x", "exec")

    def test_rejects_missing_run(self):
        bad = dict(DEFINITION, reference="def not_run(): pass")
        with pytest.raises(official_task.OfficialTaskError):
            official_task.compose_reference_source(
                json.dumps(bad).encode(), WORKLOAD.encode(), problem_id="bad"
            )

    def test_rejects_undeclared_workload_input(self):
        workload = WORKLOAD.replace('"x2"', '"zzz"')
        with pytest.raises(official_task.OfficialTaskError):
            official_task.compose_reference_source(
                json.dumps(DEFINITION).encode(), workload.encode(), problem_id="bad"
            )

    def test_rejects_bad_dtype(self):
        workload = WORKLOAD.replace('"int8"', '"complex128"')
        with pytest.raises(official_task.OfficialTaskError):
            official_task.compose_reference_source(
                json.dumps(DEFINITION).encode(), workload.encode(), problem_id="bad"
            )

    def test_rejects_empty_workload(self):
        with pytest.raises(official_task.OfficialTaskError):
            official_task.compose_reference_source(
                json.dumps(DEFINITION).encode(), b"\n", problem_id="bad"
            )


class TestComposedModule:
    def _module(self):
        return _import_composed(_compose())

    def test_structure(self):
        mod = self._module()
        assert hasattr(mod, "Model") and hasattr(mod, "get_input_groups")
        assert hasattr(mod, "get_init_inputs") and callable(mod.run)

    def test_case_count_and_tensor_spec(self):
        mod = self._module()
        groups = mod.get_input_groups()
        assert len(groups) == 2
        x1, x2, scale, output_dtype = groups[0]
        assert x1.dtype is torch.int8 and tuple(x1.shape) == (4, 8)
        assert x2.dtype is torch.int8 and tuple(x2.shape) == (8, 4)
        assert scale.dtype is torch.float32 and tuple(scale.shape) == (4,)
        assert output_dtype == "float16"

    def test_range_honored(self):
        mod = self._module()
        x1 = mod.get_input_groups()[0][0]
        assert int(x1.min()) >= -3 and int(x1.max()) <= 3

    def test_absent_params_get_sentinel_and_forward_kwargs_only(self):
        mod = self._module()
        groups = mod.get_input_groups()
        case2 = groups[1]  # only x1, x2 present
        assert mod._ABSENT in case2

        seen = {}

        def spy(**kwargs):
            seen.update(kwargs)
            return kwargs["x1"]

        original_run = mod.run
        mod.run = spy
        try:
            torch.set_default_device("cpu")
            model = mod.Model()
            model(*case2)
        finally:
            mod.run = original_run
            torch.set_default_device("npu") if torch.npu.is_available() else None
        # Absent params (scale/output_dtype/group_sizes) must NOT be passed:
        assert set(seen) == {"x1", "x2"}

    def test_forward_returns_run_result(self):
        mod = self._module()
        groups = mod.get_input_groups()
        torch.set_default_device("cpu")
        try:
            model = mod.Model()
            out = model(*groups[0])
            assert isinstance(out, torch.Tensor) and tuple(out.shape) == (4, 4)
        finally:
            if torch.npu.is_available():
                torch.set_default_device("npu")


@requires_official
class TestRealOfficialTasks:
    SELECTED = [
        ("level3", "BatchMatmul"), ("level3", "GroupedMatmul"),
        ("level4", "QuantMatmul"), ("level4", "LightningIndexer"),
        ("level4", "FusionAttention"), ("level4", "GroupedMatmulSwigluQuantV2"),
    ]

    @pytest.mark.parametrize("level,name", SELECTED)
    def test_compose_and_structure(self, level, name):
        entry = official_task.load_task(_task_dir(level, name))
        source = entry["reference_source"]
        assert entry["cases"] > 0
        compile(source, name, "exec")
        namespace = {}
        exec(compile(source, f"{name}.py", "exec"), namespace)
        assert hasattr(namespace["Model"], "forward")
        assert callable(namespace["get_input_groups"])
        # replay determinism
        again = official_task.compose_reference_source(
            (_task_dir(level, name) / "definition.json").read_bytes(),
            (_task_dir(level, name) / "workload.jsonl").read_bytes(),
            problem_id=name,
        )
        assert again == source
