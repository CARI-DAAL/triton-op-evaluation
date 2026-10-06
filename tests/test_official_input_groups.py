"""Official input materialization: ordered lazy pass is O(n) and bit-identical.

The engine iterates `_OfficialInputGroups` lazily per case. The random-access
path (`get_input_group(i)`) re-materializes cases 0..i for every i, so the old
iterator (one `__getitem__` per case) performed O(n^2) tensor generation —
51 BMM51 cases meant 1326 case materializations (~757 GB of randn churn) where
51 (~28 GB) suffice. The fixed iterator advances the adapter's own generator
once and replays the captured RNG state around each pull, so every pull starts
from exactly the state the random-access path would have (seed zero, preceding
cases consumed): bit-identical inputs, one materialization per case, and the
per-case isolation property preserved (interleaved engine RNG consumption
cannot shift later cases' inputs).
"""

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from triton_evaluation.engine import verify
from triton_evaluation.engine._common_utils import (
    _OfficialInputGroups,
    build_input_groups,
)


class _FakeTensor:
    """Values derive from the OWNING generator only (CPU tensors never see
    NPU RNG state and vice versa), like real torch tensors."""

    def __init__(self, generator_state):
        self.generator_state = generator_state

    def digest(self):
        return self.generator_state


class _FakeRandom(SimpleNamespace):
    pass


def _fake_torch_rng(npu_working=True, npu_available=True, npu_broken=False, default_device="cpu"):
    """torch double with observable, replayable CPU+NPU RNG state."""
    torch_mod = ModuleType("torch")
    torch_mod.Tensor = _FakeTensor
    torch_mod.get_default_device = lambda: default_device
    state = {"cpu": 0, "npu": 0}
    torch_mod.manual_seed = lambda seed: state.__setitem__("cpu", seed)

    def randn(*shape, dtype=None):
        # Only the ambient device's generator advances, like real torch.
        key = "cpu" if default_device == "cpu" else "npu"
        value = state[key]
        state[key] += 1
        return _FakeTensor(value)

    torch_mod.randn = randn
    torch_mod.random = _FakeRandom(
        get_rng_state=lambda: ("cpu-state", state["cpu"]),
        set_rng_state=lambda snap: state.__setitem__("cpu", snap[1]),
    )

    def npu_manual_seed(seed):
        state["npu"] = seed

    if npu_broken:
        def npu_get_rng_state():
            raise RuntimeError("device state unavailable")

        npu_set = None
    else:
        npu_get_rng_state = lambda: ("npu-state", state["npu"])
        npu_set = lambda snap: state.__setitem__("npu", snap[1])
    torch_mod.npu = SimpleNamespace(
        manual_seed=npu_manual_seed,
        get_rng_state=npu_get_rng_state,
        is_available=lambda: npu_available,
    )
    if npu_set is not None:
        torch_mod.npu.set_rng_state = npu_set
    torch_mod._state = state
    return torch_mod


def _synthetic_adapter(torch_mod, cases=5):
    module = ModuleType("synthetic_official_adapter")
    module._ADAPTER_VERSION = "official-task-v1"
    module._torch = torch_mod
    produced = {"count": 0}

    def _iter_input_groups():
        for _ in range(cases):
            produced["count"] += 1
            yield [torch_mod.randn(4)]

    module._iter_input_groups = _iter_input_groups
    module.produced = produced
    module.get_case_count = lambda: cases
    module.get_input_groups = lambda: list(_iter_input_groups())

    def get_input_group(index):
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise IndexError("negative")
        for position, inputs in enumerate(_iter_input_groups()):
            if position == index:
                return inputs
        raise IndexError("out of range")

    module.get_input_group = get_input_group
    return module


def _digests(iterable):
    return [[t.digest() for t in case] for case in iterable]


def _random_access_digests(torch_mod, module, cases, noise=False):
    result = []
    for index in range(cases):
        torch_mod.manual_seed(0)
        torch_mod.npu.manual_seed(0)
        if noise:
            torch_mod.randn(1)  # interleaved engine-side consumption
        result.append([t.digest() for t in module.get_input_group(index)])
    return result


def test_ordered_pass_is_bit_identical_and_single_materialization(monkeypatch):
    torch_mod = _fake_torch_rng()
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    module = _synthetic_adapter(torch_mod, cases=5)
    groups = build_input_groups(module)

    lazy = _digests(groups)
    lazy_materializations = module.produced["count"]

    assert lazy == _random_access_digests(torch_mod, module, 5)
    assert lazy_materializations == 5  # one materialization per case: O(n)


def test_interleaved_consumption_cannot_shift_later_cases(monkeypatch):
    torch_mod = _fake_torch_rng()
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    module = _synthetic_adapter(torch_mod, cases=5)

    clean = _digests(build_input_groups(module))
    noisy = []
    for case in build_input_groups(module):
        torch_mod.manual_seed(1234)  # candidate/reference work between pulls
        torch_mod.npu.manual_seed(99)
        torch_mod.randn(9)
        noisy.append([t.digest() for t in case])

    assert clean == noisy
    assert clean == _random_access_digests(torch_mod, module, 5)


def test_broken_npu_state_capture_falls_back_to_random_access(monkeypatch):
    # Ambient NPU set (engine normal path) but state capture fails: keep the
    # random-access semantics rather than risk interleaved NPU RNG drift.
    torch_mod = _fake_torch_rng(npu_broken=True, npu_available=True, default_device="npu")
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    module = _synthetic_adapter(torch_mod, cases=4)

    lazy = _digests(build_input_groups(module))
    fallback_materializations = module.produced["count"]

    assert lazy == _random_access_digests(torch_mod, module, 4)
    assert fallback_materializations == 4 * 5 // 2  # fallback: O(n^2), still correct


def test_module_without_generator_uses_random_access(monkeypatch):
    torch_mod = _fake_torch_rng()
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    module = _synthetic_adapter(torch_mod, cases=4)
    del module._iter_input_groups

    groups = build_input_groups(module)

    assert _digests(groups) == _random_access_digests(torch_mod, module, 4)
    assert len(groups) == 4


def test_cpu_default_device_skips_npu_state_capture(monkeypatch):
    """CPU contexts must not attempt NPU state capture (no device init, no
    O(n^2) fallback): the fast path runs with CPU-state snapshots alone."""
    torch_mod = _fake_torch_rng(npu_broken=True, npu_available=True, default_device="cpu")
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    module = _synthetic_adapter(torch_mod, cases=4)

    lazy = _digests(build_input_groups(module))
    fast_materializations = module.produced["count"]

    assert lazy == _random_access_digests(torch_mod, module, 4)
    assert fast_materializations == 4  # fast path, not the O(n^2) fallback


def test_indexed_npu_default_device_still_captures_npu_state(monkeypatch):
    """实机回归：torch.get_default_device() 带序号返回 "npu:0"（非 "npu"）。
    旧判断 str(default_device) == "npu" 恒为 False，NPU 状态从未被捕获/重放，
    引擎在 pull 间的 manual_seed(0) 重播种（_instantiate_models）于是把
    case≥2 的输入从 eager/sealed 契约上漂移（2026-10-07 实测：仅 case 1
    与 sealed input_sha256 吻合）。类型比较必须忽略序号。"""
    torch_mod = _fake_torch_rng(default_device="npu:0")
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    module = _synthetic_adapter(torch_mod, cases=5)

    clean = _digests(build_input_groups(module))
    engine_like = []
    for case in build_input_groups(module):
        engine_like.append([t.digest() for t in case])
        torch_mod.manual_seed(0)  # _instantiate_models 的同种子重建
        torch_mod.npu.manual_seed(0)

    assert clean == engine_like
    assert clean == _random_access_digests(torch_mod, module, 5)


def test_len_and_indexing_unchanged(monkeypatch):
    torch_mod = _fake_torch_rng()
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    module = _synthetic_adapter(torch_mod, cases=3)
    groups = _OfficialInputGroups(module)

    assert len(groups) == 3
    with pytest.raises(IndexError):
        groups[3]
    with pytest.raises(IndexError):
        groups[-1]


def _fake_import_module_for(reference):
    def fake_import_module(name):
        if name == "histc_torch":
            return reference
        if name == "histc_triton_ascend_impl":
            raise TypeError("boom")
        raise AssertionError(f"unexpected module import: {name}")

    return fake_import_module


def test_npu_strict_fails_fast_without_ambient_device(tmp_path, monkeypatch):
    """A strict run whose ambient NPU setup failed must die at once, not grind
    through single-threaded CPU materialization for the whole budget."""
    torch_mod = _fake_torch_rng(npu_broken=True, npu_available=False)
    del torch_mod.npu.get_rng_state
    reference = SimpleNamespace(
        Model=object,
        get_init_inputs=lambda: [],
        get_inputs=lambda: [17],
    )
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    monkeypatch.setattr(verify.importlib, "import_module", _fake_import_module_for(reference))
    monkeypatch.setattr(verify.sys, "path", list(verify.sys.path))

    with pytest.raises(RuntimeError, match="ambient NPU"):
        verify.verify_implementations(
            "histc",
            str(tmp_path),
            reference_device_policy="npu-strict",
        )


def test_cpu_first_policy_tolerates_missing_npu(tmp_path, monkeypatch):
    """cpu-first keeps its CPU fallback: no ambient device, no failure."""
    torch_mod = _fake_torch_rng(npu_broken=True, npu_available=False)
    del torch_mod.npu.get_rng_state
    reference = SimpleNamespace(
        Model=object,
        get_init_inputs=lambda: [],
        get_inputs=lambda: [17],
    )
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    monkeypatch.setattr(verify.importlib, "import_module", _fake_import_module_for(reference))
    monkeypatch.setattr(verify.sys, "path", list(verify.sys.path))

    output = tmp_path / "verify_result.json"
    assert verify.verify_implementations(
        "histc",
        str(tmp_path),
        output_path=str(output),
    ) == (0, 1)
    failure = json.loads(output.read_text())["failures"][0]
    assert failure["error_type"] == "CompilationError"
