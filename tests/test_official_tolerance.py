"""A declared workload tolerance is judged exactly like the pinned official check.

Primary source: Ascend/AscendC-KernelGen-data (Apache-2.0),
0607a29d231cde384c9dfdb8a39f54f91837c327,
npu-kernelbench/src/npu_kernelbench/data/workload.py (tolerance standards) and
npu-kernelbench/src/npu_kernelbench/evaluator/correctness.py, embedded verbatim.
"""

import math
import sys
from types import ModuleType, SimpleNamespace
from typing import Optional

import pytest
import torch

from triton_evaluation.api.operator import OperatorRequest
from triton_evaluation.engine import _official_tolerance, verify
from triton_evaluation.evaluators.operator import to_evaluation_request
from triton_evaluation.evaluators.shaping import shape_correctness

OFFICIAL_TOLERANCE_SOURCE = (
    '# ---------------------------------------------------------------------------\n'
    '\n'
    'DTYPE_TOLERANCE_STANDARDS: dict[str, dict] = {\n'
    '    "float16": {\n'
    '        "max_atol": 2**-9,       # ~1.953e-3\n'
    '        "max_rtol": 2**-9,       # ~1.953e-3\n'
    '        "required_matched_ratio": 0.99,\n'
    '        "max_error_cap": 1e-1,\n'
    '    },\n'
    '    "bfloat16": {\n'
    '        "max_atol": 2**-6,       # ~1.5625e-2\n'
    '        "max_rtol": 2**-6,       # ~1.5625e-2\n'
    '        "required_matched_ratio": 0.99,\n'
    '        "max_error_cap": 1e-0,\n'
    '    },\n'
    '    "float32": {\n'
    '        "max_atol": 2**-16,      # ~1.526e-5\n'
    '        "max_rtol": 2**-10,      # ~9.766e-4\n'
    '        "required_matched_ratio": 0.99,\n'
    '        "max_error_cap": 1e-2,\n'
    '    },\n'
    '    "float64": {\n'
    '        "max_atol": 2**-16,      # ~1.526e-5\n'
    '        "max_rtol": 2**-10,      # ~9.766e-4\n'
    '        "required_matched_ratio": 0.99,\n'
    '        "max_error_cap": 1e-2,\n'
    '    },\n'
    '    "float8_e4m3fn": {\n'
    '        "max_atol": 2**-4,       # ~6.25e-2\n'
    '        "max_rtol": 2**-2,       # ~2.5e-1\n'
    '        "required_matched_ratio": 0.99,\n'
    '        "max_error_cap": 1e-0,\n'
    '    },\n'
    '    "float8_e5m2": {\n'
    '        "max_atol": 2**-3,       # ~1.25e-1\n'
    '        "max_rtol": 2**-1,       # ~5e-1\n'
    '        "required_matched_ratio": 0.99,\n'
    '        "max_error_cap": 1e-1,\n'
    '    },\n'
    '    "hifloat32": {\n'
    '        "max_atol": 2**-10,      # ~9.766e-4\n'
    '        "max_rtol": 2**-9,       # ~1.953e-3\n'
    '        "required_matched_ratio": 0.99,\n'
    '        "max_error_cap": 1e-1,\n'
    '    },\n'
    '}\n'
    '\n'
    '# Fallback for unrecognized dtypes (use old defaults)\n'
    '_DEFAULT_TOLERANCE = {\n'
    '    "max_atol": 1e-2,\n'
    '    "max_rtol": 1e-2,\n'
    '    "required_matched_ratio": 0.99,\n'
    '    "max_error_cap": None,\n'
    '}\n'
    '\n'
    '\n'
    'def resolve_tolerance(\n'
    '    tolerance: ToleranceSpec, dtype: "torch.dtype | str | None" = None\n'
    ') -> ToleranceSpec:\n'
    '    """Resolve a ToleranceSpec by filling None fields from the dtype-based standard.\n'
    '\n'
    '    If a field is explicitly set (not None), it is preserved as-is.\n'
    '    If dtype is None or unrecognized, falls back to legacy defaults (a'
    'tol=1e-2, rtol=1e-2).\n'
    '    """\n'
    '    # Normalize dtype to string key\n'
    '    dtype_key: str | None = None\n'
    '    if dtype is not None:\n'
    '        dtype_key = str(dtype).replace("torch.", "")\n'
    '\n'
    '    standard = DTYPE_TOLERANCE_STANDARDS.get(dtype_key, _DEFAULT_TOLER'
    'ANCE) if dtype_key else _DEFAULT_TOLERANCE\n'
    '\n'
    '    return ToleranceSpec(\n'
    '        max_atol=tolerance.max_atol if tolerance.max_atol is not None '
    'else standard["max_atol"],\n'
    '        max_rtol=tolerance.max_rtol if tolerance.max_rtol is not None '
    'else standard["max_rtol"],\n'
    '        required_matched_ratio=(\n'
    '            tolerance.required_matched_ratio\n'
    '            if tolerance.required_matched_ratio is not None\n'
    '            else standard["required_matched_ratio"]\n'
    '        ),\n'
    '        max_error_cap=(\n'
    '            tolerance.max_error_cap\n'
    '            if tolerance.max_error_cap is not None\n'
    '            else standard.get("max_error_cap")\n'
    '        ),\n'
    '        allow_negative_inf=tolerance.allow_negative_inf,\n'
    '    )\n'
    '\n'
    '\n'
    '_MANTISSA_BITS: dict[str, int] = {\n'
    '    "float16": 10,\n'
    '    "bfloat16": 7,\n'
    '    "float32": 23,\n'
    '    "float64": 52,\n'
    '    "float8_e4m3fn": 3,\n'
    '    "float8_e5m2": 2,\n'
    '    "hifloat32": 11,\n'
    '}\n'
    '\n'
    '\n'
    'def _ulp_scale_32(dtype: "torch.dtype | str | None", golden) -> float:\n'
    '    """Return 32 * ULP at the golden\'s largest magnitude for ``dtype``.\n'
    '\n'
    '    ULP (unit in the last place) of a normal float is 2^(floor(log2(|v|)) -\n'
    '    mantissa_bits). Per the opbase experimental standard, the absolute error\n'
    '    hard limit is ``max(fixed_cap, 32 * ULP)``, so large-magnitude outputs\n'
    '    get a proportionally larger allowance instead of being killed by a fixed\n'
    '    cap. Falls back to 0.0 (i.e. fixed cap only) for unsupported dtypes,\n'
    '    zero/empty goldens or non-finite maxima.\n'
    '    """\n'
    '    dtype_key = str(dtype).replace("torch.", "") if dtype is not None else ""\n'
    '    mantissa = _MANTISSA_BITS.get(dtype_key)\n'
    '    if mantissa is None:\n'
    '        return 0.0\n'
    '    golden_f = golden.float() if golden.dtype != torch.float32 else golden\n'
    '    max_abs = float(golden_f.abs().max().item()) if golden_f.numel() else 0.0\n'
    '    if max_abs <= 0.0 or not (max_abs == max_abs) or max_abs == float("inf"):\n'
    '        return 0.0\n'
    '    exponent = math.floor(math.log2(max_abs))  # floor(log2(|v|)) for any |v| > 0\n'
    '    return 32.0 * (2.0 ** (exponent - mantissa))\n'
    '\n'
    '\n'
    'def resolve_max_error_cap(\n'
    '    tolerance: ToleranceSpec, dtype: "torch.dtype | str | None", golden\n'
    ') -> "float | None":\n'
    '    """Resolve the absolute error hard limit for a given golden tensor.\n'
    '\n'
    '    Per the opbase experimental standard: ``max_abs_error_limit = max(fixed\n'
    "    cap, 32 * ULP)``, where ULP is measured at the golden's largest magnitude.\n"
    '    The fixed cap comes from the (already resolved) tolerance; None means no\n'
    '    hard limit (legacy dtypes without a cap entry).\n'
    '    """\n'
    '    if tolerance.max_error_cap is None:\n'
    '        return None\n'
    '    return max(tolerance.max_error_cap, _ulp_scale_32(dtype, golden))\n'
)

OFFICIAL_CORRECTNESS_SOURCE = (
    'def check_tensor_sanity(\n'
    '    sol_tensor: torch.Tensor,\n'
    '    ref_tensor: torch.Tensor,\n'
    '    allow_negative_inf: bool = False,\n'
    ') -> Optional[Correctness]:\n'
    '    """Check for non-finite values and all-zeros output.\n'
    '\n'
    '    Returns a Correctness describing the failure, or None if ok.\n'
    '    """\n'
    '    ref_nonfinite = ~torch.isfinite(ref_tensor)\n'
    '    sol_nonfinite = ~torch.isfinite(sol_tensor)\n'
    '\n'
    '    if allow_negative_inf:\n'
    '        both_neg_inf = (ref_tensor == float("-inf")) & (sol_tensor == float("-inf"))\n'
    '        ref_nonfinite = ref_nonfinite & ~both_neg_inf\n'
    '        sol_nonfinite = sol_nonfinite & ~both_neg_inf\n'
    '\n'
    '    has_nonfinite = ref_nonfinite.any().item() or sol_nonfinite.any().item()\n'
    '    if has_nonfinite:\n'
    '        has_nan = (\n'
    '            torch.isnan(sol_tensor).any().item()\n'
    '            or torch.isnan(ref_tensor).any().item()\n'
    '        )\n'
    '        return Correctness(has_nan=has_nan, has_inf=not has_nan)\n'
    '\n'
    '    # All-zero output check\n'
    '    ref_norm = torch.linalg.vector_norm(ref_tensor.to(torch.float32))\n'
    '    if (\n'
    '        ref_norm.item() > 0\n'
    '        and torch.linalg.vector_norm(sol_tensor.to(torch.float32)).item() == 0\n'
    '    ):\n'
    '        abs_err = float(ref_norm.item())\n'
    '        return Correctness(\n'
    '            max_absolute_error=abs_err,\n'
    '            max_relative_error=abs_err,\n'
    '        )\n'
    '\n'
    '    return None\n'
    '\n'
    '\n'
    'def compute_error_stats(\n'
    '    output: torch.Tensor, reference: torch.Tensor, tolerance: ToleranceSpec\n'
    ') -> Tuple[Correctness, bool]:\n'
    '    """Compute numerical error between output and reference.\n'
    '\n'
    '    Returns (correctness, exceeds_tolerance).\n'
    '    """\n'
    '    # Opbase experimental standard: resolve dtype-based tolerance once here so\n'
    '    # every caller (DefaultEvaluator, standalone eval.py, ...) gets the same\n'
    '    # precision standard without duplicating resolve_tolerance calls.\n'
    '    from npu_kernelbench.data.workload import resolve_tolerance\n'
    '\n'
    '    tolerance = resolve_tolerance(tolerance, reference.dtype)\n'
    '\n'
    '    x = output.to(torch.float32)\n'
    '    y = reference.to(torch.float32)\n'
    '\n'
    '    allow_neg_inf = tolerance.allow_negative_inf\n'
    '\n'
    '    infs_nans = check_tensor_sanity(x, y, allow_negative_inf=allow_neg_inf)\n'
    '    if infs_nans is not None:\n'
    '        return infs_nans, True\n'
    '\n'
    '    if allow_neg_inf:\n'
    '        both_neg_inf = (x == float("-inf")) & (y == float("-inf"))\n'
    '        finite_mask = ~both_neg_inf\n'
    '        x = x[finite_mask]\n'
    '        y = y[finite_mask]\n'
    '\n'
    '    abs_error = torch.abs(x - y)\n'
    '    total_elements = abs_error.numel()\n'
    '    if total_elements == 0:\n'
    '        return Correctness(), False\n'
    '\n'
    '    max_abs = float(abs_error.max().item())\n'
    '\n'
    '    tol_bound = tolerance.max_atol + tolerance.max_rtol * torch.abs(y)\n'
    '    exceeds_tol_mask = (abs_error > tol_bound) | ~torch.isfinite(abs_error)\n'
    '\n'
    '    exceeds_count = float(exceeds_tol_mask.sum().item())\n'
    '    matched_ratio = 1.0 - (exceeds_count / float(total_elements))\n'
    '    matched_ratio = max(0.0, min(1.0, matched_ratio))\n'
    '\n'
    '    exceeds_tol = matched_ratio < tolerance.required_matched_ratio\n'
    '    if tolerance.max_error_cap is not None:\n'
    '        # Opbase experimental standard: max_abs_error_limit = max(fixed cap,\n'
    '        # 32*ULP) — resolved in workload.py so the standard stays defined in\n'
    "        # one place. ULP scales with the golden's magnitude, so large-value\n"
    '        # accumulations/reductions are not killed by a fixed absolute cap.\n'
    '        from npu_kernelbench.data.workload import resolve_max_error_cap\n'
    '\n'
    '        cap_limit = resolve_max_error_cap(tolerance, reference.dtype, y)\n'
    '        if cap_limit is not None and max_abs > cap_limit:\n'
    '            exceeds_tol = True\n'
    '\n'
    '    rel_error = abs_error / torch.clamp(torch.abs(y), min=tolerance.max_atol)\n'
    '    max_rel = float(rel_error.max().item())\n'
    '\n'
    '    return Correctness(\n'
    '        max_absolute_error=max_abs, max_relative_error=max_rel\n'
    '    ), exceeds_tol\n'
)


class ToleranceSpec(SimpleNamespace):
    def __init__(self, max_atol=None, max_rtol=None, required_matched_ratio=None,
                 max_error_cap=None, allow_negative_inf=False):
        super().__init__(
            max_atol=max_atol, max_rtol=max_rtol,
            required_matched_ratio=required_matched_ratio,
            max_error_cap=max_error_cap, allow_negative_inf=allow_negative_inf,
        )


class Correctness(SimpleNamespace):
    def __init__(self, max_relative_error=0.0, max_absolute_error=0.0,
                 has_nan=False, has_inf=False):
        super().__init__(
            max_relative_error=max_relative_error, max_absolute_error=max_absolute_error,
            has_nan=has_nan, has_inf=has_inf,
        )


@pytest.fixture
def official(monkeypatch):
    """Verbatim upstream compute_error_stats with its own workload module."""
    workload = ModuleType("npu_kernelbench.data.workload")
    workload.__dict__.update(torch=torch, math=math, ToleranceSpec=ToleranceSpec)
    exec(OFFICIAL_TOLERANCE_SOURCE, workload.__dict__)
    for name in ("npu_kernelbench", "npu_kernelbench.data"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, "npu_kernelbench.data.workload", workload)
    namespace = {
        "torch": torch, "Correctness": Correctness, "ToleranceSpec": ToleranceSpec,
        "Optional": Optional, "Tuple": tuple,
    }
    exec(OFFICIAL_CORRECTNESS_SOURCE, namespace)
    return namespace["compute_error_stats"]


def _reference(dtype=torch.float32, scale=3.0):
    generator = torch.Generator().manual_seed(7)
    return (torch.randn(4096, generator=generator) * scale).to(dtype)


def _perturbed(reference, error, fraction=1.0):
    candidate = reference.clone().to(torch.float32)
    count = int(candidate.numel() * fraction)
    candidate[:count] += error
    return candidate.to(reference.dtype)


def _with(reference, positions, value):
    out = reference.clone()
    out[positions] = value
    return out


ref32 = _reference()
CASES = {
    "f32 noise, dtype standard": (ref32, _perturbed(ref32, 1e-6), {}),
    "f32 offset, dtype standard": (ref32, _perturbed(ref32, 4e-3), {}),
    "f32 offset, declared atol": (ref32, _perturbed(ref32, 4e-3), {"max_atol": 0.01}),
    "f32 2% outliers, ratio 0.99": (ref32, _perturbed(ref32, 1.0, 0.02), {"max_atol": 0.01}),
    "f32 2% outliers, ratio 0.97": (
        ref32, _perturbed(ref32, 1e-3, 0.02),
        {"max_atol": 1e-4, "required_matched_ratio": 0.97},
    ),
    "f32 tight cap": (ref32, _perturbed(ref32, 4e-3), {"max_atol": 0.01, "max_error_cap": 1e-6}),
    "bf16 minimax": (
        _reference(torch.bfloat16), _perturbed(_reference(torch.bfloat16), 0.01),
        {"max_atol": 0.02, "max_rtol": 0.02},
    ),
    "f16 large magnitude, ULP cap": (
        _reference(torch.float16, 1000.0), _perturbed(_reference(torch.float16, 1000.0), 0.5),
        {"max_atol": 1.0},
    ),
    "matching NaN still fails": (
        _with(ref32, [3], float("nan")), _with(ref32, [3], float("nan")), {},
    ),
    "matching -inf, allowed": (
        _with(ref32, [1, 9], float("-inf")), _with(ref32, [1, 9], float("-inf")),
        {"allow_negative_inf": True},
    ),
    "matching -inf, not allowed": (
        _with(ref32, [1, 9], float("-inf")), _with(ref32, [1, 9], float("-inf")), {},
    ),
    "all-zero candidate": (ref32, torch.zeros_like(ref32), {"max_atol": 100.0}),
    "int64 indices": (
        torch.arange(64), torch.arange(64).flip(0), {"max_atol": 0.0, "max_rtol": 0.0},
    ),
    "empty": (torch.empty(0), torch.empty(0), {}),
}


@pytest.mark.parametrize("name", list(CASES))
def test_error_stats_match_the_pinned_official_check(name, official):
    reference, candidate, declared = CASES[name]
    expected, expected_exceeds = official(candidate, reference, ToleranceSpec(**declared))
    stats, exceeds = _official_tolerance.compute_error_stats(candidate, reference, declared)
    assert exceeds is expected_exceeds
    for field in ("max_absolute_error", "max_relative_error", "has_nan", "has_inf"):
        default = False if field.startswith("has_") else 0.0
        assert stats.get(field, default) == getattr(expected, field), field


def test_declared_tolerance_decides_the_engine_verdict():
    reference = torch.linspace(-3, 3, 1024)
    candidate = reference + 4e-3
    with pytest.raises(verify.AccuracyError):
        verify.compare(reference, candidate, reference.dtype)
    verify.compare(reference, candidate, reference.dtype, tolerance={"max_atol": 0.01})
    with pytest.raises(verify.AccuracyError) as caught:
        verify.compare(reference, candidate, reference.dtype, tolerance={"max_atol": 1e-3})
    metrics = caught.value.metrics
    assert metrics["tolerance"] == "declared"
    assert metrics["atol"] == 1e-3 and metrics["rtol"] == 2**-10
    assert metrics["matched_ratio"] == 0.0
    assert metrics["divergence"]["kind"] is not None


def test_declared_tolerance_rejects_dtype_mismatch_and_non_finite():
    reference = torch.linspace(-3, 3, 64)
    with pytest.raises(verify.AccuracyError, match="dtype"):
        verify.compare(reference, reference.half(), reference.dtype, tolerance={"max_atol": 1.0})
    with_nan = _with(reference, [5], float("nan"))
    verify.compare(with_nan, with_nan.clone(), reference.dtype)  # service policy accepts
    with pytest.raises(verify.AccuracyError):
        verify.compare(with_nan, with_nan.clone(), reference.dtype, tolerance={"max_atol": 1.0})


def test_declared_failure_shapes_as_a_candidate_value_mismatch():
    reference = torch.linspace(-3, 3, 256)
    with pytest.raises(verify.AccuracyError) as caught:
        verify.compare(reference, reference + 0.5, reference.dtype, tolerance={"max_atol": 0.01})
    divergence = caught.value.metrics["divergence"]
    raw = {
        "op_name": "operator", "total_cases": 1, "passed_cases": 0, "failed_cases": 1,
        "failures": [{
            "case_idx": 1, "error_type": "AccuracyError", "error_msg": str(caught.value),
            "kind": divergence["kind"], "sample": divergence["sample"],
            "metrics": caught.value.metrics,
        }],
    }
    result = shape_correctness(raw, decoy=False)
    assert result.passed is False and result.outcome == "wrong"
    assert result.failures[0].kind == "value_mismatch"
    assert result.failures[0].metrics.atol == 0.01


def test_reference_module_exposes_declared_tolerances():
    x = {"type": "random", "dtype": "float32", "shape": [4]}
    request = OperatorRequest.model_validate({
        "definition": {"inputs": {"x": {}}, "reference": "def run(x): return x"},
        "workloads": [
            {"inputs": {"x": x}},
            {"inputs": {"x": x}, "tolerance": {"max_atol": 0.02, "max_error_cap": None}},
            {"inputs": {"x": x}, "tolerance": {"allow_negative_inf": True}},
        ],
    })
    namespace = {}
    exec(to_evaluation_request(request).reference_code, namespace)
    assert namespace["get_case_tolerances"]() == [
        None, {"max_atol": 0.02}, {"allow_negative_inf": True},
    ]


def test_verify_hands_each_case_its_declared_tolerance(tmp_path, monkeypatch):
    fake_torch = ModuleType("torch")
    fake_torch.manual_seed = lambda seed: None
    fake_torch.npu = SimpleNamespace(manual_seed=lambda seed: None)
    fake_torch.device = lambda name: name
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    declared = [None, {"max_atol": 0.02}]
    reference = SimpleNamespace(get_case_tolerances=lambda: declared)
    monkeypatch.setattr(verify, "_load_verify_modules", lambda *args: {"torch_module": reference})
    monkeypatch.setattr(verify, "_ensure_npu_ambient_shared", lambda *args, **kwargs: True)
    monkeypatch.setattr(verify, "resolve_input_provider", lambda module: ([[1], [2]], 2))
    seen = []

    def run_case(modules, case_ctx, inputs, device, **kwargs):
        seen.append(case_ctx.tolerance)
        return True, None

    monkeypatch.setattr(verify, "_try_run_case", run_case)
    output = tmp_path / "verify_result.json"
    assert verify.verify_implementations("op", str(tmp_path), output_path=str(output)) == (2, 2)
    assert seen == declared
