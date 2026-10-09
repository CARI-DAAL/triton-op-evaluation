"""Pinned official correctness check for workloads that declare a tolerance.

Adapted from Ascend/AscendC-KernelGen-data (Apache-2.0), commit
0607a29d231cde384c9dfdb8a39f54f91837c327:
npu-kernelbench/src/npu_kernelbench/data/workload.py (dtype standards,
resolve_tolerance, the 32-ULP error cap) and
npu-kernelbench/src/npu_kernelbench/evaluator/correctness.py
(check_tensor_sanity, compute_error_stats). See runtime/INPUT_GENERATION_LICENSE.
ToleranceSpec and Correctness objects become plain dicts; the arithmetic is
unchanged. compute_error_stats additionally reports the matched ratio and the
resolved tolerance for diagnostics.
"""

import math

TOLERANCE_FIELDS = (
    "max_atol", "max_rtol", "required_matched_ratio", "max_error_cap", "allow_negative_inf",
)

DTYPE_TOLERANCE_STANDARDS = {
    "float16": {
        "max_atol": 2**-9,
        "max_rtol": 2**-9,
        "required_matched_ratio": 0.99,
        "max_error_cap": 1e-1,
    },
    "bfloat16": {
        "max_atol": 2**-6,
        "max_rtol": 2**-6,
        "required_matched_ratio": 0.99,
        "max_error_cap": 1e-0,
    },
    "float32": {
        "max_atol": 2**-16,
        "max_rtol": 2**-10,
        "required_matched_ratio": 0.99,
        "max_error_cap": 1e-2,
    },
    "float64": {
        "max_atol": 2**-16,
        "max_rtol": 2**-10,
        "required_matched_ratio": 0.99,
        "max_error_cap": 1e-2,
    },
    "float8_e4m3fn": {
        "max_atol": 2**-4,
        "max_rtol": 2**-2,
        "required_matched_ratio": 0.99,
        "max_error_cap": 1e-0,
    },
    "float8_e5m2": {
        "max_atol": 2**-3,
        "max_rtol": 2**-1,
        "required_matched_ratio": 0.99,
        "max_error_cap": 1e-1,
    },
    "hifloat32": {
        "max_atol": 2**-10,
        "max_rtol": 2**-9,
        "required_matched_ratio": 0.99,
        "max_error_cap": 1e-1,
    },
}

_DEFAULT_TOLERANCE = {
    "max_atol": 1e-2,
    "max_rtol": 1e-2,
    "required_matched_ratio": 0.99,
    "max_error_cap": None,
}

_MANTISSA_BITS = {
    "float16": 10,
    "bfloat16": 7,
    "float32": 23,
    "float64": 52,
    "float8_e4m3fn": 3,
    "float8_e5m2": 2,
    "hifloat32": 11,
}


def resolve_tolerance(tolerance, dtype=None):
    """Fill fields the workload left unset from the dtype-based standard."""
    dtype_key = None
    if dtype is not None:
        dtype_key = str(dtype).replace("torch.", "")
    standard = (
        DTYPE_TOLERANCE_STANDARDS.get(dtype_key, _DEFAULT_TOLERANCE)
        if dtype_key else _DEFAULT_TOLERANCE
    )

    def declared(name):
        return tolerance.get(name)

    return {
        "max_atol": (
            declared("max_atol") if declared("max_atol") is not None else standard["max_atol"]
        ),
        "max_rtol": (
            declared("max_rtol") if declared("max_rtol") is not None else standard["max_rtol"]
        ),
        "required_matched_ratio": (
            declared("required_matched_ratio")
            if declared("required_matched_ratio") is not None
            else standard["required_matched_ratio"]
        ),
        "max_error_cap": (
            declared("max_error_cap")
            if declared("max_error_cap") is not None
            else standard.get("max_error_cap")
        ),
        "allow_negative_inf": bool(tolerance.get("allow_negative_inf", False)),
    }


def _ulp_scale_32(dtype, golden):
    """Return 32 * ULP at the golden's largest magnitude for ``dtype``."""
    import torch

    dtype_key = str(dtype).replace("torch.", "") if dtype is not None else ""
    mantissa = _MANTISSA_BITS.get(dtype_key)
    if mantissa is None:
        return 0.0
    golden_f = golden.float() if golden.dtype != torch.float32 else golden
    max_abs = float(golden_f.abs().max().item()) if golden_f.numel() else 0.0
    if max_abs <= 0.0 or not (max_abs == max_abs) or max_abs == float("inf"):
        return 0.0
    exponent = math.floor(math.log2(max_abs))
    return 32.0 * (2.0 ** (exponent - mantissa))


def resolve_max_error_cap(tolerance, dtype, golden):
    """The absolute error hard limit: max(fixed cap, 32 * ULP), or None."""
    if tolerance["max_error_cap"] is None:
        return None
    return max(tolerance["max_error_cap"], _ulp_scale_32(dtype, golden))


def check_tensor_sanity(sol_tensor, ref_tensor, allow_negative_inf=False):
    """Non-finite values or an all-zero output fail; returns a failure or None."""
    import torch

    ref_nonfinite = ~torch.isfinite(ref_tensor)
    sol_nonfinite = ~torch.isfinite(sol_tensor)
    if allow_negative_inf:
        both_neg_inf = (ref_tensor == float("-inf")) & (sol_tensor == float("-inf"))
        ref_nonfinite = ref_nonfinite & ~both_neg_inf
        sol_nonfinite = sol_nonfinite & ~both_neg_inf
    has_nonfinite = ref_nonfinite.any().item() or sol_nonfinite.any().item()
    if has_nonfinite:
        has_nan = (
            torch.isnan(sol_tensor).any().item()
            or torch.isnan(ref_tensor).any().item()
        )
        return {"has_nan": has_nan, "has_inf": not has_nan}
    ref_norm = torch.linalg.vector_norm(ref_tensor.to(torch.float32))
    if (
        ref_norm.item() > 0
        and torch.linalg.vector_norm(sol_tensor.to(torch.float32)).item() == 0
    ):
        abs_err = float(ref_norm.item())
        return {"max_absolute_error": abs_err, "max_relative_error": abs_err}
    return None


def compute_error_stats(output, reference, tolerance):
    """Return (stats, exceeds_tolerance) for one output tensor."""
    import torch

    tolerance = resolve_tolerance(tolerance, reference.dtype)
    x = output.to(torch.float32)
    y = reference.to(torch.float32)
    allow_neg_inf = tolerance["allow_negative_inf"]
    infs_nans = check_tensor_sanity(x, y, allow_negative_inf=allow_neg_inf)
    if infs_nans is not None:
        return dict(infs_nans, tolerance=tolerance), True
    if allow_neg_inf:
        both_neg_inf = (x == float("-inf")) & (y == float("-inf"))
        finite_mask = ~both_neg_inf
        x = x[finite_mask]
        y = y[finite_mask]
    abs_error = torch.abs(x - y)
    total_elements = abs_error.numel()
    if total_elements == 0:
        return {"tolerance": tolerance}, False
    max_abs = float(abs_error.max().item())
    tol_bound = tolerance["max_atol"] + tolerance["max_rtol"] * torch.abs(y)
    exceeds_tol_mask = (abs_error > tol_bound) | ~torch.isfinite(abs_error)
    exceeds_count = float(exceeds_tol_mask.sum().item())
    matched_ratio = 1.0 - (exceeds_count / float(total_elements))
    matched_ratio = max(0.0, min(1.0, matched_ratio))
    exceeds_tol = matched_ratio < tolerance["required_matched_ratio"]
    cap_limit = None
    if tolerance["max_error_cap"] is not None:
        cap_limit = resolve_max_error_cap(tolerance, reference.dtype, y)
        if cap_limit is not None and max_abs > cap_limit:
            exceeds_tol = True
    rel_error = abs_error / torch.clamp(torch.abs(y), min=tolerance["max_atol"])
    max_rel = float(rel_error.max().item())
    return {
        "max_absolute_error": max_abs,
        "max_relative_error": max_rel,
        "matched_ratio": matched_ratio,
        "max_error_limit": cap_limit,
        "tolerance": tolerance,
    }, exceeds_tol
