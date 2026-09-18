"""Integer-compute metrics must stay inside the RFC 8785 number domain."""

import importlib.util
import json

import pytest

from triton_evaluation.api.contract import CaseMetrics

# CI installs only [test,dev]; the engine-comparison test needs real torch
# tensors and runs wherever the NPU release gate or a torch-equipped host does.
_has_torch = importlib.util.find_spec("torch") is not None


def test_max_abs_diff_beyond_safe_domain_serializes():
    """A garbage int64 kernel yields diffs up to 9.2e18; the metric must be a
    double so canonical serialization cannot raise IntegerDomainError."""
    import rfc8785

    metrics = CaseMetrics(max_abs_diff=9.0e18)
    raw = rfc8785.dumps(json.loads(metrics.model_dump_json()))
    # 9e18 < 1e21 serializes in plain digit form; the call itself must not raise.
    assert b"9000000000000000000" in raw


@pytest.mark.skipif(not _has_torch, reason="engine comparison test needs torch")
def test_engine_reports_max_abs_diff_as_double():
    """The integer-compute path converts the int64 tensor max to float."""
    from unittest.mock import MagicMock

    import torch

    from triton_evaluation.engine.verify import _compare_integer_compute

    garbage = torch.tensor([-4899916394579100000], dtype=torch.int64)
    reference = torch.tensor([1], dtype=torch.int64)
    ctx = MagicMock()
    ctx.data_type = "int64"
    ctx.input_type = "int"
    try:
        _compare_integer_compute(reference, garbage, ctx)
    except Exception as error:
        metrics = getattr(error, "metrics", {})
        assert isinstance(metrics.get("max_abs_diff"), float)
        assert metrics["max_abs_diff"] > 2**53
    else:
        raise AssertionError("integer mismatch must raise AccuracyError")


@pytest.mark.skipif(not _has_torch, reason="engine comparison test needs torch")
def test_engine_quant_path_reports_max_abs_diff_as_double():
    """The quant fp-to-int path converts the int64 tensor max to float too."""
    from unittest.mock import MagicMock

    import torch

    from triton_evaluation.engine.verify import _compare_quant_fp_to_int

    garbage = torch.tensor([-4899916394579100000], dtype=torch.int64)
    reference = torch.tensor([1.25], dtype=torch.float32)
    ctx = MagicMock()
    ctx.data_type = "int32"
    ctx.input_type = "float"
    try:
        _compare_quant_fp_to_int(reference, garbage, ctx)
    except Exception as error:
        metrics = getattr(error, "metrics", {})
        assert isinstance(metrics.get("max_abs_diff"), float)
        assert metrics["max_abs_diff"] > 2**53
    else:
        raise AssertionError("quant mismatch must raise AccuracyError")
