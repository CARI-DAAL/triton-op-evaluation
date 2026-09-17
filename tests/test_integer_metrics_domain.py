"""Integer-compute metrics must stay inside the RFC 8785 number domain."""

import json

from triton_evaluation.api.contract import CaseMetrics


def test_max_abs_diff_beyond_safe_domain_serializes():
    """A garbage int64 kernel yields diffs up to 9.2e18; the metric must be a
    double so canonical serialization cannot raise IntegerDomainError."""
    import rfc8785

    metrics = CaseMetrics(max_abs_diff=9.0e18)
    raw = rfc8785.dumps(json.loads(metrics.model_dump_json()))
    assert b"9" in raw  # serialized without IntegerDomainError


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
