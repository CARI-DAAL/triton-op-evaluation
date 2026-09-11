"""NPU mechanics for the event-timing path (torch.npu.Event per-iteration).

Proves only that the Event mechanics work on real hardware: a trivial op on
device ``npu`` yields a positive stable latency. The CV gate itself is pure
and covered by tests/test_benchmark_event_stats.py.
"""
import os

import pytest

torch_npu = pytest.importorskip(
    "torch_npu", reason="torch_npu is required for event-timing NPU tests"
)

pytestmark = [
    pytest.mark.npu,
    pytest.mark.skipif(
        os.environ.get("TRITON_EVAL_RUN_NPU_PROFILE_TESTS") != "1",
        reason="set TRITON_EVAL_RUN_NPU_PROFILE_TESTS=1 on an NPU host",
    ),
]

from triton_evaluation.engine import benchmark  # noqa: E402


def test_measure_event_yields_positive_stable_latency():
    import torch

    device = torch.device("npu")
    x = torch.randn(64, 64, device=device)

    def test_fn():
        with torch.no_grad():
            _ = x.relu()

    latency_ms = benchmark._measure_event(test_fn, warmup=2, repeats=10, cv_threshold=0.3)
    assert latency_ms is not None
    assert latency_ms > 0
