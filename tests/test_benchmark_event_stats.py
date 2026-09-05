"""Event-timing statistics: median latency and CV stability gate (pure)."""
from triton_evaluation.engine import benchmark


def test_median_and_cv_basic():
    assert benchmark._median([1.0, 2.0, 3.0, 4.0]) == 2.5
    assert benchmark._median([5.0]) == 5.0
    assert round(benchmark._cv([1.0, 1.0, 1.0]), 6) == 0.0
    assert benchmark._cv([1.0, 2.0, 3.0]) > 0.3


def test_stable_gate_accepts_low_cv():
    assert benchmark._stable_latency([0.31, 0.30, 0.32, 0.31], cv_threshold=0.3) == 0.31


def test_stable_gate_rejects_high_cv():
    assert benchmark._stable_latency([0.10, 0.30, 0.90], cv_threshold=0.3) is None
