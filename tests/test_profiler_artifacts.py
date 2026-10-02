"""Failed profiler evidence retention, independent of the device timing defect."""

import pytest

from triton_evaluation.engine import benchmark


@pytest.mark.parametrize("corrupt_csv", [False, True])
def test_keep_res_preserves_failed_capture_for_diagnosis(tmp_path, corrupt_csv):
    capture = tmp_path / "capture"
    capture.mkdir()
    record = capture / "raw_record.bin"
    record.write_bytes(b"original failed profiler capture")
    if corrupt_csv:
        (capture / "kernel_details.csv").write_bytes(b"")
    assert benchmark.parse_operator_latency(str(capture), 5, 20, keep_res=True) == (None, None)
    assert record.read_bytes() == b"original failed profiler capture"


def test_default_cleanup_behavior_remains_explicit(tmp_path):
    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "raw_record.bin").write_bytes(b"not requested for retention")
    assert benchmark.parse_operator_latency(str(capture), 5, 20, keep_res=False) == (None, None)
    assert not capture.exists()
