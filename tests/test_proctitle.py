"""Verify set_process_title renames the kernel comm (the field npu-smi reads).

npu-smi info displays /proc/<pid>/comm. These tests read that same file to
confirm the rename works and respects the 15-byte kernel cap.
"""
import os

from triton_evaluation.engine._common_utils import set_process_title


def _comm() -> str:
    with open(f"/proc/{os.getpid()}/comm", encoding="utf-8") as f:
        return f.read().rstrip("\n")


def test_set_process_title_renames_comm():
    original = _comm()
    try:
        set_process_title("TritonEvalWork")
        assert _comm() == "TritonEvalWork"
    finally:
        # restore so the rename doesn't leak into other tests in this process
        set_process_title(original)


def test_set_process_title_truncates_to_15_bytes():
    original = _comm()
    try:
        set_process_title("TritonEvalWorker")  # 16 chars
        # kernel TASK_COMM_LEN caps comm at 15 usable bytes
        assert _comm() == "TritonEvalWorke"
        assert len(_comm()) <= 15
    finally:
        set_process_title(original)
