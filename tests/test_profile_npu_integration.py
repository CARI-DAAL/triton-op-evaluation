import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).parent
sys.path.insert(0, str(TESTS_DIR))
from msprof_smoke_kernels import KERNEL_CODE, REFERENCE_CODE  # noqa: E402

pytestmark = pytest.mark.skipif(
    os.environ.get("TRITON_EVAL_RUN_NPU_PROFILE_TESTS") != "1",
    reason="set TRITON_EVAL_RUN_NPU_PROFILE_TESTS=1 on an NPU host",
)


def _materialize(tmp_path):
    (tmp_path / "add_torch.py").write_text(REFERENCE_CODE, encoding="utf-8")
    (tmp_path / "add_triton_ascend_impl.py").write_text(
        KERNEL_CODE,
        encoding="utf-8",
    )


def _run(tmp_path, mode, case_idx):
    output = tmp_path / f"{mode}_{case_idx}.json"
    engine = (
        Path(__file__).parents[1]
        / "triton_evaluation"
        / "engine"
        / "profile.py"
    )
    command = [
        sys.executable,
        str(engine),
        "--op_name",
        "add",
        "--verify_dir",
        str(tmp_path),
        "--triton_impl_name",
        "triton_ascend_impl",
        "--profile-mode",
        mode,
        "--timeout",
        "7200",
        "--output",
        str(output),
    ]
    if case_idx is not None:
        command.extend(["--case-idx", str(case_idx)])
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        timeout=7260,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    return json.loads(output.read_text())


def test_real_msprof_op_all_and_single_case(tmp_path):
    assert shutil.which(os.environ.get("MSPROF_BIN", "msprof"))
    _materialize(tmp_path)
    all_cases = _run(tmp_path, "op", None)
    assert all_cases["selected_cases"] == 2
    assert all(
        case["target_kernel"].startswith("add_kernel")
        for case in all_cases["per_case_results"]
    )
    one_case = _run(tmp_path, "op", 1)
    assert one_case["selected_cases"] == 1
    assert one_case["per_case_results"][0]["case_idx"] == 1


def test_real_msprof_deep_single_case(tmp_path):
    assert shutil.which(os.environ.get("MSPROF_BIN", "msprof"))
    _materialize(tmp_path)
    result = _run(tmp_path, "deep", 0)
    case = result["per_case_results"][0]
    assert case["target_kernel"].startswith("add_kernel")
    assert case["status"] in {"complete", "partial"}
    assert case["error"] is None
