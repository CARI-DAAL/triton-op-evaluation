import sys
from pathlib import Path

from triton_evaluation.models import JobSpec
from triton_evaluation.evaluators.base import materialize, run_script, read_json, tail


def test_materialize_writes_named_files(tmp_path):
    spec = JobSpec(op_name="softmax", reference_code="REF", kernel_code="IMPL")
    materialize(spec, str(tmp_path))
    assert (tmp_path / "softmax_torch.py").read_text() == "REF"
    assert (tmp_path / "softmax_triton_ascend_impl.py").read_text() == "IMPL"


def test_run_script_captures_exit_and_env(tmp_path):
    script = tmp_path / "echo_env.py"
    script.write_text(
        "import os, sys\n"
        "sys.stdout.write(os.environ.get('ASCEND_RT_VISIBLE_DEVICES',''))\n"
        "sys.exit(7)\n"
    )
    outcome = run_script([sys.executable, str(script)], str(tmp_path), device_id=3, timeout=30)
    assert outcome.exit_code == 7
    assert outcome.stdout == "3"
    assert outcome.timed_out is False


def test_run_script_times_out(tmp_path):
    script = tmp_path / "sleep.py"
    script.write_text("import time\ntime.sleep(30)\n")
    outcome = run_script([sys.executable, str(script)], str(tmp_path), device_id=0, timeout=1)
    assert outcome.timed_out is True


def test_read_json_missing_returns_none(tmp_path):
    assert read_json(str(tmp_path / "nope.json")) is None


def test_tail_truncates():
    assert tail("x" * 5000, limit=100).endswith("x" * 50)
