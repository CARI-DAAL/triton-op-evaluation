"""Decoy remains scoreable without materializing the official workload."""

import subprocess
import sys

import pytest

from triton_evaluation.evaluators.verify import _COUNT_SNIPPET


@pytest.mark.parametrize("count", [1, 79, 1046])
def test_official_decoy_count_never_materializes_inputs(tmp_path, count):
    reference = tmp_path / "reference.py"
    reference.write_text(
        "_ADAPTER_VERSION = 'official-task-v1'\n"
        f"def get_case_count(): return {count}\n"
        "def get_input_groups(): raise RuntimeError('all-case allocation forbidden')\n"
        "def get_input_group(index): raise RuntimeError('allocation forbidden')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", _COUNT_SNIPPET, str(reference)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == f'COUNT_RESULT:{{"total_cases": {count}}}\n'


def test_old_reference_preserves_input_group_counting(tmp_path):
    reference = tmp_path / "reference.py"
    reference.write_text("def get_input_groups(): return [[], [], []]\n")
    result = subprocess.run(
        [sys.executable, "-c", _COUNT_SNIPPET, str(reference)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == 'COUNT_RESULT:{"total_cases": 3}\n'
