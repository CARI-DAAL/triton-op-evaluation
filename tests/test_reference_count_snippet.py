"""The decoy fast-path count snippet must count NPU-only references.

The AST pre-check proves a decoy before the engine ever runs, and the decoy
verdict must stay scoreable as zero. Its ``total_cases`` comes from a bare
``python -c`` snippet importing the reference and calling
``get_input_groups()`` — which, for references declaring inputs with
NPU-only op/dtype pairs (integer ``torch.randn``), raises
NotImplementedError on a CPU-defaulted process. The snippet retries in the
NPU default-device context instead of failing the whole check.
"""

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

from triton_evaluation.evaluators.verify import _COUNT_SNIPPET

_HAS_TORCH_NPU = importlib.util.find_spec("torch_npu") is not None


def _run_snippet(tmp_path: Path, reference_source: str) -> subprocess.CompletedProcess:
    reference = tmp_path / "reference_torch.py"
    reference.write_text(textwrap.dedent(reference_source), encoding="utf-8")
    return subprocess.run(
        [sys.executable, "-c", _COUNT_SNIPPET, str(reference)],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_snippet_is_valid_python():
    compile(_COUNT_SNIPPET, "<count-snippet>", "exec")


def test_snippet_counts_plain_reference(tmp_path):
    result = _run_snippet(
        tmp_path,
        """
        def get_input_groups():
            return [[1], [2], [3]]
        """,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {"total_cases": 3}


def test_snippet_counts_get_inputs_reference_as_one_case(tmp_path):
    result = _run_snippet(
        tmp_path,
        """
        def get_inputs():
            return [1]
        """,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {"total_cases": 1}


def test_snippet_without_provider_exits_nonzero(tmp_path):
    result = _run_snippet(tmp_path, "x = 1\n")
    assert result.returncode != 0


def test_snippet_fails_when_retry_also_raises(tmp_path):
    """A reference whose construction fails for a non-NPU reason must still
    fail loudly — the retry only covers the missing-CPU-kernel signal."""
    result = _run_snippet(
        tmp_path,
        """
        def get_input_groups():
            raise NotImplementedError(" genuinely broken reference ")
        """,
    )
    assert result.returncode != 0
    if _HAS_TORCH_NPU:
        # On a torch_npu host the retry runs and re-raises the original
        # error; without torch the retry import fails — either way, loud.
        assert "genuinely broken reference" in result.stderr
