import pytest

from triton_evaluation.engine import benchmark


def test_benchmark_cli_cannot_bypass_verify_gate():
    parser = benchmark._build_argparser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(
            [
                "--op_name",
                "softmax",
                "--verify_not_required",
            ]
        )

    assert exc_info.value.code == 2
