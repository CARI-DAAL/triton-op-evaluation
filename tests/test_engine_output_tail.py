"""The crash-tail contract: engine deaths carry their cause in the result."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from conftest import make_request

from triton_evaluation.api.contract import MAX_ERROR_MESSAGE_BYTES
from triton_evaluation.evaluators.base import (
    ENGINE_TAIL_LIMIT,
    ScriptOutcome,
    engine_output_tail,
)
from triton_evaluation.evaluators.benchmark import BenchmarkEvaluator
from triton_evaluation.evaluators.profile import ProfileEvaluator
from triton_evaluation.evaluators.verify import VerifyEvaluator


def _outcome(stderr: str = "", stdout: str = "") -> ScriptOutcome:
    return ScriptOutcome(exit_code=1, stdout=stdout, stderr=stderr, timed_out=False)


def test_empty_streams_produce_no_tail():
    assert engine_output_tail(_outcome()) == ""


def test_stderr_only_is_marked_and_bounded():
    tail = engine_output_tail(_outcome(stderr="EE9999: device assert"))
    assert tail.startswith("; [engine stderr tail] EE9999: device assert")
    assert "stdout" not in tail


def test_both_streams_stderr_first():
    tail = engine_output_tail(
        _outcome(stdout="progress line", stderr="RuntimeError: abort")
    )
    assert tail.index("stderr") < tail.index("stdout")


def test_tail_is_limited():
    tail = engine_output_tail(_outcome(stderr="x" * (ENGINE_TAIL_LIMIT * 4)))
    assert len(tail) <= ENGINE_TAIL_LIMIT + len("; [engine stderr tail] ") + 8


@pytest.mark.parametrize("limit", [0, 1, 40, 100, ENGINE_TAIL_LIMIT])
def test_combined_multibyte_streams_fit_the_byte_budget(limit):
    tail = engine_output_tail(_outcome(stderr="错误\n" * 2000, stdout="输出\r\n" * 2000), limit)
    assert len(tail.encode("utf-8")) <= limit
    assert "\n" not in tail and "\r" not in tail


@pytest.mark.parametrize(
    ("evaluator_type", "check"),
    [
        (VerifyEvaluator, "correctness"),
        (BenchmarkEvaluator, "performance"),
        (ProfileEvaluator, "profile"),
    ],
)
def test_evaluator_crashes_preserve_tail_under_the_public_message_limit(
    evaluator_type, check, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    def runner(*args, **kwargs):
        return _outcome(
            stderr="Traceback:\n" + "错误\n" * 2000 + "STDERR-FINAL",
            stdout="进度\r\n" * 2000 + "STDOUT-FINAL",
        )

    request = make_request(checks=sorted({"correctness", check}))
    deadline = datetime.now(timezone.utc) + timedelta(minutes=5)
    result = evaluator_type("engine", runner=runner).run(request, tmp_path / "run", 0, deadline)
    assert result.status == "failed"
    assert result.error.code == "EVALUATOR_EXECUTION_FAILED"
    assert result.error.check == check and result.error.retryable is True
    message = result.error.message
    assert "engine exit=1" in message
    assert "[engine stderr tail]" in message and "STDERR-FINAL" in message
    assert "[engine stdout tail]" in message and "STDOUT-FINAL" in message
    assert "\n" not in message and "\r" not in message
    assert len(message.encode("utf-8")) <= MAX_ERROR_MESSAGE_BYTES


@pytest.mark.parametrize("check", ["correctness", "performance", "profile"])
def test_engine_failure_messages_can_carry_the_tail(check):
    from triton_evaluation.evaluators.base import engine_failure

    result = engine_failure(
        check,
        f"{check} engine produced no usable result (engine exit=1)"
        f"{engine_output_tail(_outcome(stderr='abort reason'))}",
    )
    assert result.error is not None
    assert "[engine stderr tail] abort reason" in result.error.message
