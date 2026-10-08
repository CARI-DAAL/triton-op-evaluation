"""Worker command line selects one execution resource without opening another."""
from types import SimpleNamespace
from unittest import mock

import pytest

from scripts import run_worker


def test_cpu_cli_never_holds_an_npu(monkeypatch):
    worker = SimpleNamespace(settings=object(), serve_forever=mock.Mock())
    build = mock.Mock(return_value=worker)
    monkeypatch.setattr("sys.argv", ["run_worker.py", "--cpu"])
    monkeypatch.setattr(run_worker, "build_worker", build)
    monkeypatch.setattr(run_worker, "hold_device", mock.Mock(side_effect=AssertionError))
    monkeypatch.setattr(run_worker, "set_process_title", lambda: None)
    monkeypatch.setattr(run_worker, "_install_signal_handlers", lambda: "stop")
    run_worker.main()
    build.assert_called_once_with(None)
    worker.serve_forever.assert_called_once_with(stop_event="stop")


@pytest.mark.parametrize("options", [[], ["--cpu", "--card", "0"], ["--card", "-1"]])
def test_worker_cli_requires_exactly_one_valid_execution_resource(monkeypatch, options):
    monkeypatch.setattr("sys.argv", ["run_worker.py", *options])
    with pytest.raises(SystemExit) as error:
        run_worker.main()
    assert error.value.code == 2
