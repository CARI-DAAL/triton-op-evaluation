from fastapi.testclient import TestClient

from triton_evaluation.api import create_app


class _RecordingBroker:
    """Minimal broker double to observe lifespan startup/shutdown hooks."""

    def __init__(self):
        self.pinged = 0
        self.closed = 0

    def ping(self):
        self.pinged += 1

    def close(self):
        self.closed += 1


def test_lifespan_pings_on_startup_and_closes_on_shutdown():
    broker = _RecordingBroker()
    app = create_app(broker)
    assert broker.pinged == 0
    assert broker.closed == 0

    with TestClient(app):
        assert broker.pinged == 1

    assert broker.closed == 1
