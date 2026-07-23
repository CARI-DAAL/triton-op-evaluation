import fakeredis
import pytest

from triton_evaluation.broker import Broker
from triton_evaluation.config import Settings


@pytest.fixture
def settings(tmp_path):
    return Settings(engine_dir=str(tmp_path), dequeue_block_seconds=1)


@pytest.fixture
def broker(settings):
    client = fakeredis.FakeStrictRedis(decode_responses=True)
    return Broker(client, settings)
