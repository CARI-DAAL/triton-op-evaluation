import pytest
import fakeredis

from triton_evaluation.config import Settings
from triton_evaluation.broker import Broker


@pytest.fixture
def settings(tmp_path):
    return Settings(engine_dir=str(tmp_path), dequeue_block_seconds=1)


@pytest.fixture
def broker(settings):
    client = fakeredis.FakeStrictRedis(decode_responses=True)
    return Broker(client, settings)
