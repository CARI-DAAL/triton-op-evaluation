"""NPU-only: real canary against a live card. Never runs in CI (no torch_npu)."""
import os

import pytest

pytestmark = pytest.mark.npu

health = pytest.importorskip("triton_evaluation.health", reason="health module")


def _npu_available() -> bool:
    try:
        import torch  # noqa: F401
        import torch_npu  # noqa: F401
        return bool(torch.npu.is_available())
    except Exception:
        return False


@pytest.mark.skipif(not _npu_available(), reason="no Ascend NPU visible")
def test_canary_passes_on_healthy_card():
    card = int(os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0").split(",")[0])
    assert health.run_canary(device_id=card, budget_seconds=120) is True
