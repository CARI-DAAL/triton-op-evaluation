"""Card health: claim eligibility, canary probes and recovery (spec §10).

The canary runs a vendored known-good mini kernel in-process under a SIGALRM
budget; the recovery ladder escalates in-process cleanup -> a configured
device-reset command -> full process restart via os.execv. Host reboot is
deliberately never automatic. All torch/triton imports are lazy; hardware
actions only execute on NPU hosts. CardHealth owns the state and
retry timing; callers never coordinate individual recovery rungs.
"""
from __future__ import annotations

import gc
import logging
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .broker import EvaluationStore
    from .config import Settings
    from .metrics import RedisMetrics

logger = logging.getLogger(__name__)


class _State(Enum):
    HEALTHY = "healthy"
    SUSPECT = "suspect"
    DRAINED = "drained"


class CardHealth:
    """One worker's card health and recovery protocol.

    ``prepare_claim`` gates both heartbeats and claims. A drained worker's
    recovery iteration never claims: success admits the next iteration.
    ``mark_suspicious`` requires a fresh canary before any more work.

    Probe and recovery callables are the hardware seam. Redis publications
    must succeed before granting eligibility; metrics remain best effort.
    Only the worker owning this card calls this module, on its main thread.
    """

    def __init__(
        self,
        store: EvaluationStore,
        settings: Settings,
        *,
        device_id: int,
        clock: Callable[[], int] | None = None,
        canary: Callable[..., bool] | None = None,
        recovery_step: Callable[..., str] | None = None,
        metrics: RedisMetrics | None = None,
    ):
        self._store = store
        self._settings = settings
        self._device_id = device_id
        self._clock = clock or (lambda: int(time.monotonic() * 1000))
        self._canary = canary or run_canary
        self._recovery_step = recovery_step or run_recovery_step
        self._metrics = metrics
        self._state = _State.SUSPECT if settings.card_canary_on_startup else _State.HEALTHY
        self._last_probe_ms: int | None = None
        self._last_recovery_ms: int | None = None

    def mark_suspicious(self) -> None:
        """Require a canary; repeated reports cannot bypass drain cooldowns."""
        if self._state is _State.HEALTHY:
            self._state = _State.SUSPECT

    def prepare_claim(self) -> bool:
        """Advance health recovery and admit heartbeats/claims when ready."""
        if self._state is _State.HEALTHY:
            return True
        was_drained = self._state is _State.DRAINED
        if (
            was_drained
            and self._last_probe_ms is not None
            and self._clock() - self._last_probe_ms
            < self._settings.canary_interval_seconds * 1000
        ):
            return False
        if self._probe():
            return not was_drained
        self._recover()
        return False

    def _probe(self) -> bool:
        self._last_probe_ms = self._clock()
        try:
            healthy = self._canary(
                device_id=self._device_id,
                budget_seconds=self._settings.canary_budget_seconds,
            )
        except Exception as error:  # noqa: BLE001 — unavailable is not healthy
            logger.warning(
                "card %s canary failed (%s)", self._device_id, type(error).__name__
            )
            healthy = False
        self._bump("card_canary_total", outcome="pass" if healthy else "fail")
        if healthy:
            self._store.clear_card_unhealthy(self._device_id)
            self._state = _State.HEALTHY
        else:
            # Fail closed locally even if publishing the unhealthy marker fails.
            self._state = _State.DRAINED
            self._store.set_card_unhealthy(self._device_id)
        if self._metrics is not None:
            try:
                self._metrics.set_gauge(
                    "card_health_state", {"card": str(self._device_id)},
                    value=0 if healthy else 1,
                )
            except Exception as error:  # noqa: BLE001 — metrics cannot gate recovery
                logger.warning("metrics write failed (%s)", type(error).__name__)
        return healthy

    def _recover(self) -> None:
        now_ms = self._clock()
        if (
            self._last_recovery_ms is not None
            and now_ms - self._last_recovery_ms
            < self._settings.card_recovery_retry_seconds * 1000
        ):
            return
        self._last_recovery_ms = now_ms
        for step in range(1, self._settings.card_recovery_max_step + 1):
            try:
                name = self._recovery_step(
                    step, card=self._device_id,
                    reset_command=self._settings.card_reset_command,
                )
            except Exception as error:  # noqa: BLE001 — a failed rung escalates
                logger.warning(
                    "card %s recovery step %s failed (%s)",
                    self._device_id, step, type(error).__name__,
                )
                name = None
            if name is not None:
                self._bump("card_recovery_total", step=name)
            if self._probe():
                return

    def _bump(self, name: str, **labels) -> None:
        if self._metrics is None:
            return
        try:
            self._metrics.increment(name, labels)
        except Exception as error:  # noqa: BLE001 — metrics cannot gate recovery
            logger.warning("metrics write failed (%s)", type(error).__name__)


CANARY_ADD_KERNEL = '''
import torch
import triton
import triton.language as tl

@triton.jit
def _canary_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)

class ModelNew(torch.nn.Module):
    def forward(self, x, y):
        output = torch.empty_like(x)
        n = x.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        _canary_add_kernel[grid](x, y, output, n, BLOCK_SIZE=1024)
        return output
'''


class CanaryBudgetExceeded(TimeoutError):
    pass


def _cleanup_in_process() -> None:
    import torch  # noqa: F401
    import torch_npu  # noqa: F401

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.synchronize()


def run_canary(*, device_id: int, budget_seconds: int) -> bool:
    """Run the known-good mini kernel on this worker's card. True = healthy.

    A hang raises via SIGALRM and is reported as unhealthy; the bisheng
    subprocesses launched by the kernel never outlive the budget because the
    same timer fires during their sync window. Caveat: SIGALRM cannot fire its
    Python handler while the main thread is inside an uninterruptible driver
    C call, so a fully wedged driver may outlive the budget — the system
    backstop is lease expiry -> reaper -> suspicious-death counting -> poison
    quarantine. device_id is provenance/logging only: device selection happens
    via ASCEND_RT_VISIBLE_DEVICES pinning, so the code sets device 0.
    """
    import torch  # noqa: F401
    import torch_npu  # noqa: F401

    def _handler(signum, frame):
        raise CanaryBudgetExceeded(f"canary exceeded {budget_seconds}s budget")

    previous_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, budget_seconds)
    try:
        torch.npu.set_device(0)  # pinned to the card via ASCEND_RT_VISIBLE_DEVICES
        namespace: dict = {}
        exec(compile(CANARY_ADD_KERNEL, "<canary>", "exec"), namespace)
        model = namespace["ModelNew"]().npu()
        x = torch.randn(1024, device="npu")
        y = torch.randn(1024, device="npu")
        with torch.no_grad():
            out = model(x, y)
        torch.npu.synchronize()
        return bool(torch.equal(out, x + y))
    except Exception as error:  # noqa: BLE001 — any failure means unhealthy
        logger.warning("card %s canary failed (%s)", device_id, type(error).__name__)
        return False
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def run_recovery_step(step: int, *, card: int, reset_command: str) -> str:
    """Execute one ladder rung; returns its name. Step 3 replaces the process
    image and never returns. A missing reset command makes rung 2 a named
    no-op so callers can keep rung numbering stable."""
    if step == 1:
        _cleanup_in_process()
        return "cleanup"
    if step == 2:
        if reset_command:
            command = reset_command.replace("{card}", str(card))
            subprocess.run(command, shell=True, timeout=120, check=False,
                           capture_output=True)
        return "reset"
    os.execv(sys.executable, [sys.executable, *sys.argv])
    raise AssertionError("unreachable after execv")
