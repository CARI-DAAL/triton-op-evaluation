"""Card health: canary probe and recovery ladder (spec §10).

The canary runs a vendored known-good mini kernel in-process under a SIGALRM
budget; the recovery ladder escalates in-process cleanup -> a configured
device-reset command -> full process restart via os.execv. Host reboot is
deliberately never automatic. All torch/triton imports are lazy: this module
only executes on NPU hosts.
"""
import gc
import logging
import os
import signal
import subprocess
import sys

logger = logging.getLogger(__name__)

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
    same timer fires during their sync window.
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
