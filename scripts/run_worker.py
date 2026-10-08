"""Durable evaluation worker entrypoint: CPU prechecks or one pinned NPU."""
import argparse
import logging
import os
import signal

import redis

from triton_evaluation.config import Settings
from triton_evaluation.evaluators.evaluation import EvaluationEvaluator
from triton_evaluation.runtime.worker import Worker
from triton_evaluation.service.broker import EvaluationStore

logger = logging.getLogger(__name__)

# Process title shown in `npu-smi info` / ps; 15-char /proc/<pid>/comm limit.
_PROCESS_TITLE = "EvalWorker"

# Keeps the startup device-hold tensor alive for the process lifetime.
_DEVICE_HOLD = None


def set_process_title() -> None:
    """Rename the worker so npu-smi shows `EvalWorker` instead of `python`.
    Best effort: missing setproctitle only keeps the default name."""
    try:
        from setproctitle import setproctitle

        setproctitle(_PROCESS_TITLE)
    except ImportError:
        logger.info("setproctitle not installed; process keeps the default name")


def build_store(settings: Settings) -> EvaluationStore:
    client = redis.Redis.from_url(settings.redis_url, decode_responses=False)
    return EvaluationStore(client, settings)


def build_worker(card: int | None, settings: Settings | None = None) -> Worker:
    settings = settings or Settings.from_env()
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "" if card is None else str(card)
    store = build_store(settings)
    evaluator = EvaluationEvaluator(settings.engine_dir, settings=settings)
    return Worker(
        store, evaluator, device_id=card, settings=settings
    )


def hold_device(settings: Settings, card: int) -> None:
    """Open the pinned NPU and keep a small allocation alive so `npu-smi info`
    shows this worker continuously, not only while a job's engine subprocess
    is running. Best effort: a failure is logged and serving continues.
    """
    global _DEVICE_HOLD
    if settings.device_hold_mb <= 0:
        return
    try:
        import torch
        import torch_npu  # noqa: F401 — registers the NPU backend on import

        # ASCEND_RT_VISIBLE_DEVICES pins this process to one physical card,
        # which becomes logical device 0.
        torch.npu.set_device(0)
        _DEVICE_HOLD = torch.empty(
            settings.device_hold_mb * 1024 * 1024,  # MB → bytes (uint8 elements)
            dtype=torch.uint8,
            device="npu",
        )
        logger.info(
            "holding %d MB on card %d for npu-smi visibility",
            settings.device_hold_mb, card,
        )
    except Exception as error:  # noqa: BLE001 — visibility is best effort
        _DEVICE_HOLD = None
        logger.warning(
            "device hold failed (%s); worker serves but npu-smi may not "
            "show it while idle",
            type(error).__name__,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    execution = parser.add_mutually_exclusive_group(required=True)
    execution.add_argument(
        "--card", type=int, help="Physical NPU card id to pin this worker to"
    )
    execution.add_argument("--cpu", action="store_true", help="Execute compile-only attempts")
    args = parser.parse_args()
    if args.card is not None and args.card < 0:
        parser.error("--card must be nonnegative")
    set_process_title()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    worker = build_worker(args.card)
    if args.card is not None:
        hold_device(worker.settings, args.card)
    stop = _install_signal_handlers()
    worker.serve_forever(stop_event=stop)


def _install_signal_handlers():
    import threading

    stop = threading.Event()

    def _handle(signum, frame):
        stop.set()

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)
    return stop


if __name__ == "__main__":
    main()
