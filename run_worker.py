"""Durable evaluation worker entrypoint: one process per NPU card."""
import argparse
import os
import signal

import redis

from triton_evaluation.broker import EvaluationStore
from triton_evaluation.config import Settings
from triton_evaluation.evaluators.evaluation import EvaluationEvaluator
from triton_evaluation.worker import Worker


def build_store(settings: Settings) -> EvaluationStore:
    client = redis.Redis.from_url(settings.redis_url, decode_responses=False)
    return EvaluationStore(client, settings)


def build_worker(card: int, settings: Settings | None = None) -> Worker:
    settings = settings or Settings.from_env()
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(card)
    store = build_store(settings)
    evaluator = EvaluationEvaluator(settings.engine_dir)
    return Worker(
        store, evaluator, device_id=card, settings=settings
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--card", type=int, required=True, help="Physical NPU card id to pin this worker to"
    )
    args = parser.parse_args()
    worker = build_worker(args.card)
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
