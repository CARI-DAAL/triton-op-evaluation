import argparse
import os

from triton_evaluation.config import Settings
from triton_evaluation.broker import Broker
from triton_evaluation.evaluators import build_registry
from triton_evaluation.worker import Worker


def build_worker(card: int) -> Worker:
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(card)
    settings = Settings.from_env()
    broker = Broker.from_settings(settings)
    registry = build_registry(settings.skill_scripts_dir)
    return Worker(broker, registry, device_id=card, settings=settings)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", type=int, required=True, help="Physical NPU card id to pin this worker to")
    args = ap.parse_args()
    build_worker(args.card).serve_forever()


if __name__ == "__main__":
    main()
