"""Advisory Redis storage for measured PyTorch baseline case latencies."""

import hashlib
import json
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass

import redis

from triton_evaluation.serialization import canonical_json_bytes

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BaselineSnapshot:
    total_cases: int
    latencies: dict[int, float]


def _valid_total(value: object) -> bool:
    return type(value) is int and 1 <= value <= 128


def _valid_latency(value: object) -> bool:
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value) and value > 0
    )


class BaselineCache:
    def __init__(self, client, *, prefix: str, namespace: str = "torch-baseline-v1"):
        self._client = client
        self._prefix = prefix
        self._namespace = namespace

    def key_for(self, reference_code: str, *, method: str, warmup: int, repeats: int) -> str:
        protocol = {
            "namespace": self._namespace,
            "reference_sha256": hashlib.sha256(reference_code.encode("utf-8")).hexdigest(),
            "method": method,
            "warmup": warmup,
            "repeats": repeats,
            "sample_count": 1,
            "cv_threshold": 0.3,
            "clear_l2_cache": False,
        }
        digest = hashlib.sha256(canonical_json_bytes(protocol)).hexdigest()
        return f"{self._prefix}:baseline:{digest}"

    def read(self, key: str) -> BaselineSnapshot | None:
        try:
            record = self._client.hgetall(key)
            if not record:
                return None
            decoded = {
                name.decode() if isinstance(name, bytes) else name: json.loads(value)
                for name, value in record.items()
            }
            total = decoded.pop("total_cases", None)
            if not _valid_total(total):
                return None
            latencies = {}
            for field, value in decoded.items():
                if not isinstance(field, str) or not field.startswith("case:"):
                    return None
                index = int(field[5:])
                if field != f"case:{index}" or not 1 <= index <= total or not _valid_latency(value):
                    return None
                latencies[index] = float(value)
            return BaselineSnapshot(total, latencies)
        except Exception as error:  # noqa: BLE001 — advisory cache is fail-open
            logger.warning("baseline cache read failed (%s)", type(error).__name__)
            return None

    def publish(self, key: str, total_cases: int, latencies: Mapping[int, float]) -> None:
        if not _valid_total(total_cases):
            return
        try:
            values = dict(latencies)
            if any(
                type(index) is not int or not 1 <= index <= total_cases
                or not _valid_latency(value)
                for index, value in values.items()
            ):
                return
            if not values:
                return
            # WATCH binds metadata validation to the same transaction as case fields.
            for _ in range(8):
                try:
                    with self._client.pipeline() as pipe:
                        pipe.watch(key)
                        current = pipe.hget(key, "total_cases")
                        if current is not None:
                            total = json.loads(current)
                            if not _valid_total(total) or total != total_cases:
                                return
                        pipe.multi()
                        pipe.hsetnx(key, "total_cases", json.dumps(total_cases))
                        for index, value in values.items():
                            pipe.hsetnx(
                                key, f"case:{index}", json.dumps(float(value), allow_nan=False)
                            )
                        pipe.execute()
                        return
                except redis.WatchError:
                    continue
        except Exception as error:  # noqa: BLE001 — advisory cache is fail-open
            logger.warning("baseline cache write failed (%s)", type(error).__name__)
