from .verify import VerifyEvaluator
from .benchmark import BenchmarkEvaluator


def build_registry(engine_dir: str) -> dict:
    return {
        "verify": VerifyEvaluator(engine_dir),
        "benchmark": BenchmarkEvaluator(engine_dir),
    }
