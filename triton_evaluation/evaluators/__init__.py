from .verify import VerifyEvaluator
from .benchmark import BenchmarkEvaluator


def build_registry(scripts_dir: str) -> dict:
    return {
        "verify": VerifyEvaluator(scripts_dir),
        "benchmark": BenchmarkEvaluator(scripts_dir),
    }
