from .verify import VerifyEvaluator


def build_registry(engine_dir: str) -> dict:
    return {
        "verify": VerifyEvaluator(engine_dir),
    }
