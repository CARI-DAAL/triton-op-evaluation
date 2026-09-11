"""Admission pre-check subprocess entry (spec §8, Plan D-on_card).

Classifies a candidate module without any NPU: syntax parse, then import, then
input-case counting. Never launches kernels. Always exits 0; the verdict is the
last stdout line as JSON. Stdlib-only so it runs in any worker/API env.
"""
import ast
import importlib.util
import json
import sys

# Missing runtime deps say nothing about the candidate: a torch-less host must
# not turn a good module into a false compile_error verdict.
_INCONCLUSIVE_MISSING = frozenset({"torch", "torch_npu", "triton"})


def main() -> int:
    module_path, impl_name = sys.argv[1], sys.argv[2]
    try:
        source = open(module_path, encoding="utf-8").read()
    except OSError as error:
        _emit({"stage": "import", "error": f"unreadable module: {error}"})
        return 0
    try:
        ast.parse(source)
    except SyntaxError as error:
        _emit({"stage": "syntax", "error": f"SyntaxError: {error}"})
        return 0
    try:
        spec = importlib.util.spec_from_file_location(f"{impl_name}_candidate", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as error:  # noqa: BLE001 — any import failure is a verdict
        missing = getattr(error, "name", None)
        if isinstance(error, ModuleNotFoundError) and missing in _INCONCLUSIVE_MISSING:
            _emit({"stage": None, "total_cases": 0, "inconclusive": True})
            return 0
        _emit({"stage": "import", "error": f"{type(error).__name__}: {error}"[:400]})
        return 0
    total_cases = 0
    try:
        if hasattr(module, "get_input_groups"):
            total_cases = len(module.get_input_groups())
        elif hasattr(module, "get_inputs"):
            total_cases = 1
    except Exception:  # noqa: BLE001 — case counting is best effort
        total_cases = 0
    _emit({"stage": None, "total_cases": total_cases})
    return 0


def _emit(verdict: dict) -> None:
    print("VERDICT " + json.dumps(verdict))


if __name__ == "__main__":
    sys.exit(main())
