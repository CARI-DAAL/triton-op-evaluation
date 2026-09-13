"""Attempt-owned syntax/import subprocess entry.

Classifies a candidate module with syntax parsing and import. It never calls
candidate input factories. Import can execute arbitrary candidate code, so the
worker owns the process group's lease and deadline. Stdlib-only.
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
    except OSError:
        _emit({"stage": None, "inconclusive": True})
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
    except Exception as error:  # noqa: BLE001 — import executes candidate code
        missing = getattr(error, "name", None)
        missing_module = (
            isinstance(error, ModuleNotFoundError)
            and missing is not None
            and missing not in _INCONCLUSIVE_MISSING
        )
        missing_symbol = (
            isinstance(error, ImportError)
            and missing is not None
            and str(error).startswith("cannot import name ")
        )
        # Arbitrary import-time code can fail due to temporary I/O, a native
        # runtime loader, or unavailable hardware. Only import resolution
        # failures establish a negative verdict within this toolchain scope.
        if not missing_module and not missing_symbol:
            _emit({"stage": None, "inconclusive": True})
            return 0
        _emit({"stage": "import", "error": f"{type(error).__name__}: {error}"[:400]})
        return 0
    _emit({"stage": None})
    return 0


def _emit(verdict: dict) -> None:
    print("VERDICT " + json.dumps(verdict))


if __name__ == "__main__":
    sys.exit(main())
