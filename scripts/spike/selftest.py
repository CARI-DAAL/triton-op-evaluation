#!/usr/bin/env python3
"""THROWAWAY spike: corpus integrity self-test. Stdlib only; runs on any machine."""
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    try:
        from corpus import EXTRA_OPTIONS, MODULES, SIGNATURES
    except ImportError as exc:
        print(f"FAIL: cannot import corpus: {exc}")
        return 1

    failures = []
    for name, entry in MODULES.items():
        src, expect = entry["source"], entry["expect"]
        try:
            tree = ast.parse(src)
        except SyntaxError:
            if expect != "syntax":
                failures.append(f"{name}: unexpected SyntaxError (expect={expect})")
            continue
        if expect == "syntax":
            failures.append(f"{name}: expected SyntaxError but parsed cleanly")
            continue
        if expect == "ok":
            if "ModelNew" not in src:
                failures.append(f"{name}: ok module missing ModelNew")
            if "@triton.jit" not in src:
                failures.append(f"{name}: ok module missing @triton.jit")
        fn_args = {
            n.name: {a.arg for a in n.args.args}
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef)
        }
        for fn_name, spec in SIGNATURES.get(name, {}).items():
            if fn_name not in fn_args:
                failures.append(f"{name}: signature spec for unknown fn {fn_name}")
                continue
            for arg_name, kind in spec:
                if arg_name not in fn_args[fn_name]:
                    failures.append(f"{name}.{fn_name}: spec arg {arg_name} not in def")
                if kind.startswith("constexpr:") and not kind.split(":")[1].isdigit():
                    failures.append(f"{name}.{fn_name}: bad constexpr kind {kind}")
    for name in EXTRA_OPTIONS:
        if name not in MODULES:
            failures.append(f"EXTRA_OPTIONS references unknown module {name}")

    for msg in failures:
        print(f"FAIL: {msg}")
    print(f"selftest: {len(MODULES)} modules, {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
