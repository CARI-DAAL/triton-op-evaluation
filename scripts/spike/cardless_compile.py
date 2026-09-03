#!/usr/bin/env python3
"""THROWAWAY spike: compile triton-ascend kernels with NO NPU device.

Strategies tried per kernel, in order (first success wins, all recorded):
  S1 warmup      — JITFunction.warmup(*dtypes, grid=(1,))  (triton's own
                   device-free compile path; MockTensor wrap, data_ptr()==0)
  S2 ast_source  — triton.compile(ASTSource(...), target=<explicit backend target>)

Exit 0: every module matched its corpus expectation AND every "ok" module
compiled via >=1 strategy. Exit 1: expectation mismatch. Exit 2: environment
violation (an NPU device is visible — this run proves nothing).
"""
import argparse
import importlib.metadata
import importlib.util
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus import EXTRA_OPTIONS, MODULES, SIGNATURES  # noqa: E402


def env_facts() -> dict:
    facts = {"davinci_nodes": sorted(p.name for p in Path("/dev").glob("davinci*"))}
    for dist in ("torch", "torch-npu", "triton", "triton-ascend"):
        try:
            facts[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            facts[dist] = "not-installed"
    try:
        import torch  # noqa: F401
        import torch_npu  # noqa: F401
        facts["npu_available"] = bool(torch.npu.is_available())
    except Exception as exc:  # import itself must not kill the probe
        facts["npu_available"] = f"import-error: {exc}"
    try:
        import triton.backends
        facts["triton_backends"] = sorted(triton.backends.backends.keys())
    except Exception as exc:
        facts["triton_backends"] = f"error: {exc}"
    return facts


def count_cache_files(cache_dir: Path) -> int:
    return sum(1 for _ in cache_dir.rglob("*") if _.is_file()) if cache_dir.exists() else 0


def build_args(spec):
    """Map signature spec to warmup-style args (torch dtypes for tensors)."""
    import torch

    args, kwargs = [], {}
    for name, kind in spec:
        family, _, value = kind.partition(":")
        if family == "tensor":
            args.append(getattr(torch, {"fp32": "float32", "fp16": "float16"}[value]))
        elif family == "scalar":
            args.append(4096 if value == "i32" else 2.0)  # 4096: divisible by 16
        elif family == "constexpr":
            kwargs[name] = int(value)
    return args, kwargs


def strategy_warmup(jit_fn, spec, extra):
    from triton.runtime.autotuner import Autotuner

    if isinstance(jit_fn, Autotuner):
        jit_fn = jit_fn.fn  # default config only; no sweep on CPU (spec §8)
    args, kwargs = build_args(spec)
    kwargs.update(extra)
    t0 = time.perf_counter()
    jit_fn.warmup(*args, grid=(1,), **kwargs)
    return time.perf_counter() - t0


def strategy_ast_source(jit_fn, spec, extra):
    import triton
    import triton.backends
    from triton.compiler import ASTSource
    from triton.runtime.autotuner import Autotuner

    if isinstance(jit_fn, Autotuner):
        jit_fn = jit_fn.fn
    candidates = [n for n in triton.backends.backends
                  if n.lower() not in ("cuda", "nvidia", "hip", "amd", "cpu")]
    if not candidates:
        raise RuntimeError("no non-CUDA triton backend registered")
    # `.driver` is the CLASS — instantiate it (its __init__ touches no device).
    # The arch must come from TRITON_ASCEND_ARCH: get_current_target() would
    # otherwise query the device, which cardless runs do not have.
    driver = triton.backends.backends[candidates[0]].driver()
    target = driver.get_current_target()
    signature, constexprs = {}, {}
    for _i, (name, kind) in enumerate(spec):
        family, _, value = kind.partition(":")
        if family == "tensor":
            signature[name] = {"fp32": "*fp32", "fp16": "*fp16"}[value]
        elif family == "scalar":
            signature[name] = "i32" if value == "i32" else "fp32"
        else:
            signature[name] = "constexpr"
            constexprs[name] = int(value)
    # triton 3.5 renamed the kwarg: constexprs -> constants.
    src = ASTSource(fn=jit_fn, signature=signature, constants=constexprs)
    t0 = time.perf_counter()
    triton.compile(src, target=target, options=dict(extra or {}))
    return time.perf_counter() - t0


STRATEGIES = [("warmup", strategy_warmup), ("ast_source", strategy_ast_source)]


def import_module_from(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_jit_fns(module) -> dict:
    out = {}
    try:
        from triton.runtime.autotuner import Autotuner
        from triton.runtime.jit import JITFunction
    except ImportError:
        return out
    for attr, obj in vars(module).items():
        if isinstance(obj, (JITFunction, Autotuner)):
            out[attr] = obj
    return out


def run_module(name, entry, workdir: Path, cache_dir: Path) -> dict:
    result = {"expect": entry["expect"], "stage": None, "ok": False, "attempts": []}
    src_path = workdir / f"{name}.py"
    src_path.write_text(entry["source"])
    try:
        module = import_module_from(name, src_path)
    except SyntaxError:
        result.update(stage="syntax", ok=entry["expect"] == "syntax")
        return result
    except Exception:
        result.update(stage="import", ok=entry["expect"] == "import",
                      error=traceback.format_exc(limit=3))
        return result

    jit_fns = find_jit_fns(module)
    if not jit_fns:
        result.update(stage="no_jit_fn", ok=False)
        return result

    all_ok = True
    for fn_name, jit_fn in jit_fns.items():
        spec = SIGNATURES.get(name, {}).get(fn_name)
        if spec is None:
            result["attempts"].append({"fn": fn_name, "strategy": None,
                                       "ok": False, "error": "no signature spec"})
            all_ok = False
            continue
        before = count_cache_files(cache_dir)
        compiled = False
        fn_attempts = []
        for strat_name, strat in STRATEGIES:
            attempt = {"fn": fn_name, "strategy": strat_name}
            try:
                attempt["wall_ms"] = round(strat(jit_fn, spec,
                                                 EXTRA_OPTIONS.get(name, {})) * 1e3, 2)
                attempt["ok"] = True
                compiled = True
            except Exception:
                attempt["ok"] = False
                attempt["error"] = traceback.format_exc(limit=5)
            fn_attempts.append(attempt)
        result["attempts"].extend(fn_attempts)
        result["compile_wall_ms"] = result.get("compile_wall_ms", 0.0) + sum(
            a.get("wall_ms", 0.0) for a in fn_attempts if a.get("ok"))
        result["cache_files_delta"] = count_cache_files(cache_dir) - before
        if entry["expect"] == "ok" and not compiled:
            all_ok = False
        if entry["expect"] in ("frontend", "any_compile") and compiled:
            all_ok = False  # failure corpus must NOT compile cleanly
    if entry["expect"] == "ok":
        result.update(stage="compiled", ok=all_ok)
    elif entry["expect"] in ("frontend", "any_compile"):
        result.update(stage="compile_failed", ok=all_ok)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/spike-cache")
    ap.add_argument("--report", default="/spike-report/cardless.json")
    ap.add_argument("--self-test", action="store_true",
                    help="corpus integrity only; no torch/triton imports")
    args = ap.parse_args()

    if args.self_test:
        import selftest
        return selftest.main()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir)  # before env_facts(): triton may import here

    # DISCOVERED WORKAROUND (spike purpose): triton-ascend 3.2.1's compile
    # pipeline fetches the SoC arch via the device driver
    # (NPUUtils().get_arch() -> rtGetSocVersion) even for pure compilation.
    # NPUUtils.__init__ reads TRITON_ASCEND_ARCH but never uses it. Shim the
    # class method so an explicitly supplied arch makes compilation cardless;
    # the bisheng compile subprocesses themselves need no device.
    _env_arch = os.getenv("TRITON_ASCEND_ARCH", "")
    if _env_arch:
        from triton.backends.ascend.driver import NPUUtils

        NPUUtils.get_arch = lambda self, _a=_env_arch: _a

    facts = env_facts()
    print(json.dumps({"env_facts": facts}, indent=2))
    if facts["davinci_nodes"] or facts["npu_available"] is True:
        print("ENV VIOLATION: an NPU device is visible; cardless result would be void")
        return 2

    report = {"env_facts": facts, "modules": {}}
    with tempfile.TemporaryDirectory(prefix="spike_") as work:
        for name, entry in MODULES.items():
            report["modules"][name] = run_module(name, entry, Path(work), cache_dir)

    report["all_ok"] = all(m["ok"] for m in report["modules"].values())
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=out.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(report, fh, indent=2)
    os.replace(tmp, out)
    print(json.dumps({n: {"stage": m["stage"], "ok": m["ok"],
                          "wall_ms": m.get("compile_wall_ms")}
                      for n, m in report["modules"].items()}, indent=2))
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
