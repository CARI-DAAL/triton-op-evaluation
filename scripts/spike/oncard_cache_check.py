#!/usr/bin/env python3
"""THROWAWAY spike: on-card cache-reuse measurement.

Parent re-execs one child per (module, cache-dir) so TRITON_CACHE_DIR is set
before triton initializes. hit := warm run adds ~no cache files AND is much
faster than the cold run (recompile avoided on card).
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

HIT_MAX_NEW_FILES = 2      # metadata churn tolerance
HIT_MAX_TIME_RATIO = 0.6   # warm first-forward < 60% of cold


def count_files(d: Path) -> int:
    return sum(1 for _ in d.rglob("*") if _.is_file()) if d.exists() else 0


def child(module_name: str, workdir: Path) -> dict:
    """Runs with TRITON_CACHE_DIR already in env (set by parent pre-exec)."""
    import torch  # noqa: F401
    import torch_npu  # noqa: F401
    from corpus import MODULES

    cache_dir = Path(os.environ["TRITON_CACHE_DIR"])
    src = workdir / f"{module_name}.py"
    src.write_text(MODULES[module_name]["source"])
    spec = importlib.util.spec_from_file_location(module_name, src)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model = module.ModelNew().npu()
    inputs = [[x.npu() if isinstance(x, torch.Tensor) else x
               for x in group] for group in module.get_input_groups()][0]
    before = count_files(cache_dir)
    t0 = time.perf_counter()
    with torch.no_grad():
        model(*inputs)
    torch.npu.synchronize()
    wall_ms = (time.perf_counter() - t0) * 1e3
    return {"first_forward_ms": round(wall_ms, 2),
            "new_cache_files": count_files(cache_dir) - before}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/spike-cache")
    ap.add_argument("--cold-cache-dir", default="/spike-cold-cache")
    ap.add_argument("--report", default="/spike-report/oncard_cache.json")
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--module")
    args = ap.parse_args()

    if args.child:
        with tempfile.TemporaryDirectory(prefix="spike_oncard_") as work:
            print("RESULT " + json.dumps(child(args.module, Path(work))))
        return 0

    from corpus import MODULES

    ok_modules = [n for n, e in MODULES.items() if e["expect"] == "ok"]
    report = {"modules": {}}
    for name in ok_modules:
        row = {}
        for label, cdir in (("warm", args.cache_dir), ("cold", args.cold_cache_dir)):
            env = dict(os.environ, TRITON_CACHE_DIR=cdir)
            Path(cdir).mkdir(parents=True, exist_ok=True)
            proc = subprocess.run(
                [sys.executable, __file__, "--child", "--module", name],
                env=env, capture_output=True, text=True, timeout=900)
            line = next((ln for ln in proc.stdout.splitlines()
                         if ln.startswith("RESULT ")), None)
            if proc.returncode != 0 or line is None:
                row[label] = {"error": (proc.stderr or proc.stdout)[-500:]}
                continue
            data = json.loads(line[len("RESULT "):])
            row[f"{label}_ms"] = data["first_forward_ms"]
            row[f"{label}_new_files"] = data["new_cache_files"]
        if "warm_ms" in row and "cold_ms" in row:
            row["hit"] = (row["warm_new_files"] <= HIT_MAX_NEW_FILES
                          and row["warm_ms"] < HIT_MAX_TIME_RATIO * row["cold_ms"])
        report["modules"][name] = row

    hits = [m.get("hit") for m in report["modules"].values() if "hit" in m]
    report["hit_rate"] = round(sum(hits) / len(hits), 3) if hits else 0.0
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd_tmp = out.with_suffix(".tmp")
    fd_tmp.write_text(json.dumps(report, indent=2))
    fd_tmp.replace(out)
    print(json.dumps(report, indent=2))
    return 0  # report-only; the gate decision is human (Task 5)


if __name__ == "__main__":
    sys.exit(main())
