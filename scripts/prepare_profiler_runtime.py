#!/usr/bin/env python3
"""Seal an existing SQLite ELF as a private CANN profiler runtime alias.

This creates a fresh service-owned immutable directory. It does not install,
load or replace a library, change ldconfig, modify system paths, or measure NPU
work. Deploy its manifest/env pair and read-only mount before device acceptance.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from triton_evaluation.profiler_runtime import prepare_runtime, sha  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sqlite-library", required=True, type=Path)
    parser.add_argument("--sqlite-library-sha256", required=True)
    parser.add_argument("--sqlite-consumer", required=True, type=Path)
    parser.add_argument("--sqlite-consumer-sha256", required=True)
    parser.add_argument("--msprof", required=True, type=Path)
    parser.add_argument("--msprof-sha256", required=True)
    args = parser.parse_args()
    target = prepare_runtime(
        args.output,
        args.sqlite_library,
        args.sqlite_library_sha256,
        args.sqlite_consumer,
        args.sqlite_consumer_sha256,
        args.msprof,
        args.msprof_sha256,
    )
    print(json.dumps({"manifest_path": str(target), "manifest_sha256": sha(target)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
