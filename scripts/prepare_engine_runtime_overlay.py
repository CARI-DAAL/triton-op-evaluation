#!/usr/bin/env python3
"""Prepare a pinned sparse runtime overlay without changing the sandbox anchor."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from triton_evaluation.evaluators.sandbox import (  # noqa: E402
    EngineRuntimeOverlay,
    EngineSandbox,
    apply_runtime_overlay,
)
from triton_evaluation.profiler_runtime import DIGEST_ENV, MANIFEST_ENV  # noqa: E402
from triton_evaluation.reference_cache import (  # noqa: E402
    BINDING_ENV,
    CACHE_ENV,
    STATE_ENV,
)
from triton_evaluation.reference_cache import DIGEST_ENV as CACHE_DIGEST_ENV  # noqa: E402


def prepare(output: Path, base: Path, base_sha: str, environment: dict[str, str]) -> dict[str, Any]:
    raw = base.read_bytes()
    if hashlib.sha256(raw).hexdigest() != base_sha:
        raise ValueError("Original sandbox bytes differ from their unchanged anchor SHA")
    overlay = EngineRuntimeOverlay(environment=environment)
    effective, evidence = apply_runtime_overlay(EngineSandbox.model_validate_json(raw), overlay)
    # Validate all derived mounts before publishing the sparse file. Their
    # actual execution and host-canary denial still require native acceptance.
    if not effective.readonly_mounts:
        raise ValueError("Runtime overlay did not resolve its service-owned dependencies")
    payload = (overlay.model_dump_json(indent=2) + "\n").encode()
    with output.open("xb") as stream:
        stream.write(payload)
    output.chmod(0o444)
    if base.read_bytes() != raw:
        raise RuntimeError("Original sandbox bytes changed during overlay preparation")
    return {
        "overlay_path": str(output.resolve()),
        "overlay_sha256": hashlib.sha256(payload).hexdigest(),
        "base_manifest_path": str(base.resolve()),
        "base_manifest_sha256": base_sha,
        "effective_sandbox_sha256": evidence["effective_sandbox_sha256"],
        "executed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--profiler-manifest", type=Path)
    parser.add_argument("--profiler-sha256")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--cache-index", type=Path)
    parser.add_argument("--cache-sha256")
    parser.add_argument("--cache-state-dir", type=Path)
    args = parser.parse_args()
    values = {
        MANIFEST_ENV: args.profiler_manifest,
        DIGEST_ENV: args.profiler_sha256,
        CACHE_ENV: args.cache_dir,
        BINDING_ENV: args.cache_index,
        CACHE_DIGEST_ENV: args.cache_sha256,
        STATE_ENV: args.cache_state_dir,
    }
    environment = {key: str(value) for key, value in values.items() if value is not None}
    print(json.dumps(prepare(args.output, args.base_manifest, args.base_sha256, environment)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
