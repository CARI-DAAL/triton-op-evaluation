#!/usr/bin/env python3
"""Validate audited per-card evidence and seal one F-63 index (no reference execution)."""

import argparse
import json
from pathlib import Path

from triton_evaluation.runtime.reference_cache import (
    CacheDeployment,
    CaseManifest,
    ReferenceCache,
    ReferenceCacheError,
    sha256,
)


def seal_index(cache_root: Path, deployment: CacheDeployment, output: Path) -> dict:
    entries = {}
    coverage = {}
    for path in sorted((cache_root / "cases").glob("*.json")):
        if path.is_symlink():
            raise ReferenceCacheError("reference manifest must not be a symlink")
        payload = path.read_bytes()
        manifest = CaseManifest.model_validate_json(payload)
        binding = manifest.binding
        if binding.deployment_sha256 != deployment.digest:
            raise ReferenceCacheError("cache root contains evidence from a different deployment")
        if (
            path.name != f"{binding.key}.json"
            or binding.device_id not in deployment.device_conditions
        ):
            raise ReferenceCacheError("manifest identity or physical card is not admitted")
        cache = ReferenceCache(cache_root, deployment, binding.device_id)
        # Checks every referenced payload SHA, tensor dtype, shape and structure.
        cache.outputs(manifest)
        entries[binding.key] = sha256(payload)
        source = coverage.setdefault(binding.reference_source_sha256, {})
        source.setdefault(binding.device_id, []).append(binding.case_idx)
    if not entries:
        raise ReferenceCacheError("cannot seal an empty audited reference cache")
    cache = ReferenceCache(cache_root, deployment, next(iter(deployment.device_conditions)))
    index_sha = cache.write_index(output, entries)
    return {
        "schema_version": "reference-cache-coverage-v1",
        "index_sha256": index_sha,
        "deployment_sha256": deployment.digest,
        "entry_count": len(entries),
        "observed_case_indices_by_source_and_card": coverage,
        "coverage_complete": None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coverage-output", type=Path, required=True)
    args = parser.parse_args()
    deployment = CacheDeployment.model_validate_json(args.deployment.read_bytes())
    coverage = seal_index(args.cache_root, deployment, args.output)
    args.coverage_output.write_text(json.dumps(coverage, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in coverage.items()
                if key != "observed_case_indices_by_source_and_card"
            }
        )
    )


if __name__ == "__main__":
    main()
