#!/usr/bin/env python
"""Generate or verify the published evaluation-api 2.1 contract artifacts.

``--write`` atomically replaces the published tree under
``contracts/evaluation-api/2.1``;
``--check`` never writes and exits 1 with the sorted relative paths whose
file set membership or bytes differ.
"""
import argparse
import sys
from collections.abc import Sequence

from triton_evaluation.api.contract_artifacts import (
    CONTRACT_ROOT,
    build_artifacts,
    verify_artifacts,
    write_artifacts,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true", help="regenerate artifacts")
    action.add_argument("--check", action="store_true", help="verify without writing")
    args = parser.parse_args(argv)

    artifacts = build_artifacts()
    if args.write:
        write_artifacts(CONTRACT_ROOT, artifacts)
        print(f"wrote contract artifacts under {CONTRACT_ROOT}")
        return 0
    differing = verify_artifacts(CONTRACT_ROOT, artifacts)
    if differing:
        print(f"contract artifacts differ under {CONTRACT_ROOT}:", file=sys.stderr)
        for path in differing:
            print(path, file=sys.stderr)
        return 1
    print("contract artifacts are current")
    return 0


if __name__ == "__main__":
    sys.exit(main())
