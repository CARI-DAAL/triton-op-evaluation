#!/usr/bin/env python
"""Generate or verify the published evaluation-api 2.0 and 2.1 contract artifacts.

``--write`` atomically replaces both published trees under
``contracts/evaluation-api/2.0`` and ``contracts/evaluation-api/2.1``;
``--check`` never writes and exits 1 with the sorted relative paths whose
file set membership or bytes differ.
"""
import argparse
import sys
from collections.abc import Sequence

from triton_evaluation.api.contract_artifacts import (
    CONTRACT_ROOT,
    CONTRACT_ROOT_V21,
    build_v20_artifacts,
    build_v21_artifacts,
    verify_artifacts,
    write_artifacts,
)

_TREES = (
    (CONTRACT_ROOT, build_v20_artifacts),
    (CONTRACT_ROOT_V21, build_v21_artifacts),
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true", help="regenerate artifacts")
    action.add_argument("--check", action="store_true", help="verify without writing")
    args = parser.parse_args(argv)

    if args.write:
        for root, build in _TREES:
            write_artifacts(root, build())
            print(f"wrote contract artifacts under {root}")
        return 0
    problems = False
    for root, build in _TREES:
        differing = verify_artifacts(root, build())
        if differing:
            problems = True
            print(f"contract artifacts differ under {root}:", file=sys.stderr)
            for path in differing:
                print(path, file=sys.stderr)
    if problems:
        return 1
    print("contract artifacts are current")
    return 0


if __name__ == "__main__":
    sys.exit(main())
