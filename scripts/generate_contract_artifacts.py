#!/usr/bin/env python
"""Generate or verify the published evaluation-api 2.0 contract artifacts.

``--write`` atomically replaces the published tree under
``contracts/evaluation-api/2.0``; ``--check`` never writes and exits 1 with the
sorted relative paths whose file set membership or bytes differ.
"""
import argparse
import sys
from collections.abc import Sequence

from triton_evaluation.api.contract_artifacts import (
    CONTRACT_ROOT,
    verify_artifacts,
    write_artifacts,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true", help="regenerate artifacts")
    action.add_argument("--check", action="store_true", help="verify without writing")
    args = parser.parse_args(argv)

    if args.write:
        write_artifacts(CONTRACT_ROOT)
        print(f"wrote contract artifacts under {CONTRACT_ROOT}")
        return 0
    problems = verify_artifacts(CONTRACT_ROOT)
    if problems:
        print("contract artifacts differ:", file=sys.stderr)
        for path in problems:
            print(path, file=sys.stderr)
        return 1
    print("contract artifacts are current")
    return 0


if __name__ == "__main__":
    sys.exit(main())
