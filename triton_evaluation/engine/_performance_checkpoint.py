"""Atomic performance snapshots; termination is classified by the host."""

import json
import os
import tempfile
from pathlib import Path


def case_progress(selected, completed):
    """Use original zero-based identities, independently of engine row formats.

    Completed means a case produced a measured or failed outcome, not that its
    measurement succeeded. Pending cases have no fabricated result rows.
    """
    return {
        "selected_case_indices": list(selected),
        "completed_case_indices": list(completed),
        "complete": list(completed) == list(selected),
    }


def write_checkpoint(path, result):
    """Keep the previous complete JSON visible until its replacement is ready."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
