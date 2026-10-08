#!/usr/bin/env python3
"""Register or replay an explicit official B1 selection without executing it."""

from __future__ import annotations

import argparse
import fcntl
import json
import re
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Support both installed-package usage and `python scripts/register_official_tasks.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from triton_evaluation.official_task import load_task  # noqa: E402
from triton_evaluation.service.catalog import ReleaseCatalog  # noqa: E402

DEFAULT_MANIFEST_ID = "npukernelbench-official-v1"
SOURCE_REPOSITORY = "https://gitcode.com/Ascend/ascendc-kernelgen-data"


class TaskSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    level: Literal["level3", "level4"]
    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$", max_length=116)
    declared_problem_id: str | None = Field(default=None, alias="problem_id")
    official_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    cases: int | None = Field(default=None, ge=1)

    @property
    def problem_id(self) -> str:
        return f"official_l{self.level[-1]}_{self.name}"


def read_selection(path: Path) -> list[TaskSelection]:
    """Read an explicit list or the official-b1-v1 selection document."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        if raw.get("schema_version") != "official-b1-v1":
            raise ValueError("selection document must use schema_version official-b1-v1")
        if raw.get("manifest_id") != DEFAULT_MANIFEST_ID:
            raise ValueError("selection document does not declare the official-v1 manifest")
        raw = raw.get("tasks")
    if not isinstance(raw, list):
        raise ValueError("selection must be a JSON list of {level, name} objects")
    selected = [TaskSelection.model_validate(item) for item in raw]
    _validate_selection(selected)
    return selected


def _validate_selection(selected: list[TaskSelection]) -> None:
    if len(selected) != 20 or Counter(task.level for task in selected) != {
        "level3": 10,
        "level4": 10,
    }:
        raise ValueError("B1 must explicitly select exactly 10 level3 and 10 level4 tasks")
    if len({task.problem_id for task in selected}) != len(selected):
        raise ValueError("selection contains duplicate task identities")
    if any(
        task.declared_problem_id is not None and task.declared_problem_id != task.problem_id
        for task in selected
    ):
        raise ValueError("declared problem_id does not match the canonical official task identity")
    if any(task.name in {"MsaSparseAttnFwd", "PagedAttentionV2"} for task in selected):
        raise ValueError("selection includes an excluded device-risk task")


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def build_provenance(task_entry: dict, selection: TaskSelection, manifest_id: str) -> dict:
    """Portable provenance; source_dir is relative to the official source root."""
    return {
        "schema_version": "1.0",
        "manifest_id": manifest_id,
        "problem_id": selection.problem_id,
        "op_name": task_entry["op_name"],
        "official_digest": task_entry["official_digest"],
        "definition_digest": task_entry["definition_digest"],
        "workload_digest": task_entry["workload_digest"],
        "adapter_version": task_entry["adapter_version"],
        "cases": task_entry["cases"],
        "source_repository": SOURCE_REPOSITORY,
        "source_dir": f"{selection.level}/{selection.name}",
        "reference_digest": task_entry["reference_digest"],
    }


def _assemble(official_root: Path, selected: list[TaskSelection], manifest_id: str) -> list[dict]:
    _validate_selection(selected)
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", manifest_id):
        raise ValueError("manifest_id must contain 1..128 safe identifier characters")
    artifacts = []
    official_root = official_root.resolve(strict=True)
    for selection in selected:
        task_dir = official_root / selection.level / selection.name
        if not task_dir.resolve(strict=True).is_relative_to(official_root):
            raise ValueError("official task directory escapes the official source root")
        entry = load_task(task_dir, problem_id=selection.problem_id)
        if (
            selection.official_digest is not None
            and selection.official_digest != entry["official_digest"]
        ):
            raise ValueError(f"official source digest differs for {selection.problem_id}")
        if selection.cases is not None and selection.cases != entry["cases"]:
            raise ValueError(f"official case count differs for {selection.problem_id}")
        # Syntax-only validation: no official imports, tensors or NPU work here.
        compile(entry["reference_source"], selection.problem_id, "exec")
        reference_path = f"references/{selection.problem_id}.py"
        artifacts.append(
            {
                "entry": {
                    "manifest_id": manifest_id,
                    "problem_id": selection.problem_id,
                    "reference_digest": entry["reference_digest"],
                    "reference_path": reference_path,
                },
                "source": entry["reference_source"].encode("utf-8"),
                "provenance": _json_bytes(build_provenance(entry, selection, manifest_id)),
                "provenance_path": reference_path + ".PROVENANCE.json",
                "cases": entry["cases"],
                "official_digest": entry["official_digest"],
            }
        )
    return artifacts


def _manifest_bytes(root: Path) -> bytes | None:
    path = root / "catalog.json"
    return path.read_bytes() if path.exists() else None


def _prepare(stage: Path, artifacts: list[dict]) -> tuple[bytes, list[tuple[str, bytes]]]:
    manifest = _manifest_bytes(stage)
    if manifest is None:
        payload = {"schema_version": "1.0", "entries": []}
    else:
        ReleaseCatalog.load(stage, stage / "catalog.json")
        payload = json.loads(manifest)
    by_key = {(entry["manifest_id"], entry["problem_id"]): entry for entry in payload["entries"]}
    additions = []
    for artifact in artifacts:
        entry = artifact["entry"]
        key = (entry["manifest_id"], entry["problem_id"])
        existing = by_key.get(key)
        if existing is not None and existing != entry:
            raise ValueError(f"immutable catalog binding differs for {entry['problem_id']}")
        for relative, blob in (
            (entry["reference_path"], artifact["source"]),
            (artifact["provenance_path"], artifact["provenance"]),
        ):
            path = stage / relative
            if path.exists() or path.is_symlink():
                if not path.resolve(strict=True).is_relative_to(stage.resolve()):
                    raise ValueError("official artifact escapes the catalog root")
                if not path.is_file() or path.read_bytes() != blob:
                    raise ValueError(
                        f"refusing to overwrite different artifact for {entry['problem_id']}"
                    )
            else:
                # A published binding requires its original artifact/sidecar;
                # registration cannot silently repair a missing evidence file.
                if existing is not None:
                    raise ValueError(
                        f"existing official binding lacks artifact for {entry['problem_id']}"
                    )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(blob)
                additions.append((relative, blob))
        if existing is None:
            payload["entries"].append(entry)
            by_key[key] = entry
    new_manifest = _json_bytes(payload)
    (stage / "catalog.json").write_bytes(new_manifest)
    catalog = ReleaseCatalog.load(stage, stage / "catalog.json")
    for artifact in artifacts:
        entry = artifact["entry"]
        catalog.resolve_official_task(entry["problem_id"], manifest_id=entry["manifest_id"])
    return new_manifest, additions


def _publish_file(path: Path, data: bytes) -> None:
    # Exclusive create protects pre-existing files even when an external writer
    # changes the catalog between staging and publication.
    stream = path.open("xb")
    try:
        with stream:
            stream.write(data)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def register_tasks(
    official_root: Path,
    catalog_root: Path,
    selected: list[TaskSelection],
    *,
    manifest_id: str = DEFAULT_MANIFEST_ID,
) -> dict:
    """Validate a staged catalog, then publish atomically with failure rollback.

    This updates catalog data only. It does not change service environment,
    restart processes, perform admission, or publish official source data.
    """
    artifacts = _assemble(official_root, selected, manifest_id)
    root = catalog_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("catalog_root must be an existing directory")
    # Serializes this registrar; also detect changes by other catalog writers.
    lock_path = root.parent / f".{root.name}.official-registration.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        original_manifest = _manifest_bytes(root)
        with tempfile.TemporaryDirectory(prefix=".official-register-", dir=root.parent) as work:
            stage = Path(work) / "catalog"
            shutil.copytree(root, stage, symlinks=True)
            new_manifest, additions = _prepare(stage, artifacts)
            if not additions and original_manifest is not None:
                # Idempotent registrations preserve even original JSON formatting.
                return _report(artifacts, "unchanged")
            created: list[Path] = []
            created_dirs: list[Path] = []
            manifest_published = False
            manifest_temp: Path | None = None
            try:
                for relative, blob in additions:
                    target = root / relative
                    if not target.parent.exists():
                        target.parent.mkdir()
                        created_dirs.append(target.parent)
                    if not target.parent.resolve().is_relative_to(root):
                        raise ValueError("publication path escapes the catalog root")
                    if target.exists() or target.is_symlink():
                        raise ValueError("publication artifact already exists")
                    _publish_file(target, blob)
                    created.append(target)
                if _manifest_bytes(root) != original_manifest:
                    raise ValueError("catalog changed while official registration was staged")
                with tempfile.NamedTemporaryFile(
                    prefix=".catalog.json.official-register-", dir=root, delete=False
                ) as stream:
                    manifest_temp = Path(stream.name)
                    stream.write(new_manifest)
                manifest_temp.replace(root / "catalog.json")
                manifest_published = True
                ReleaseCatalog.load(root, root / "catalog.json")
            except BaseException:
                if manifest_temp is not None:
                    manifest_temp.unlink(missing_ok=True)
                if manifest_published:
                    if original_manifest is None:
                        (root / "catalog.json").unlink(missing_ok=True)
                    else:
                        # This temp name is owned by this transaction, even after
                        # it was renamed into the active manifest.
                        if manifest_temp is None:
                            raise RuntimeError(
                                "published manifest lacks transaction temporary path"
                            ) from None
                        manifest_temp.write_bytes(original_manifest)
                        manifest_temp.replace(root / "catalog.json")
                for path in reversed(created):
                    path.unlink(missing_ok=True)
                for path in reversed(created_dirs):
                    path.rmdir()
                raise
    return _report(artifacts, "registered")


def replay_tasks(
    official_root: Path,
    catalog_root: Path,
    selected: list[TaskSelection],
    *,
    manifest_id: str = DEFAULT_MANIFEST_ID,
) -> dict:
    """Rebuild all artifacts, compare exact bytes, then validate catalog loading."""
    artifacts = _assemble(official_root, selected, manifest_id)
    root = catalog_root.resolve(strict=True)
    catalog = ReleaseCatalog.load(root, root / "catalog.json")
    for artifact in artifacts:
        entry = artifact["entry"]
        task = catalog.resolve_official_task(entry["problem_id"], manifest_id=manifest_id)
        if task.grade_context.reference_digest != entry["reference_digest"]:
            raise ValueError(f"reassembled binding differs for {entry['problem_id']}")
        for relative, expected in (
            (entry["reference_path"], artifact["source"]),
            (artifact["provenance_path"], artifact["provenance"]),
        ):
            if (root / relative).read_bytes() != expected:
                raise ValueError(f"reassembled bytes differ for {entry['problem_id']}")
    return _report(artifacts, "replay_identical")


def _report(artifacts: list[dict], status: str) -> dict:
    return {
        "status": status,
        "tasks": len(artifacts),
        "total_cases": sum(artifact["cases"] for artifact in artifacts),
        "entries": [
            {
                **artifact["entry"],
                "official_digest": artifact["official_digest"],
                "cases": artifact["cases"],
            }
            for artifact in artifacts
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--catalog-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--manifest-id", default=DEFAULT_MANIFEST_ID)
    parser.add_argument("--replay", action="store_true", help="read-only exact-byte rebuild check")
    args = parser.parse_args(argv)
    selected = read_selection(args.selection)
    operation = replay_tasks if args.replay else register_tasks
    report = operation(
        args.official_root, args.catalog_root, selected, manifest_id=args.manifest_id
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
