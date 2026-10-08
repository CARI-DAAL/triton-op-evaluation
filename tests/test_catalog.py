"""RED tests for the trusted release catalog (design section 3.2)."""
import hashlib
import json
import os

import pytest

from triton_evaluation.api.contract import GradeContext
from triton_evaluation.service.catalog import (
    CatalogBinding,
    ManifestBindingError,
    ReleaseCatalog,
)

REFERENCE_CODE = "import torch\n\n\nclass Model(torch.nn.Module):\n    pass\n"
OTHER_CODE = "import torch\n\n\nclass Other(torch.nn.Module):\n    pass\n"


def digest_of(data) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def make_entry(reference_path="references/addmm-1.py", code=REFERENCE_CODE, **overrides):
    entry = {
        "manifest_id": "release-1",
        "problem_id": "addmm-1",
        "reference_digest": digest_of(code),
        "reference_path": reference_path,
    }
    entry.update(overrides)
    return entry


def build_catalog(tmp_path, entries, artifacts=None, **payload_overrides):
    root = tmp_path / "catalog"
    root.mkdir(exist_ok=True)
    for relative, data in (artifacts or {}).items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, str):
            data = data.encode("utf-8")
        target.write_bytes(data)
    payload = {"schema_version": "1.0", "entries": entries}
    payload.update(payload_overrides)
    manifest = root / "catalog.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return root, manifest


def load_catalog(tmp_path, entries, artifacts=None, **payload_overrides):
    root, manifest = build_catalog(tmp_path, entries, artifacts, **payload_overrides)
    return ReleaseCatalog.load(root, manifest)


def context(reference_digest, manifest_id="release-1", problem_id="addmm-1"):
    return GradeContext(
        manifest_id=manifest_id, problem_id=problem_id, reference_digest=reference_digest
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_load_valid_catalog_and_bind(tmp_path):
    catalog = load_catalog(
        tmp_path, [make_entry()], {"references/addmm-1.py": REFERENCE_CODE}
    )
    binding = catalog.bind(context(digest_of(REFERENCE_CODE)), REFERENCE_CODE)
    assert binding == CatalogBinding(
        manifest_id="release-1",
        problem_id="addmm-1",
        reference_digest=digest_of(REFERENCE_CODE),
    )


def test_valid_artifact_loads_beneath_a_read_only_root(tmp_path):
    root, manifest = build_catalog(
        tmp_path, [make_entry()], {"references/addmm-1.py": REFERENCE_CODE}
    )
    os.chmod(root, 0o555)
    try:
        catalog = ReleaseCatalog.load(root, manifest)
    finally:
        os.chmod(root, 0o755)
    assert catalog.bind(context(digest_of(REFERENCE_CODE)), REFERENCE_CODE).manifest_id == (
        "release-1"
    )


def test_one_manifest_may_publish_multiple_problems(tmp_path):
    entries = [
        make_entry(),
        make_entry(problem_id="relu-1", reference_path="references/relu-1.py",
                   code=OTHER_CODE),
    ]
    artifacts = {
        "references/addmm-1.py": REFERENCE_CODE,
        "references/relu-1.py": OTHER_CODE,
    }
    catalog = load_catalog(tmp_path, entries, artifacts)
    binding = catalog.bind(context(digest_of(OTHER_CODE), problem_id="relu-1"), OTHER_CODE)
    assert binding.problem_id == "relu-1"


# ---------------------------------------------------------------------------
# Identity uniqueness and immutability
# ---------------------------------------------------------------------------


def test_duplicate_manifest_problem_pair_is_rejected(tmp_path):
    entries = [make_entry(), make_entry()]
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        load_catalog(tmp_path, entries, {"references/addmm-1.py": REFERENCE_CODE})


def test_rebinding_a_pair_to_new_reference_bytes_is_rejected(tmp_path):
    entries = [
        make_entry(),
        make_entry(reference_path="references/other.py", code=OTHER_CODE),
    ]
    artifacts = {
        "references/addmm-1.py": REFERENCE_CODE,
        "references/other.py": OTHER_CODE,
    }
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        load_catalog(tmp_path, entries, artifacts)


def test_unknown_schema_version_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_catalog(
            tmp_path, [make_entry()], {"references/addmm-1.py": REFERENCE_CODE},
            schema_version="2.1",
        )


def test_unexpected_manifest_fields_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_catalog(
            tmp_path, [make_entry()], {"references/addmm-1.py": REFERENCE_CODE},
            comment="not part of the fixed catalog schema",
        )


# ---------------------------------------------------------------------------
# Reference artifact validation
# ---------------------------------------------------------------------------


def test_missing_reference_file_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="missing") as excinfo:
        load_catalog(tmp_path, [make_entry()])
    assert "addmm-1.py" not in str(excinfo.value)


def test_reference_digest_mismatch_is_rejected(tmp_path):
    entry = make_entry(reference_digest=digest_of(OTHER_CODE))
    with pytest.raises(ValueError, match="digest") as excinfo:
        load_catalog(tmp_path, [entry], {"references/addmm-1.py": REFERENCE_CODE})
    assert "addmm-1.py" not in str(excinfo.value)


def test_absolute_reference_path_is_rejected(tmp_path):
    entry = make_entry(reference_path="/etc/hostname")
    with pytest.raises(ValueError, match="relative"):
        load_catalog(tmp_path, [entry])


def test_dot_dot_traversal_is_rejected_even_when_it_lands_inside(tmp_path):
    entry = make_entry(reference_path="references/../references/addmm-1.py")
    with pytest.raises(ValueError, match=r"\.\."):
        load_catalog(tmp_path, [entry], {"references/addmm-1.py": REFERENCE_CODE})


def test_symlink_escaping_the_root_is_rejected(tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_text(REFERENCE_CODE, encoding="utf-8")
    entry = make_entry(reference_path="references/link.py")
    root, manifest = build_catalog(
        tmp_path, [entry], {"references/addmm-1.py": REFERENCE_CODE}
    )
    os.symlink(outside, root / "references" / "link.py")
    with pytest.raises(ValueError, match="escapes"):
        ReleaseCatalog.load(root, manifest)


def test_reference_path_pointing_at_a_directory_is_rejected(tmp_path):
    entry = make_entry(reference_path="references")
    with pytest.raises(ValueError, match="regular file"):
        load_catalog(tmp_path, [entry], {"references/addmm-1.py": REFERENCE_CODE})


def test_non_utf8_artifact_is_rejected(tmp_path):
    raw = b"\xff\xfe\xffnot utf-8"
    entry = make_entry(reference_digest=digest_of(raw))
    with pytest.raises(ValueError, match="UTF-8"):
        load_catalog(tmp_path, [entry], {"references/addmm-1.py": raw})


def test_manifest_outside_the_root_is_rejected(tmp_path):
    root, _ = build_catalog(
        tmp_path, [make_entry()], {"references/addmm-1.py": REFERENCE_CODE}
    )
    outside_manifest = tmp_path / "catalog.json"
    outside_manifest.write_text(
        json.dumps({"schema_version": "1.0", "entries": []}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="escapes"):
        ReleaseCatalog.load(root, outside_manifest)


# ---------------------------------------------------------------------------
# Binding failures
# ---------------------------------------------------------------------------


def test_bind_rejects_an_unknown_manifest(tmp_path):
    catalog = load_catalog(
        tmp_path, [make_entry()], {"references/addmm-1.py": REFERENCE_CODE}
    )
    with pytest.raises(ManifestBindingError) as excinfo:
        catalog.bind(
            context(digest_of(REFERENCE_CODE), manifest_id="release-9"), REFERENCE_CODE
        )
    message = str(excinfo.value)
    assert excinfo.value.code == "MANIFEST_BINDING_FAILED"
    assert "release-9" not in message
    assert REFERENCE_CODE not in message
    assert "addmm-1.py" not in message


def test_bind_rejects_an_unknown_problem(tmp_path):
    catalog = load_catalog(
        tmp_path, [make_entry()], {"references/addmm-1.py": REFERENCE_CODE}
    )
    with pytest.raises(ManifestBindingError):
        catalog.bind(context(digest_of(REFERENCE_CODE), problem_id="relu-9"),
                     REFERENCE_CODE)


def test_bind_rejects_a_context_digest_mismatch(tmp_path):
    catalog = load_catalog(
        tmp_path, [make_entry()], {"references/addmm-1.py": REFERENCE_CODE}
    )
    with pytest.raises(ManifestBindingError):
        catalog.bind(context(digest_of(OTHER_CODE)), REFERENCE_CODE)


def test_bind_rejects_a_body_digest_mismatch(tmp_path):
    catalog = load_catalog(
        tmp_path, [make_entry()], {"references/addmm-1.py": REFERENCE_CODE}
    )
    with pytest.raises(ManifestBindingError) as excinfo:
        catalog.bind(context(digest_of(REFERENCE_CODE)), OTHER_CODE)
    assert OTHER_CODE not in str(excinfo.value)


# ---------------------------------------------------------------------------
# Task resolution for evaluation tools
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_resolve_plain_task_without_sidecar_or_generated_header(tmp_path, newline):
    source = '# 自定义参考实现\n"""User reference."""\n' + REFERENCE_CODE
    source = source.replace("\n", newline)
    catalog = load_catalog(
        tmp_path, [make_entry(code=source)], {"references/addmm-1.py": source}
    )
    task = catalog.resolve_task("addmm-1")
    assert task.problem_id == "addmm-1"
    assert task.reference_code == source
    assert task.grade_context == context(digest_of(source))
    assert catalog.bind(task.grade_context, task.reference_code) == CatalogBinding(
        manifest_id="release-1", problem_id="addmm-1", reference_digest=digest_of(source)
    )


def test_resolve_task_ignores_unrelated_sidecar(tmp_path):
    catalog = load_catalog(
        tmp_path,
        [make_entry()],
        {
            "references/addmm-1.py": REFERENCE_CODE,
            "references/addmm-1.py.PROVENANCE.json": "not JSON or serving metadata",
        },
    )
    assert catalog.resolve_task("addmm-1").reference_code == REFERENCE_CODE


def test_resolve_task_requires_unique_manifest_or_explicit_selection(tmp_path):
    catalog = load_catalog(
        tmp_path,
        [
            make_entry(),
            make_entry(manifest_id="release-2", reference_path="references/v2.py", code=OTHER_CODE),
        ],
        {"references/addmm-1.py": REFERENCE_CODE, "references/v2.py": OTHER_CODE},
    )
    with pytest.raises(ManifestBindingError):
        catalog.resolve_task("addmm-1")
    first = catalog.resolve_task("addmm-1", manifest_id="release-1")
    second = catalog.resolve_task("addmm-1", manifest_id="release-2")
    assert first.reference_code == REFERENCE_CODE
    assert second.reference_code == OTHER_CODE
    assert second.grade_context == context(digest_of(OTHER_CODE), manifest_id="release-2")
    for task_id, manifest_id in [("unknown-task", None), ("addmm-1", "unknown-release")]:
        with pytest.raises(ManifestBindingError):
            catalog.resolve_task(task_id, manifest_id=manifest_id)


def test_resolve_task_without_mounted_root_is_rejected():
    with pytest.raises(ManifestBindingError):
        ReleaseCatalog({}).resolve_task("addmm-1")


def test_resolve_task_rechecks_reference_digest_on_every_call(tmp_path):
    root, manifest = build_catalog(
        tmp_path, [make_entry()], {"references/addmm-1.py": REFERENCE_CODE}
    )
    catalog = ReleaseCatalog.load(root, manifest)
    assert catalog.resolve_task("addmm-1").reference_code == REFERENCE_CODE
    (root / "references/addmm-1.py").write_text(OTHER_CODE, encoding="utf-8")
    with pytest.raises(ManifestBindingError) as excinfo:
        catalog.resolve_task("addmm-1")
    assert str(excinfo.value) == "reference does not match the bound manifest entry"


@pytest.mark.parametrize("replacement", ["missing", "directory", "outside_symlink"])
def test_resolve_task_rejects_changed_reference_path(tmp_path, replacement):
    root, manifest = build_catalog(
        tmp_path, [make_entry()], {"references/addmm-1.py": REFERENCE_CODE}
    )
    catalog = ReleaseCatalog.load(root, manifest)
    reference = root / "references/addmm-1.py"
    reference.unlink()
    if replacement == "directory":
        reference.mkdir()
    elif replacement == "outside_symlink":
        outside = tmp_path / "outside.py"
        outside.write_text(REFERENCE_CODE, encoding="utf-8")
        reference.symlink_to(outside)
    with pytest.raises(ManifestBindingError):
        catalog.resolve_task("addmm-1")


def test_resolve_task_rejects_unreadable_artifact(tmp_path, monkeypatch):
    root, manifest = build_catalog(
        tmp_path, [make_entry()], {"references/addmm-1.py": REFERENCE_CODE}
    )
    catalog = ReleaseCatalog.load(root, manifest)
    reference = root / "references/addmm-1.py"
    original_read = type(reference).read_bytes

    def unreadable(path):
        if path == reference:
            raise PermissionError("private artifact location")
        return original_read(path)

    monkeypatch.setattr(type(reference), "read_bytes", unreadable)
    with pytest.raises(ManifestBindingError) as excinfo:
        catalog.resolve_task("addmm-1")
    assert "private artifact location" not in str(excinfo.value)


def test_resolve_returns_the_same_bytes_it_validates(tmp_path, monkeypatch):
    root, manifest = build_catalog(
        tmp_path, [make_entry()], {"references/addmm-1.py": REFERENCE_CODE}
    )
    catalog = ReleaseCatalog.load(root, manifest)
    reference = root / "references/addmm-1.py"
    original_read = type(reference).read_bytes
    reads = []

    def replace_after_read(path):
        blob = original_read(path)
        if path == reference:
            reads.append(blob)
            reference.write_text(OTHER_CODE, encoding="utf-8")
        return blob

    monkeypatch.setattr(type(reference), "read_bytes", replace_after_read)
    task = catalog.resolve_task("addmm-1")
    assert task.reference_code == REFERENCE_CODE
    assert reads == [REFERENCE_CODE.encode("utf-8")]
    assert task.grade_context.reference_digest == digest_of(task.reference_code)
