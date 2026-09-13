"""RED tests for the trusted release catalog (design section 3.2)."""
import hashlib
import json
import os

import pytest

from triton_evaluation.api.contract import GradeContext
from triton_evaluation.catalog import (
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
