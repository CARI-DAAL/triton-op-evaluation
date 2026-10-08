"""Official catalog publication preserves bindings and rolls back failed writes."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import register_official_tasks as registrar  # noqa: E402
from triton_evaluation.api.contract import GradeContext
from triton_evaluation.service.catalog import ManifestBindingError, ReleaseCatalog


def _snapshot(root: Path):
    return {
        str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }


@pytest.fixture
def publication(tmp_path):
    official_root = tmp_path / "official"
    catalog_root = tmp_path / "catalog"
    catalog_root.mkdir()
    selected = [
        registrar.TaskSelection(level=level, name=f"Demo{i}")
        for level in ("level3", "level4")
        for i in range(10)
    ]
    for selection in selected:
        task_dir = official_root / selection.level / selection.name
        task_dir.mkdir(parents=True)
        (task_dir / "definition.json").write_text(
            json.dumps(
                {
                    "name": selection.name,
                    "inputs": {"x": {}},
                    "reference": "def run(x=None):\n    return x\n\n",
                }
            ),
            encoding="utf-8",
        )
        (task_dir / "workload.jsonl").write_text(
            '{"inputs":{"x":{"type":"scalar","value":null}}}\n',
            encoding="utf-8",
        )
    legacy_source = "legacy reference\n"
    (catalog_root / "legacy.py").write_text(legacy_source, encoding="utf-8")
    legacy_digest = hashlib.sha256(legacy_source.encode()).hexdigest()
    legacy = {
        "manifest_id": "local-v1",
        "problem_id": "old-problem",
        "reference_digest": legacy_digest,
        "reference_path": "legacy.py",
    }
    (catalog_root / "catalog.json").write_text(
        json.dumps({"schema_version": "1.0", "entries": [legacy]}),
        encoding="utf-8",
    )
    return official_root, catalog_root, selected, legacy


def test_register_replay_and_legacy_bindings(publication):
    official, root, selected, legacy = publication
    legacy_bytes = (root / "legacy.py").read_bytes()
    report = registrar.register_tasks(official, root, selected)
    assert report["status"] == "registered" and report["tasks"] == 20
    assert report["total_cases"] == 20
    payload = json.loads((root / "catalog.json").read_bytes())
    assert len(payload["entries"]) == 21
    assert payload["entries"][0] == legacy
    assert (root / "legacy.py").read_bytes() == legacy_bytes
    catalog = ReleaseCatalog.load(root, root / "catalog.json")
    catalog.bind(
        GradeContext(
            **{
                key: legacy[key]
                for key in (
                    "manifest_id",
                    "problem_id",
                    "reference_digest",
                )
            }
        ),
        legacy_bytes.decode(),
    )
    task = catalog.resolve_official_task(selected[0].problem_id)
    assert task.op_name == selected[0].name
    assert task.grade_context.manifest_id == registrar.DEFAULT_MANIFEST_ID
    before = _snapshot(root)
    assert registrar.replay_tasks(official, root, selected)["status"] == "replay_identical"
    assert _snapshot(root) == before
    assert registrar.register_tasks(official, root, selected)["status"] == "unchanged"
    assert _snapshot(root) == before


def test_changed_official_bytes_cannot_rebind(publication):
    official, root, selected, _ = publication
    registrar.register_tasks(official, root, selected)
    before = _snapshot(root)
    task = official / selected[0].level / selected[0].name / "definition.json"
    task.write_bytes(task.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="immutable catalog binding"):
        registrar.register_tasks(official, root, selected)
    assert _snapshot(root) == before
    with pytest.raises(ValueError, match="reassembled binding"):
        registrar.replay_tasks(official, root, selected)


def test_publication_failure_rolls_back_created_artifacts(publication, monkeypatch):
    official, root, selected, _ = publication
    before = _snapshot(root)
    original = registrar._publish_file
    calls = 0

    def fail_on_third_file(path, data):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected disk write failure")
        original(path, data)

    monkeypatch.setattr(registrar, "_publish_file", fail_on_third_file)
    with pytest.raises(OSError, match="injected"):
        registrar.register_tasks(official, root, selected)
    assert _snapshot(root) == before
    assert not (root / "references").exists()
    ReleaseCatalog.load(root, root / "catalog.json")


def test_invalid_existing_catalog_is_untouched(publication):
    official, root, selected, _ = publication
    (root / "legacy.py").write_bytes(b"changed existing bytes")
    before = _snapshot(root)
    with pytest.raises(ValueError, match="digest mismatch"):
        registrar.register_tasks(official, root, selected)
    assert _snapshot(root) == before


def test_existing_artifact_collision_is_untouched(publication):
    official, root, selected, _ = publication
    (root / "references").mkdir()
    (root / "references" / (selected[0].problem_id + ".py")).write_bytes(b"other source")
    before = _snapshot(root)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        registrar.register_tasks(official, root, selected)
    assert _snapshot(root) == before


def test_replay_rejects_changed_provenance(publication):
    official, root, selected, _ = publication
    registrar.register_tasks(official, root, selected)
    path = root / "references" / (selected[0].problem_id + ".py.PROVENANCE.json")
    payload = json.loads(path.read_bytes())
    payload["official_digest"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ManifestBindingError):
        registrar.replay_tasks(official, root, selected)


def test_resolver_rechecks_reference_file_on_each_call(publication):
    official, root, selected, _ = publication
    registrar.register_tasks(official, root, selected)
    catalog = ReleaseCatalog.load(root, root / "catalog.json")
    path = root / "references" / (selected[0].problem_id + ".py")
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ManifestBindingError):
        catalog.resolve_official_task(selected[0].problem_id)


def test_resolver_rejects_missing_and_ambiguous_manifest(publication):
    official, root, selected, _ = publication
    registrar.register_tasks(official, root, selected)
    payload = json.loads((root / "catalog.json").read_bytes())
    payload["entries"].append(dict(payload["entries"][1], manifest_id="another-official-release"))
    (root / "catalog.json").write_text(json.dumps(payload), encoding="utf-8")
    catalog = ReleaseCatalog.load(root, root / "catalog.json")
    with pytest.raises(ManifestBindingError):
        catalog.resolve_official_task(selected[0].problem_id)
    with pytest.raises(ManifestBindingError):
        catalog.resolve_official_task("official_l3_Missing")
    assert (
        catalog.resolve_official_task(
            selected[0].problem_id,
            manifest_id=registrar.DEFAULT_MANIFEST_ID,
        ).problem_id
        == selected[0].problem_id
    )


def test_selection_rejects_wrong_size_and_duplicates(publication):
    _, _, selected, _ = publication
    with pytest.raises(ValueError, match="exactly 10"):
        registrar._validate_selection(selected[:19])
    duplicate = [selected[1], *selected[1:]]
    with pytest.raises(ValueError, match="duplicate"):
        registrar._validate_selection(duplicate)


def test_selection_document_keeps_official_digest_constraints(publication, tmp_path):
    official, root, selected, _ = publication
    document = {
        "schema_version": "official-b1-v1",
        "manifest_id": registrar.DEFAULT_MANIFEST_ID,
        "tasks": [item.model_dump(by_alias=True, exclude_none=True) for item in selected],
    }
    document["tasks"][0]["official_digest"] = "0" * 64
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    parsed = registrar.read_selection(path)
    before = _snapshot(root)
    with pytest.raises(ValueError, match="official source digest"):
        registrar.register_tasks(official, root, parsed)
    assert _snapshot(root) == before


def test_provenance_does_not_bind_to_machine_specific_absolute_path(publication):
    official, root, selected, _ = publication
    registrar.register_tasks(official, root, selected)
    path = root / "references" / (selected[0].problem_id + ".py.PROVENANCE.json")
    payload = json.loads(path.read_bytes())
    assert payload["source_dir"] == "level3/Demo0"
    assert payload["definition_digest"] and payload["workload_digest"]
    assert str(official) not in path.read_text()


def test_failed_final_validation_restores_original_manifest(publication, monkeypatch):
    official, root, selected, _ = publication
    before = _snapshot(root)
    original = ReleaseCatalog.load

    def fail_live_final_load(cls, candidate_root, manifest_path):
        if Path(candidate_root) == root:
            raise ValueError("injected final validation failure")
        return original(candidate_root, manifest_path)

    monkeypatch.setattr(ReleaseCatalog, "load", classmethod(fail_live_final_load))
    with pytest.raises(ValueError, match="injected final"):
        registrar.register_tasks(official, root, selected)
    assert _snapshot(root) == before
    assert not (root / "references").exists()
    original(root, root / "catalog.json")


@pytest.mark.parametrize(
    "field,value",
    [
        ("reference_digest", "0" * 64),
        ("adapter_version", "wrong-adapter"),
        ("manifest_id", "wrong-release"),
        ("problem_id", "official_l3_Wrong"),
    ],
)
def test_resolver_rejects_inconsistent_provenance(publication, field, value):
    official, root, selected, _ = publication
    registrar.register_tasks(official, root, selected)
    path = root / "references" / (selected[0].problem_id + ".py.PROVENANCE.json")
    payload = json.loads(path.read_bytes())
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")
    catalog = ReleaseCatalog.load(root, root / "catalog.json")
    with pytest.raises(ManifestBindingError):
        catalog.resolve_official_task(selected[0].problem_id)
