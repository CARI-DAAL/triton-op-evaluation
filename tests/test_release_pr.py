"""Release automation with real Git repositories and a substituted GitHub adapter."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts import release_manifest, release_pr
from triton_evaluation import versioning


@pytest.fixture
def repository(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    root.mkdir()
    original_root = release_pr.ROOT
    for name in ["pyproject.toml", *release_manifest.PLUGIN_MANIFESTS, release_manifest.CLI_SCRIPT]:
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original_root / name, destination)
    # Keep historical test scenarios independent of the live release version.
    project = root / "pyproject.toml"
    project.write_text(re.sub(r'^version = "[^"]+"', 'version = "0.1.0"',
                              project.read_text(), count=1, flags=re.MULTILINE))
    (root / "CHANGELOG.md").write_text("# Changelog\n\n## Unreleased\n")
    (root / "triton_evaluation").mkdir()
    (root / "triton_evaluation/__init__.py").write_text('"""Test release source."""\n')
    monkeypatch.setattr(release_pr, "ROOT", root)
    # The workflow installs each release's distribution before generating its manifest.
    monkeypatch.setattr(
        versioning, "service_version", lambda: release_manifest.pyproject_version(root),
    )
    real_run = release_pr.run
    real_run("git", "init", "-b", "main")
    real_run("git", "config", "user.name", "Release tests")
    real_run("git", "config", "user.email", "release@example.invalid")
    real_run("git", "add", ".")
    real_run("git", "commit", "-m", "initial")
    remote = tmp_path / "remote.git"
    real_run("git", "init", "--bare", str(remote))
    real_run("git", "remote", "add", "origin", str(remote))
    real_run("git", "push", "-u", "origin", "main")
    state = {"releases": [], "assets": {}, "calls": [], "fail_create": False}

    def run(*args):
        if args[0] != "gh":
            return real_run(*args)
        state["calls"].append(args)
        if args[1:4] == ("api", "--paginate", "--slurp"):
            if state.get("api_error"):
                raise subprocess.CalledProcessError(1, args, stderr="GitHub unavailable")
            return json.dumps([state["releases"]])
        if args[1:3] == ("release", "create"):
            if state["fail_create"]:
                raise subprocess.CalledProcessError(1, args, stderr="transient publish failure")
            tag, artifact = args[3:5]
            assert "--verify-tag" in args
            state["assets"][tag] = Path(artifact).read_bytes()
            state["releases"].append({"tag_name": tag, "draft": False,
                                       "assets": [{"name": "release-manifest.json"}]})
            return ""
        if args[1:3] == ("release", "download"):
            destination = Path(args[args.index("--dir") + 1])
            (destination / "release-manifest.json").write_bytes(state["assets"][args[3]])
            return ""
        if args[1:3] == ("release", "upload"):
            state["assets"][args[3]] = Path(args[4]).read_bytes()
            state["releases"][0]["assets"] = [{"name": "release-manifest.json"}]
            return ""
        if args[1:3] == ("pr", "view"):
            return json.dumps(state["pr"])
        if args[1:3] == ("pr", "close"):
            state["pr"]["state"] = "CLOSED"
            return ""
        pytest.fail(f"Unexpected GitHub call: {args}")

    monkeypatch.setattr(release_pr, "run", run)
    return root, state, real_run


def prepare(root, tmp_path):
    output, body = tmp_path / "outputs", tmp_path / "body"
    release_pr.main(["prepare", "--body-file", str(body), "--output-file", str(output)])
    return output.read_text(), body


def commit_release(run):
    run("git", "add", "pyproject.toml", "CHANGELOG.md")
    run("git", "commit", "-m", "chore(release): reviewed version")
    run("git", "push", "origin", "main")
    return run("git", "rev-parse", "HEAD")


def service_change(root, run, message):
    source = root / "triton_evaluation/__init__.py"
    source.write_text(source.read_text() + "# Service change\n")
    run("git", "add", "triton_evaluation/__init__.py")
    run("git", "commit", "-m", message)


def publish(head):
    release_pr.main(["publish", "--repo", "example/evaluation", "--reviewed-head", head])


@pytest.mark.parametrize("current,messages,expected", [
    ("0.1.0", ["docs: explain", "test: cover", "chore: clean", "refactor: move"], None),
    ("0.1.0", ["fix(cli): repair"], "patch"),
    ("0.1.0", ["perf: faster"], "patch"),
    ("0.1.0", ["revert: restore"], "patch"),
    ("0.1.0", ["fix: repair", "feat: add option"], "minor"),
    ("0.1.0", ["refactor!: change contract"], "minor"),
    ("0.1.0", ["fix: change\n\nBREAKING CHANGE: migration"], "minor"),
    ("1.2.3", ["feat: add", "fix: change\n\nBREAKING-CHANGE: migration"], "major"),
    ("1.2.3", ["feat!: change contract"], "major"),
    ("1.2.3", ["Merge pull request #1", "fix: nested commit"], "patch"),
])
def test_version_policy(current, messages, expected):
    assert release_pr.suggested_bump(current, messages) == expected


@pytest.mark.parametrize("version", ["01.2.3", "1.0.0rc1", "1.2", "v1.2.3"])
def test_nonstable_versions_are_refused(version):
    with pytest.raises(ValueError):
        release_pr.version_parts(version)


def test_initial_release_ignores_backup_tags_and_preserves_client_version(repository, tmp_path):
    root, _, run = repository
    for tag in ["backup/machine2", "archive/spike/cardless-compile", "recovery/old"]:
        run("git", "tag", tag)
    original = run("git", "rev-parse", "HEAD")
    client = release_manifest.client_version(root)
    output, body = prepare(root, tmp_path)
    assert "pending=true\nversion=0.1.0\n" in output
    assert "initial release" in body.read_text()
    assert release_manifest.pyproject_version(root) == "0.1.0"
    assert release_manifest.client_version(root) == client
    assert release_pr.release_entry((root / "CHANGELOG.md").read_text(), "0.1.0")
    assert run("git", "rev-parse", "HEAD") == original
    assert release_pr.stable_tags() == []


def test_prepare_is_deterministic_and_rolls_migration_notes(repository, tmp_path):
    root, _, run = repository
    (root / "CHANGELOG.md").write_text("# Changelog\n\n## Unreleased\n\nMigration note.\n")
    run("git", "add", "CHANGELOG.md")
    run("git", "commit", "-m", "docs: migration")
    run("git", "tag", "v0.1.0")
    service_change(root, run, "feat: new counter")
    output, body = prepare(root, tmp_path)
    generated = {name: (root / name).read_bytes() for name in ["pyproject.toml", "CHANGELOG.md"]}
    notes = body.read_bytes()
    # Restore only test-owned preparation files to repeat the same input.
    run("git", "restore", "pyproject.toml", "CHANGELOG.md")
    assert prepare(root, tmp_path)[0] == output
    assert body.read_bytes() == notes
    assert generated == {name: (root / name).read_bytes() for name in generated}
    assert release_manifest.pyproject_version(root) == "0.2.0"
    entry = release_pr.release_entry((root / "CHANGELOG.md").read_text(), "0.2.0")
    assert "Migration note." in entry


def test_docs_only_changes_do_not_prepare_a_release(repository, tmp_path):
    root, _, run = repository
    run("git", "tag", "v0.1.0")
    run("git", "commit", "--allow-empty", "-m", "docs: clarify")
    output, body = prepare(root, tmp_path)
    assert output == "pending=false\n" and not body.exists()
    assert run("git", "status", "--porcelain") == ""


@pytest.mark.parametrize("initial", [True, False])
def test_merged_release_waits_for_publication_and_publishes_once(repository, tmp_path, initial):
    root, state, run = repository
    if not initial:
        run("git", "tag", "v0.1.0")
        service_change(root, run, "fix: repair")
    prepare(root, tmp_path)
    head = commit_release(run)
    assert prepare(root, tmp_path)[0] == "pending=false\n"
    version = "0.1.0" if initial else "0.1.1"
    publish(head)
    publish(head)
    assert [item["tag_name"] for item in state["releases"]] == ["v" + version]
    assert run("git", "rev-parse", f"refs/tags/v{version}^{{commit}}") == head
    assert head in run("git", "ls-remote", "--tags", "origin", f"refs/tags/v{version}^{{}}")
    manifest = json.loads(state["assets"]["v" + version])
    assert manifest["service_version"] == version
    assert manifest["source_revision"] == head and manifest["evaluator_build"] == "git:" + head
    assert manifest["client_version"] == release_manifest.client_version(root)
    assert run("git", "status", "--porcelain") == ""


def test_retry_after_tag_push_repairs_failed_release_creation(repository, tmp_path):
    root, state, run = repository
    prepare(root, tmp_path)
    head = commit_release(run)
    state["fail_create"] = True
    with pytest.raises(subprocess.CalledProcessError):
        publish(head)
    assert release_pr.stable_tags() == ["v0.1.0"]
    state["fail_create"] = False
    publish(head)
    assert len(state["releases"]) == 1


def test_bootstrap_release_can_be_retried_after_a_later_release(repository, tmp_path):
    root, state, run = repository
    prepare(root, tmp_path)
    initial = commit_release(run)
    publish(initial)
    service_change(root, run, "fix: later release")
    prepare(root, tmp_path)
    patch = commit_release(run)
    publish(patch)
    run("git", "checkout", initial)
    state["releases"][0]["assets"] = []
    del state["assets"]["v0.1.0"]
    publish(initial)
    assert json.loads(state["assets"]["v0.1.0"])["source_revision"] == initial
    assert len(state["releases"]) == 2


def test_client_only_changes_do_not_bump_service_but_mixed_changes_do(repository, tmp_path):
    root, _, run = repository
    run("git", "tag", "v0.1.0")
    cli = root / release_manifest.CLI_SCRIPT
    cli.write_text(cli.read_text() + "\n# Client feature\n")
    run("git", "add", release_manifest.CLI_SCRIPT)
    run("git", "commit", "-m", "feat(cli): new client option")
    assert prepare(root, tmp_path)[0] == "pending=false\n"
    cli.write_text(cli.read_text() + "# Companion client fix\n")
    run("git", "add", release_manifest.CLI_SCRIPT)
    service_change(root, run, "fix: server and client integration")
    assert "version=0.1.1\n" in prepare(root, tmp_path)[0]


def test_existing_asset_cannot_be_overwritten(repository, tmp_path):
    root, state, run = repository
    prepare(root, tmp_path)
    head = commit_release(run)
    publish(head)
    state["assets"]["v0.1.0"] = b"different artifact"
    with pytest.raises(ValueError, match="refusing to overwrite"):
        publish(head)
    assert state["assets"]["v0.1.0"] == b"different artifact"


def test_retry_can_restore_a_missing_manifest_upload(repository, tmp_path):
    root, state, run = repository
    prepare(root, tmp_path)
    head = commit_release(run)
    publish(head)
    state["releases"][0]["assets"] = []
    del state["assets"]["v0.1.0"]
    publish(head)
    assert json.loads(state["assets"]["v0.1.0"])["source_revision"] == head


def test_api_failure_does_not_create_a_tag(repository, tmp_path):
    root, state, run = repository
    prepare(root, tmp_path)
    head = commit_release(run)
    state["api_error"] = True
    with pytest.raises(subprocess.CalledProcessError):
        publish(head)
    assert release_pr.stable_tags() == []


def test_changed_tree_since_review_refuses_publication(repository, tmp_path):
    root, state, run = repository
    before = run("git", "rev-parse", "HEAD")
    prepare(root, tmp_path)
    commit_release(run)
    with pytest.raises(subprocess.CalledProcessError):
        publish(before)
    assert not state["calls"] and release_pr.stable_tags() == []


def test_code_changes_inside_release_pr_are_refused(repository, tmp_path):
    root, state, run = repository
    prepare(root, tmp_path)
    (root / "triton_evaluation/__init__.py").write_text("changed = True\n")
    run("git", "add", "triton_evaluation/__init__.py")
    head = commit_release(run)
    with pytest.raises(ValueError, match="only pyproject"):
        publish(head)
    assert not state["calls"]


def test_existing_version_tag_is_never_moved(repository, tmp_path):
    root, state, run = repository
    prepare(root, tmp_path)
    old = run("git", "rev-parse", "HEAD")
    run("git", "tag", "v0.1.0", old)
    head = commit_release(run)
    with pytest.raises(ValueError, match="refusing to move"):
        publish(head)
    assert run("git", "rev-parse", "refs/tags/v0.1.0") == old
    assert not state["calls"]


def test_installed_version_mismatch_prevents_publication(repository, tmp_path, monkeypatch):
    root, state, run = repository
    prepare(root, tmp_path)
    head = commit_release(run)
    monkeypatch.setattr(versioning, "service_version", lambda: "9.9.9")
    with pytest.raises(ValueError, match="installed-service-version-mismatch"):
        publish(head)
    assert release_pr.stable_tags() == [] and not state["calls"]


def test_stale_bootstrap_stays_draft_until_refreshed(repository, tmp_path):
    root, state, run = repository
    base = run("git", "rev-parse", "HEAD")
    run("git", "commit", "--allow-empty", "-m", "docs: advance main")
    run("git", "push", "origin", "main")
    state["pr"] = {"state": "OPEN", "isDraft": True,
                   "headRefName": "automation/release", "headRefOid": "a" * 40}
    output = tmp_path / "outputs"
    release_pr.check_base(base, "0.1.0", "example/evaluation", output, "7", "a" * 40)
    assert output.read_text() == "current=false\n"
    assert state["pr"]["state"] == "OPEN"
    release_pr.check_base(run("git", "rev-parse", "HEAD"), "0.1.0", "example/evaluation",
                          output, "7", "a" * 40)
    assert output.read_text() == "current=true\n"


def test_changed_pr_head_cannot_be_marked_ready(repository, tmp_path):
    _, state, run = repository
    state["pr"] = {"state": "OPEN", "isDraft": True,
                   "headRefName": "automation/release", "headRefOid": "b" * 40}
    output = tmp_path / "outputs"
    release_pr.check_base(run("git", "rev-parse", "HEAD"), "0.1.0", "example/evaluation",
                          output, "7", "a" * 40)
    assert output.read_text() == "current=false\n"
