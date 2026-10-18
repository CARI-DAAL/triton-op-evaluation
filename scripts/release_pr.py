"""Maintain a reviewed release PR and publish its exact merged source commit.

Preparation only edits pyproject.toml and CHANGELOG.md. GitHub Actions owns PR
updates and CI dispatch. Publication never changes main or moves an existing tag.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

from scripts import release_manifest

ROOT = release_manifest.PROJECT_ROOT
VERSION = r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
CONVENTIONAL = re.compile(r"^(\w+)(?:\([^\n]+\))?(!)?: .+")
BREAKING = re.compile(r"^BREAKING[ -]CHANGE: ", re.MULTILINE)
RELEASE_BRANCH = "automation/release"
SERVICE_PATHS = ("triton_evaluation", "scripts", "pyproject.toml", "docker", "adapters",
                 ".env.example")


def run(*args: str) -> str:
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=True)
    return result.stdout.strip()


def version_parts(version: str) -> tuple[int, ...]:
    if not re.fullmatch(VERSION, version):
        raise ValueError(f"not a stable release version: {version!r}")
    return tuple(map(int, version.split(".")))


def suggested_bump(current: str, messages: list[str]) -> str | None:
    """Match OpenEvo's policy; pre-1.0 breaking changes stay below 1.0."""
    version_parts(current)
    result = None
    for message in messages:
        match = CONVENTIONAL.match(message)
        if match is None:
            continue
        if match[2] or BREAKING.search(message):
            return "minor" if current.startswith("0.") else "major"
        if match[1] == "feat":
            result = "minor"
        elif match[1] in {"fix", "perf", "revert"} and result is None:
            result = "patch"
    return result


def bumped(current: str, part: str) -> str:
    major, minor, patch = version_parts(current)
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def stable_tags(*, merged: str | None = None) -> list[str]:
    # Backup/archive tags must never become a release baseline.
    options = ("--merged", merged) if merged is not None else ()
    return [tag for tag in run("git", "tag", "--list", *options).splitlines()
            if re.fullmatch("v" + VERSION, tag)]


def release_entry(changelog: str, version: str) -> str | None:
    match = re.search(
        rf"^## {re.escape(version)} — \d{{4}}-\d{{2}}-\d{{2}}\n(.*?)(?=^## |\Z)",
        changelog, re.MULTILINE | re.DOTALL,
    )
    return match[1].strip() if match else None


def prepare(body_file: Path, output_file: Path) -> None:
    current = release_manifest.pyproject_version(ROOT)
    version_parts(current)
    release_manifest.client_version(ROOT)
    tags = stable_tags()
    tag = f"v{current}"
    changelog_path = ROOT / "CHANGELOG.md"
    changelog = changelog_path.read_text()
    if tag not in tags:
        if tags or release_entry(changelog, current) is not None:
            # Includes the first release PR merged while publication is pending.
            output_file.write_text("pending=false\n")
            print(f"Waiting for {tag} publication; no new release PR prepared.")
            return
        version, part, revision_range = current, "initial release", "HEAD"
    else:
        run("git", "merge-base", "--is-ancestor", f"refs/tags/{tag}", "HEAD")
        revision_range = f"refs/tags/{tag}..HEAD"
        messages = run("git", "log", revision_range, "--format=%B%x00",
                       "--", *SERVICE_PATHS).split("\0")
        part = suggested_bump(current, [message.strip() for message in messages])
        if part is None:
            output_file.write_text("pending=false\n")
            print("No unreleased feature, fix, performance improvement or breaking change.")
            return
        version = bumped(current, part)
    if release_entry(changelog, version) is not None:
        raise ValueError("changelog already contains the proposed release")
    date = run("git", "show", "-s", "--format=%cs", "HEAD")
    base_sha = run("git", "rev-parse", "HEAD")
    commits = run("git", "log", revision_range, "--no-merges", "--max-count=100",
                  "--format=- %h %s")
    if "## Unreleased\n" not in changelog:
        raise ValueError("CHANGELOG.md requires a ## Unreleased section")
    # Preserve hand-written migration notes in Unreleased when rolling the entry.
    prefix, unreleased = changelog.split("## Unreleased\n", 1)
    following = re.search(r"^## ", unreleased, re.MULTILINE)
    notes = unreleased[:following.start()] if following else unreleased
    older = unreleased[following.start():] if following else ""
    body = notes.strip()
    if body:
        body += "\n\n"
    body += "### Commits\n\n" + commits + "\n"
    changelog_path.write_text(
        prefix + f"## Unreleased\n\n## {version} — {date}\n\n" + body + "\n" + older
    )
    project = ROOT / "pyproject.toml"
    source, count = re.subn(
        r'(?ms)(^\[project\]\n(?:(?!^\[).)*?^version = ")[^"]+("$)',
        lambda match: match[1] + version + match[2], project.read_text(), count=1,
    )
    if count != 1:
        raise ValueError("cannot update the project version")
    project.write_text(source)
    body_file.write_text(
        f"## Summary\n\nPrepare service **{version}** ({part}; current version {current}).\n\n"
        "```text\nmain changes → release PR → review + CI → merge → tag + Release\n```\n\n"
        "Merging this PR authorizes publication of the reviewed source commit as a GitHub "
        "Release with its bound release manifest. It does not publish to PyPI or deploy NPUs. "
        "The client version is recorded independently and is not bumped here.\n\n"
        f"### Included commits (up to 100)\n\n{commits}\n\n"
        "## Evidence\n\n"
        "Review the changelog and version diff and wait for the CPU matrix on the latest "
        "PR commit. Device acceptance remains a separate release decision.\n\n"
        "## Merge Danger\n\nMerging publishes an immutable version tag. Corrections require "
        "another release; existing tags are never moved.\n"
    )
    output_file.write_text(f"pending=true\nversion={version}\nbase_sha={base_sha}\n")


def check_base(base_sha: str, version: str, repo: str, output_file: Path,
               pr: str | None = None, head_sha: str | None = None) -> None:
    """Leave stale preparation in draft; never expose it as ready for review."""
    version_parts(version)
    run("git", "fetch", "--no-tags", "origin", "main")
    latest = run("git", "rev-parse", "FETCH_HEAD")
    details = None
    if pr:
        details = json.loads(run("gh", "pr", "view", pr, "--repo", repo,
                                 "--json", "state,isDraft,headRefName,headRefOid"))
        if (details["state"] != "OPEN" or not details["isDraft"]
                or details["headRefName"] != RELEASE_BRANCH
                or details["headRefOid"] != head_sha):
            output_file.write_text("current=false\n")
            return
    if latest == base_sha:
        output_file.write_text("current=true\n")
        return
    output_file.write_text("current=false\n")
    print("main advanced; a subsequent run will refresh the release PR.")
    source = run("git", "show", f"{latest}:pyproject.toml")
    current = re.search(r'^version = "([^"]+)"', source, re.MULTILINE)[1]
    released = version_parts(current) > version_parts(version)
    if details and current == version:
        released = release_entry(run("git", "show", f"{latest}:CHANGELOG.md"), version) is not None
    if details and released:
        # A merge may have happened between the first check and create-pull-request.
        run("gh", "pr", "close", pr, "--repo", repo)


def publish(repo: str, reviewed_head: str) -> None:
    """Publish exactly the reviewed tree; retry partial tag/release creation safely."""
    if not re.fullmatch(r"[0-9a-f]{40}", reviewed_head):
        raise ValueError("reviewed head must be a commit SHA")
    commit = run("git", "rev-parse", "HEAD")
    run("git", "merge-base", "--is-ancestor", commit, "refs/remotes/origin/main")
    run("git", "diff", "--exit-code", reviewed_head, commit)
    changed = set(run("git", "diff", "--name-only", "HEAD^", "HEAD").splitlines())
    if not changed or changed - {"pyproject.toml", "CHANGELOG.md"}:
        raise ValueError("release PR must change only pyproject.toml and CHANGELOG.md")
    version = release_manifest.pyproject_version(ROOT)
    version_parts(version)
    previous_project = run("git", "show", "HEAD^:pyproject.toml")
    previous = re.search(r'^version = "([^"]+)"', previous_project, re.MULTILINE)[1]
    previous_changelog = run("git", "show", "HEAD^:CHANGELOG.md")
    notes = release_entry((ROOT / "CHANGELOG.md").read_text(), version)
    if not notes:
        raise ValueError("changelog has no nonempty entry for the release version")
    tag = f"v{version}"
    tags = stable_tags()
    if tag in tags:
        if run("git", "rev-parse", f"refs/tags/{tag}^{{commit}}") != commit:
            raise ValueError(f"{tag} belongs to another commit; refusing to move it")
    bootstrap = (version == previous and not stable_tags(merged="HEAD^")
                 and release_entry(previous_changelog, version) is None)
    if not bootstrap and version_parts(version) <= version_parts(previous):
        raise ValueError("release commit must increase the service version")
    # Check the actual installed distribution and client versions before publishing.
    manifest = release_manifest.generate(ROOT, revision=commit)
    manifest = release_manifest._import_versioning().bind_build(
        manifest, evaluator_build=f"git:{commit}",
    )
    with tempfile.TemporaryDirectory(prefix="triton-release-") as directory:
        artifact = Path(directory) / "release-manifest.json"
        release_manifest._emit(manifest, artifact)
        note_path = Path(directory) / "notes.md"
        note_path.write_text(notes + "\n")
        # API/list errors fail closed, rather than treating auth/network errors as absence.
        releases = json.loads(run("gh", "api", "--paginate", "--slurp",
                                  f"repos/{repo}/releases?per_page=100"))
        existing = next((item for page in releases for item in page
                         if item["tag_name"] == tag), None)
        if existing is not None and existing["draft"]:
            raise ValueError("an existing draft release requires maintainer review")
        if tag not in tags:
            run("git", "tag", "-a", tag, "-m", f"Release {version}")
        run("git", "push", "origin", f"refs/tags/{tag}")
        if existing is None:
            run("gh", "release", "create", tag, str(artifact), "--repo", repo,
                "--verify-tag", "--title", tag, "--notes-file", str(note_path))
        else:
            # Never overwrite an existing release asset. Repair a missing upload only.
            if any(asset["name"] == artifact.name for asset in existing["assets"]):
                target = Path(directory) / "existing"
                target.mkdir()
                run("gh", "release", "download", tag, "--repo", repo,
                    "--pattern", artifact.name, "--dir", str(target))
                if (target / artifact.name).read_bytes() != artifact.read_bytes():
                    raise ValueError("existing release manifest differs; refusing to overwrite")
            else:
                run("gh", "release", "upload", tag, str(artifact), "--repo", repo)
    print(f"Published {tag} at {commit}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--body-file", type=Path, required=True)
    prepare_parser.add_argument("--output-file", type=Path, required=True)
    checker = commands.add_parser("check-base")
    checker.add_argument("--base-sha", required=True)
    checker.add_argument("--version", required=True)
    checker.add_argument("--repo", required=True)
    checker.add_argument("--output-file", type=Path, required=True)
    checker.add_argument("--pr")
    checker.add_argument("--head-sha")
    publisher = commands.add_parser("publish")
    publisher.add_argument("--repo", required=True)
    publisher.add_argument("--reviewed-head", required=True)
    args = parser.parse_args(argv)
    if args.command == "check-base":
        if bool(args.pr) != bool(args.head_sha):
            parser.error("--pr and --head-sha must be supplied together")
        check_base(args.base_sha, args.version, args.repo, args.output_file, args.pr, args.head_sha)
        return
    if run("git", "status", "--porcelain"):
        raise ValueError("use a clean checkout")
    if args.command == "prepare":
        prepare(args.body_file, args.output_file)
    else:
        publish(args.repo, args.reviewed_head)


if __name__ == "__main__":
    main()
