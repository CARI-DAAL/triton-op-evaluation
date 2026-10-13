"""The plugin artifact remains usable without the service checkout or dependencies."""
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1] / "plugins" / "triton-eval"


def test_plugin_archive_discovers_skill_and_runs_isolated_client(tmp_path):
    archive_path = tmp_path / "plugin.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for path in PLUGIN_ROOT.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.write(path, path.relative_to(PLUGIN_ROOT))
    isolated = tmp_path / "extracted"
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(isolated)

    manifest = json.loads((isolated / "plugin.json").read_text())
    assert manifest["$schema"] == "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json"
    assert manifest["version"] == "0.4.0"
    skills = set((isolated / "skills").glob("*/SKILL.md"))
    assert skills
    for directory in (".codex-plugin", ".claude-plugin"):
        overlay_file = isolated / directory / "plugin.json"
        assert overlay_file.is_file(), "archive omitted client compatibility metadata"
        overlay = json.loads(overlay_file.read_text())
        for name in ("name", "version", "description", "repository"):
            assert overlay[name] == manifest[name], f"{directory}: metadata drift for {name}"
        skill_path = overlay["skills"]
        assert skill_path.startswith("./")
        skill_root = (isolated / skill_path).resolve()
        assert skill_root.is_relative_to(isolated)
        assert set(skill_root.glob("*/SKILL.md")) == skills
        assert not (isolated / directory / "skills").exists()
    for skill in skills:
        for link in re.findall(r"\]\(([^)]+)\)", skill.read_text()):
            if "://" not in link:
                target = (skill.parent / link).resolve()
                assert target.is_relative_to(isolated)
                assert target.is_file(), link

    scripts = list((isolated / "skills").glob("*/scripts/*.py"))
    assert len(scripts) == 1
    for arguments in [
        [], ["verify"], ["benchmark"], ["profile"], ["evaluate"], ["get"], ["progress"],
    ]:
        result = subprocess.run(
            [sys.executable, "-I", "-S", str(scripts[0]), *arguments, "--help"],
            cwd=tmp_path, capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout
