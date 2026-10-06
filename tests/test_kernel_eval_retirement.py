"""Historical bridge evidence cannot provide an executable deployment entry."""

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_bridge_entry_and_config_are_removed_but_archived_bytes_remain():
    retired = ROOT / "adapters/kernel-eval-legacy-bridge"
    # Removed entirely upstream (PR #38); only the non-executable archive remains.
    assert not retired.exists()
    archive = ROOT / "docs/archives/kernel-eval-legacy-bridge"
    manifest = json.loads((archive / "ARCHIVE-MANIFEST.json").read_text())
    for name, record in manifest["files"].items():
        assert record["executable"] is False
        archived = ROOT / record["archived_as"]
        assert archived.suffix == ".txt"
        assert hashlib.sha256(archived.read_bytes()).hexdigest() == record["sha256"]
        assert not (retired / name).exists() or name == "README.md"


def test_native_runtime_has_no_import_or_path_dependency_on_the_retired_adapter():
    for path in (ROOT / "triton_evaluation").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert "legacy_contract" not in (node.module or "")
            elif isinstance(node, ast.Import):
                assert all("legacy_contract" not in alias.name for alias in node.names)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "Path"
            ):
                for argument in node.args:
                    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                        assert "open-evo-bridge" not in argument.value
                        assert "kernel-eval-legacy-bridge" not in argument.value
