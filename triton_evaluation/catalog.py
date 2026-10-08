"""Trusted release catalog for grade admission (design section 3.2).

The catalog is a read-only mounted release directory with one manifest of the
fixed shape ``{"schema_version": "1.0", "entries": [{"manifest_id",
"problem_id", "reference_digest", "reference_path"}]}``. Entries are immutable:
a manifest ID may publish many problems, but each ``(manifest_id, problem_id)``
pair appears once and can never rebind to different reference bytes. Reference
artifacts are validated at load time — resolved under the catalog root with no
absolute paths, ``..`` traversal, or symlink escapes; regular files only;
valid UTF-8; and hashed over their exact bytes. Error messages never include
reference paths, artifact contents, or caller-supplied identifiers.
"""

import ast
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field, ValidationError

from triton_evaluation.api.contract import GradeContext, StrictModel
from triton_evaluation.serialization import sha256_hex


class ManifestBindingError(Exception):
    """Public ``MANIFEST_BINDING_FAILED`` rejection, generic on purpose."""

    code = "MANIFEST_BINDING_FAILED"
    phase = "admission"
    retryable = False

    def __init__(self) -> None:
        super().__init__("reference does not match the bound manifest entry")


@dataclass(frozen=True)
class CatalogBinding:
    manifest_id: str
    problem_id: str
    reference_digest: str


@dataclass(frozen=True)
class OfficialCatalogTask:
    problem_id: str
    op_name: str
    reference_code: str
    grade_context: GradeContext
    official_digest: str
    adapter_version: str
    cases: int


class _CatalogEntry(StrictModel):
    manifest_id: str = Field(min_length=1, max_length=128)
    problem_id: str = Field(min_length=1, max_length=128)
    reference_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_path: str = Field(min_length=1, max_length=512)


class _CatalogFile(StrictModel):
    schema_version: Literal["1.0"]
    entries: list[_CatalogEntry]


def _resolve_under(root: Path, candidate: Path, what: str) -> Path:
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        raise ValueError(f"{what} is missing or unreadable") from None
    if not resolved.is_relative_to(root):
        raise ValueError(f"{what} escapes the catalog root")
    return resolved


class ReleaseCatalog:
    """Validated immutable view over the mounted release catalog."""

    def __init__(
        self, entries: dict[tuple[str, str], _CatalogEntry], *, root: Path | None = None
    ) -> None:
        self._entries = entries
        self._root = root

    @classmethod
    def load(cls, root: Path, manifest_path: Path) -> "ReleaseCatalog":
        """Parse the manifest and validate every referenced artifact."""
        try:
            root = Path(root).resolve(strict=True)
        except OSError:
            raise ValueError("catalog root is missing or unreadable") from None
        if not root.is_dir():
            raise ValueError("catalog root is not a directory")
        manifest = _resolve_under(root, Path(manifest_path), "catalog manifest")
        if not manifest.is_file():
            raise ValueError("catalog manifest is not a regular file")
        try:
            payload = json.loads(manifest.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError("catalog manifest is not valid UTF-8 JSON") from None
        try:
            parsed = _CatalogFile.model_validate(payload)
        except ValidationError:
            raise ValueError("catalog manifest failed schema validation") from None
        entries: dict[tuple[str, str], _CatalogEntry] = {}
        for entry in parsed.entries:
            key = (entry.manifest_id, entry.problem_id)
            if key in entries:
                raise ValueError("duplicate catalog entry for one manifest/problem pair")
            cls._validate_reference(root, entry)
            entries[key] = entry
        return cls(entries, root=root)

    def resolve_official_task(
        self, task_id: str, *, manifest_id: str | None = None
    ) -> OfficialCatalogTask:
        """Resolve an exact official ID, validating its artifact and sidecar.

        No official source checkout or NPU dependency is needed at serving
        time. Source replay belongs to the registration/replay CLI. Ambiguous,
        missing or inconsistent entries share the public binding rejection.
        """
        if self._root is None or not task_id.startswith(("official_l3_", "official_l4_")):
            raise ManifestBindingError()
        entries = [
            entry
            for entry in self._entries.values()
            if entry.problem_id == task_id
            and (manifest_id is None or entry.manifest_id == manifest_id)
        ]
        if len(entries) != 1:
            raise ManifestBindingError()
        entry = entries[0]
        try:
            self._validate_reference(self._root, entry)
            reference_path = _resolve_under(
                self._root, self._root / entry.reference_path, "catalog reference artifact"
            )
            sidecar_path = _resolve_under(
                self._root,
                self._root / f"{entry.reference_path}.PROVENANCE.json",
                "official provenance",
            )
            if not sidecar_path.is_file():
                raise ValueError("official provenance is not a regular file")
            provenance = json.loads(sidecar_path.read_bytes())
            source = reference_path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            header = ast.get_docstring(tree, clean=False) or ""
            if (
                not isinstance(provenance, dict)
                or provenance.get("schema_version") != "1.0"
                or provenance.get("manifest_id") != entry.manifest_id
                or provenance.get("problem_id") != entry.problem_id
                or provenance.get("reference_digest") != entry.reference_digest
                or not isinstance(provenance.get("op_name"), str)
                or not provenance["op_name"]
                or not isinstance(provenance.get("adapter_version"), str)
                or not provenance["adapter_version"]
                or not isinstance(provenance.get("cases"), int)
                or isinstance(provenance["cases"], bool)
                or provenance["cases"] < 1
            ):
                raise ValueError("official provenance fields do not match the catalog")
            official_digest = provenance.get("official_digest")
            if (
                not isinstance(official_digest, str)
                or len(official_digest) != 64
                or any(c not in "0123456789abcdef" for c in official_digest)
                or header.splitlines()[:3]
                != [
                    "Composed by triton_evaluation.official_task " + provenance["adapter_version"],
                    "problem_id: " + entry.problem_id,
                    "official_digest: " + official_digest,
                ]
            ):
                raise ValueError("official provenance digest does not match the reference header")
            context = GradeContext(
                manifest_id=entry.manifest_id,
                problem_id=entry.problem_id,
                reference_digest=entry.reference_digest,
            )
            # Re-read/re-hash above, then retain the existing binding semantics.
            self.bind(context, source)
        except (OSError, ValueError, TypeError, SyntaxError, KeyError):
            raise ManifestBindingError() from None
        return OfficialCatalogTask(
            problem_id=entry.problem_id,
            op_name=provenance["op_name"],
            reference_code=source,
            grade_context=context,
            official_digest=official_digest,
            adapter_version=provenance["adapter_version"],
            cases=provenance["cases"],
        )

    @staticmethod
    def _validate_reference(root: Path, entry: _CatalogEntry) -> None:
        relative = PurePosixPath(entry.reference_path)
        if relative.is_absolute():
            raise ValueError("catalog reference path must be relative")
        if ".." in relative.parts:
            raise ValueError("catalog reference path must not contain '..'")
        resolved = _resolve_under(root, root / entry.reference_path, "catalog reference artifact")
        if not resolved.is_file():
            raise ValueError("catalog reference artifact is not a regular file")
        blob = resolved.read_bytes()
        if sha256_hex(blob) != entry.reference_digest:
            raise ValueError("catalog reference artifact digest mismatch")
        try:
            blob.decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError("catalog reference artifact is not valid UTF-8") from None

    def bind(self, context: GradeContext, reference_code: str) -> CatalogBinding:
        """Bind a grade request to its immutable catalog entry.

        The ``(manifest_id, problem_id, reference_digest)`` triple must match a
        catalog entry exactly, and the request body's ``reference_code`` UTF-8
        bytes must hash to the bound digest. Both failures raise the same
        generic :class:`ManifestBindingError`.
        """
        entry = self._entries.get((context.manifest_id, context.problem_id))
        if entry is None or entry.reference_digest != context.reference_digest:
            raise ManifestBindingError()
        if sha256_hex(reference_code.encode("utf-8")) != entry.reference_digest:
            raise ManifestBindingError()
        return CatalogBinding(
            manifest_id=entry.manifest_id,
            problem_id=entry.problem_id,
            reference_digest=entry.reference_digest,
        )
