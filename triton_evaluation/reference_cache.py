"""F-63 immutable, per-card reference evidence (JSON and tensor bytes, never pickle).

Only audited official tasks are eligible. Enabling the cache pins an index; a missing,
changed or incompatible official entry raises instead of silently selecting an answer.
Measurement invalidation is a separate, monotone per-card/generation marker.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import math
import os
import re
import statistics
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CACHE_ENV = "TRITON_EVAL_REFERENCE_CACHE_DIR"
BINDING_ENV = "TRITON_EVAL_REFERENCE_CACHE_BINDING"
DIGEST_ENV = "TRITON_EVAL_REFERENCE_CACHE_SHA256"
STATE_ENV = "TRITON_EVAL_REFERENCE_CACHE_STATE_DIR"
DEVICE_ENV = "TRITON_EVAL_REFERENCE_CACHE_DEVICE_ID"
CACHE_ENV_KEYS = frozenset({CACHE_ENV, BINDING_ENV, DIGEST_ENV, STATE_ENV})
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_DTYPES = frozenset(
    {
        "bool",
        "uint8",
        "int8",
        "int16",
        "int32",
        "int64",
        "float16",
        "float32",
        "float64",
        "bfloat16",
        "complex64",
        "complex128",
    }
)


class ReferenceCacheMiss(LookupError):
    """No audited entry: execute the original reference and record a miss."""


class ReferenceCacheError(RuntimeError):
    """Untrusted or incompatible reference evidence; never a candidate verdict."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class CacheMeasurement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    method: Literal["event", "profiler"]
    warmup: int = Field(ge=0)
    repeats: int = Field(ge=1)
    cv_threshold: float = Field(gt=0, le=1)
    clear_l2_cache: bool = False


class CacheDeployment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    generation: str = Field(min_length=1)
    runtime_versions: dict[str, str] = Field(min_length=1)
    # Operator supplies the actual immutable CANN/driver version files, not labels alone.
    runtime_files: dict[str, str] = Field(min_length=1)
    # Engine basenames, plus the one explicitly named sibling runtime helper.
    algorithm_files: dict[str, str] = Field(min_length=1)
    device_conditions: dict[str, dict[str, str]] = Field(min_length=1)
    measurement: CacheMeasurement

    @model_validator(mode="after")
    def validate_digests(self):
        for path, digest in self.runtime_files.items():
            if not Path(path).is_absolute() or not _DIGEST.fullmatch(digest):
                raise ValueError("runtime files require absolute paths and SHA256")
        for path, digest in self.algorithm_files.items():
            if (
                (Path(path).name != path or path in {".", ".."})
                and path != "../profiler_runtime.py"
            ) or not _DIGEST.fullmatch(digest):
                raise ValueError("algorithm files require engine basenames and SHA256")
        if not {"verify.py", "benchmark.py", "_common_utils.py"} <= self.algorithm_files.keys():
            raise ValueError("verify, benchmark and common algorithms must all be pinned")
        if any(
            not card.isdigit() or not condition
            for card, condition in self.device_conditions.items()
        ):
            raise ValueError("device conditions require physical numeric card identities")
        return self

    @property
    def digest(self) -> str:
        return sha256(canonical_bytes(self.model_dump(mode="json")))

    def check_runtime(self, engine_dir: Path | None = None) -> None:
        for distribution, expected in self.runtime_versions.items():
            try:
                observed = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError as exc:
                raise ReferenceCacheError(
                    f"cache runtime distribution absent: {distribution}"
                ) from exc
            if observed != expected:
                raise ReferenceCacheError(f"cache runtime version changed: {distribution}")
        for path, expected in self.runtime_files.items():
            if sha256(Path(path).read_bytes()) != expected:
                raise ReferenceCacheError("cache runtime file changed")
        engine_dir = engine_dir or Path(__file__).parent / "engine"
        from .profiler_runtime import (
            DIGEST_ENV,
            MANIFEST_ENV,
            RUNTIME_ENV_KEYS,
            load_runtime,
            read_cann_environment,
        )

        if any(key in os.environ for key in RUNTIME_ENV_KEYS):
            if "../profiler_runtime.py" not in self.algorithm_files:
                raise ReferenceCacheError(
                    "cache must pin the configured profiler runtime algorithm"
                )
            try:
                profiler = load_runtime()
            except (OSError, ValueError) as exc:
                raise ReferenceCacheError(
                    "cache profiler runtime configuration is invalid"
                ) from exc
            if profiler is None:
                raise ReferenceCacheError("cache profiler runtime configuration is absent")
            required_runtime = {os.environ[MANIFEST_ENV]: os.environ[DIGEST_ENV]}
            for field in ("library", "consumer", "msprof", "alias", "elf_observations"):
                required_runtime[str(getattr(profiler, field + "_path"))] = getattr(
                    profiler, field + "_sha256"
                )
            cann = read_cann_environment(profiler)
            if cann is not None:
                for field in ("cann_set_env", "cann_environment"):
                    required_runtime[str(getattr(profiler, field + "_path"))] = getattr(
                        profiler, field + "_sha256"
                    )
                required_runtime.update(cann.runtime_files)
            if any(
                self.runtime_files.get(path) != digest for path, digest in required_runtime.items()
            ):
                raise ReferenceCacheError("cache must pin every configured profiler runtime file")
        for name, expected in self.algorithm_files.items():
            algorithm = engine_dir / name
            if name == "../profiler_runtime.py":
                # This is the one fixed service-owned sibling, never a general
                # traversal permission or part of candidate material.
                root = engine_dir.parent.resolve(strict=True)
                algorithm = engine_dir.parent / "profiler_runtime.py"
                if (
                    engine_dir.resolve(strict=True) != engine_dir.absolute()
                    or algorithm.is_symlink()
                    or algorithm.resolve(strict=True) != root / "profiler_runtime.py"
                ):
                    raise ReferenceCacheError(
                        "profiler runtime algorithm must stay in service root"
                    )
            if sha256(algorithm.read_bytes()) != expected:
                raise ReferenceCacheError(f"cache algorithm changed: {name}")


class CacheIndex(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["reference-cache-index-v1"] = "reference-cache-index-v1"
    deployment: CacheDeployment
    entries: dict[str, str]

    @model_validator(mode="after")
    def validate_entries(self):
        if any(
            not _DIGEST.fullmatch(key) or not _DIGEST.fullmatch(value)
            for key, value in self.entries.items()
        ):
            raise ValueError("cache index entries require SHA256 keys and manifest digests")
        return self


class CaseBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    deployment_sha256: str
    reference_source_sha256: str
    official_digest: str
    adapter_version: str
    input_seed: Literal[0] = 0
    case_idx: int = Field(ge=1)
    input_sha256: str
    device_id: str
    reference_device_policy: Literal["cpu-first", "npu-strict"]

    @property
    def key(self) -> str:
        return sha256(canonical_bytes(self.model_dump(mode="json")))


class CaseManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["reference-case-v1"] = "reference-case-v1"
    binding: CaseBinding
    selected_reference_device: str
    outputs: dict[str, Any]
    output_sha256: str
    latency_samples_ms: list[float] = Field(min_length=8, max_length=8)
    reference_latency_median_ms: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_evidence(self):
        if any(not math.isfinite(value) or value <= 0 for value in self.latency_samples_ms):
            raise ValueError("reference measurements must be finite and positive")
        if self.reference_latency_median_ms != statistics.median(self.latency_samples_ms):
            raise ValueError("reference denominator must be median of eight measurements")
        if self.output_sha256 != sha256(canonical_bytes(self.outputs)):
            raise ValueError("output structure hash changed")
        if self.binding.reference_device_policy == "npu-strict" and not (
            self.selected_reference_device == "npu"
            or self.selected_reference_device.startswith("npu:")
        ):
            raise ValueError("strict NPU cache cannot contain a CPU reference")
        return self


def official_source_metadata(source: str) -> tuple[str, str] | None:
    """Recognize the assembler's literal metadata without executing task source."""
    digest_match = re.search(r"^official_digest: ([0-9a-f]{64})$", source, re.MULTILINE)
    if digest_match is None:
        return None
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_ADAPTER_VERSION"
            for target in node.targets
        ):
            version = ast.literal_eval(node.value)
            if isinstance(version, str) and version:
                return digest_match[1], version
    raise ReferenceCacheError("official reference has no literal adapter version")


def _tensor_bytes(value) -> bytes:
    import torch

    if value.layout != torch.strided or value.is_quantized:
        raise ReferenceCacheError("reference cache supports dense non-quantized tensors only")
    return value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def _encode(value, blobs: dict[str, bytes] | None = None, *, input_value: bool = False) -> dict:
    import torch

    if isinstance(value, torch.Tensor):
        dtype = str(value.dtype).removeprefix("torch.")
        if dtype not in _DTYPES:
            raise ReferenceCacheError(f"unsupported cached tensor dtype: {dtype}")
        payload = _tensor_bytes(value)
        digest = sha256(payload)
        if blobs is not None:
            blobs[digest] = payload
        result = {
            "kind": "tensor",
            "dtype": dtype,
            "shape": list(value.shape),
            "sha256": digest,
            "nbytes": len(payload),
        }
        if input_value:
            result.update(
                stride=list(value.stride()),
                storage_offset=value.storage_offset(),
                device_type=value.device.type,
            )
        return result
    if isinstance(value, (list, tuple)):
        return {
            "kind": "tuple" if isinstance(value, tuple) else "list",
            "items": [_encode(item, blobs, input_value=input_value) for item in value],
        }
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {
            "kind": "dict",
            "items": {
                key: _encode(item, blobs, input_value=input_value) for key, item in value.items()
            },
        }
    if input_value and type(value) is object:
        # The official adapter's singleton _ABSENT for omitted keyword parameters.
        return {"kind": "absent"}
    if value is None or type(value) in {str, int, bool}:
        return {"kind": "scalar", "value": value}
    if type(value) is float:
        return {"kind": "float", "hex": value.hex()}
    raise ReferenceCacheError(f"unsupported cached value type: {type(value).__name__}")


def value_sha256(value, *, input_value: bool = False) -> str:
    return sha256(canonical_bytes(_encode(value, input_value=input_value)))


def _read_blob(root: Path, digest: str) -> bytes:
    if not _DIGEST.fullmatch(digest):
        raise ReferenceCacheError("invalid tensor blob identity")
    path = root / "blobs" / f"{digest}.bin"
    if path.is_symlink() or path.parent.is_symlink():
        raise ReferenceCacheError("reference blob must not be a symlink")
    payload = path.read_bytes()
    if sha256(payload) != digest:
        raise ReferenceCacheError("reference tensor blob hash mismatch")
    return payload


def _decode(node: dict, root: Path):
    import torch

    kind = node.get("kind")
    if kind == "tensor":
        dtype_name = node.get("dtype")
        shape = node.get("shape")
        if (
            dtype_name not in _DTYPES
            or not isinstance(shape, list)
            or any(type(dim) is not int or dim < 0 for dim in shape)
        ):
            raise ReferenceCacheError("invalid cached tensor shape or dtype")
        dtype = getattr(torch, dtype_name)
        payload = _read_blob(root, node["sha256"])
        element_size = torch.empty((), dtype=dtype, device="cpu").element_size()
        if len(payload) != math.prod(shape) * element_size or node.get("nbytes") != len(payload):
            raise ReferenceCacheError("cached tensor shape/dtype does not match bytes")
        if not payload:
            return torch.empty(shape, dtype=dtype, device="cpu")
        # Own writable storage: candidate/view mutation never changes evidence or another load.
        return (
            torch.frombuffer(bytearray(payload), dtype=torch.uint8)
            .clone()
            .view(dtype)
            .reshape(shape)
        )
    if kind in {"list", "tuple"}:
        items = [_decode(item, root) for item in node["items"]]
        return tuple(items) if kind == "tuple" else items
    if kind == "dict":
        return {key: _decode(item, root) for key, item in node["items"].items()}
    if kind == "scalar":
        return node["value"]
    if kind == "float":
        return float.fromhex(node["hex"])
    raise ReferenceCacheError("invalid cached output structure")


def _atomic_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.parent.is_symlink():
        raise ReferenceCacheError("cache artifacts must not be symlinks")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != payload:
                    raise ReferenceCacheError(
                        "immutable cache artifact already has different bytes"
                    ) from None
        finally:
            temporary.unlink(missing_ok=True)


class ReferenceCache:
    def __init__(
        self,
        root: Path,
        deployment: CacheDeployment,
        device_id: str,
        *,
        entries: dict[str, str] | None = None,
        state_dir: Path | None = None,
    ):
        self.root = Path(root)
        self.deployment = deployment
        self.device_id = str(device_id)
        self.entries = entries
        self.state_dir = Path(state_dir) if state_dir is not None else None

    @classmethod
    def from_environment(cls) -> ReferenceCache | None:
        present = {key: os.environ[key] for key in CACHE_ENV_KEYS if key in os.environ}
        if not present:
            return None
        if present.keys() != CACHE_ENV_KEYS:
            raise ReferenceCacheError("all reference cache deployment variables are required")
        if any(not Path(present[key]).is_absolute() for key in (CACHE_ENV, BINDING_ENV, STATE_ENV)):
            raise ReferenceCacheError("reference cache deployment paths must be absolute")
        pinned = Path(present[BINDING_ENV]).read_bytes()
        if sha256(pinned) != present[DIGEST_ENV]:
            raise ReferenceCacheError("reference cache index differs from pinned SHA256")
        index = CacheIndex.model_validate_json(pinned)
        index.deployment.check_runtime()
        physical_card = os.environ.get(DEVICE_ENV, os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ""))
        if not physical_card.isdigit():
            raise ReferenceCacheError("reference cache requires an unambiguous physical card")
        return cls(
            Path(present[CACHE_ENV]),
            index.deployment,
            physical_card,
            entries=index.entries,
            state_dir=Path(present[STATE_ENV]),
        )

    def binding(
        self, reference_source: str, case_idx: int, inputs, *, reference_device_policy: str
    ) -> CaseBinding:
        metadata = official_source_metadata(reference_source)
        if metadata is None:
            raise ReferenceCacheError("only official assembled tasks are reference-cache eligible")
        return CaseBinding(
            deployment_sha256=self.deployment.digest,
            reference_source_sha256=sha256(reference_source.encode("utf-8")),
            official_digest=metadata[0],
            adapter_version=metadata[1],
            case_idx=case_idx,
            input_sha256=value_sha256(inputs, input_value=True),
            device_id=self.device_id,
            reference_device_policy=reference_device_policy,
        )

    def write_case(
        self,
        reference_source: str,
        case_idx: int,
        inputs,
        outputs,
        *,
        repeated_outputs,
        reference_device_policy: str,
        selected_reference_device: str,
        latency_samples_ms: list[float],
    ) -> tuple[str, str]:
        """Audit-only writer: bind independently replayed, deterministic outputs."""
        if self.device_id not in self.deployment.device_conditions:
            raise ReferenceCacheError("cannot audit an undeclared physical card")
        binding = self.binding(
            reference_source, case_idx, inputs, reference_device_policy=reference_device_policy
        )
        blobs: dict[str, bytes] = {}
        tree = _encode(outputs, blobs)
        output_digest = sha256(canonical_bytes(tree))
        if value_sha256(repeated_outputs) != output_digest:
            raise ReferenceCacheError(
                "reference outputs are not bitwise deterministic at seed zero"
            )
        manifest = CaseManifest(
            binding=binding,
            selected_reference_device=selected_reference_device,
            outputs=tree,
            output_sha256=output_digest,
            latency_samples_ms=latency_samples_ms,
            reference_latency_median_ms=statistics.median(latency_samples_ms),
        )
        payload = canonical_bytes(manifest.model_dump(mode="json"))
        for digest, data in blobs.items():
            _atomic_immutable(self.root / "blobs" / f"{digest}.bin", data)
        _atomic_immutable(self.root / "cases" / f"{binding.key}.json", payload)
        return binding.key, sha256(payload)

    def write_index(self, path: Path, entries: dict[str, str]) -> str:
        index = CacheIndex(deployment=self.deployment, entries=entries)
        payload = canonical_bytes(index.model_dump(mode="json"))
        _atomic_immutable(Path(path), payload)
        return sha256(payload)

    def load_case(
        self, reference_source: str, case_idx: int, inputs, *, reference_device_policy: str
    ) -> CaseManifest:
        binding = self.binding(
            reference_source, case_idx, inputs, reference_device_policy=reference_device_policy
        )
        if self.entries is None or binding.key not in self.entries:
            raise ReferenceCacheMiss("official case has no pinned reference cache entry")
        path = self.root / "cases" / f"{binding.key}.json"
        if path.is_symlink() or path.parent.is_symlink():
            raise ReferenceCacheError("reference manifest must not be a symlink")
        payload = path.read_bytes()
        if sha256(payload) != self.entries[binding.key]:
            raise ReferenceCacheError("reference manifest hash mismatch")
        manifest = CaseManifest.model_validate_json(payload)
        if manifest.binding != binding:
            raise ReferenceCacheError("reference cache immutable binding mismatch")
        return manifest

    def outputs(self, manifest: CaseManifest):
        return _decode(manifest.outputs, self.root)

    @property
    def invalidation_path(self) -> Path:
        if self.state_dir is None:
            raise ReferenceCacheError("cache measurement state directory is required")
        return self.state_dir / self.deployment.digest / f"card-{self.device_id}.invalid.json"

    def needs_remeasurement(self, cached_ms: float, measured_ms: float) -> bool:
        if (
            not math.isfinite(cached_ms)
            or cached_ms <= 0
            or not math.isfinite(measured_ms)
            or measured_ms <= 0
        ):
            raise ReferenceCacheError("sentinel measurement is not finite and positive")
        if self.invalidation_path.exists():
            return True
        drift = abs(measured_ms / cached_ms - 1.0)
        if drift > 0.03 and not math.isclose(drift, 0.03, rel_tol=1e-12, abs_tol=1e-12):
            # Any corruption or write failure raises; it cannot enable denominator reuse.
            self.invalidation_path.parent.mkdir(parents=True, exist_ok=True)
            marker = canonical_bytes(
                {
                    "schema_version": "reference-cache-invalidation-v1",
                    "deployment_sha256": self.deployment.digest,
                    "device_id": self.device_id,
                    "relative_drift": drift,
                }
            )
            try:
                with self.invalidation_path.open("xb") as stream:
                    stream.write(marker)
            except FileExistsError:
                pass
            return True
        return False

    def check_measurement(self, config) -> None:
        observed = CacheMeasurement(
            method=config.measurement_method,
            warmup=config.warmup,
            repeats=config.repeats,
            cv_threshold=config.cv_threshold,
            clear_l2_cache=config.clear_l2_cache,
        )
        if observed != self.deployment.measurement:
            raise ReferenceCacheError("reference denominator measurement configuration changed")
