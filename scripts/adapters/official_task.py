"""Pure, replayable assembler for official npu-kernelbench task files.

``definition.json`` and ``workload.jsonl`` remain the source of truth. The
reference string is embedded verbatim; a fixed loader exposes the engine's
``Model``, ``get_input_groups`` and ``get_init_inputs`` interface. No NPU stack
is imported by this module.

Materialization uses the ambient device and RNG (the engine supplies NPU and
seed 0). Signed integers without a range use [-5, 5), uint8 uses [0, 5),
explicit integer ranges include both endpoints, floats use randn or a uniform
range, and bool uses rand > 0.5. Scalars, including custom/None inputs, pass
through unchanged. Missing parameters use a sentinel so run() applies its own
defaults; an explicit None remains a supplied argument.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

OFFICIAL_ADAPTER_VERSION = "official-task-v1"

_ALLOWED_DTYPES = {
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "bool",
    "float16",
    "float32",
    "float64",
    "bfloat16",
}
# Stable literal order is part of the artifact, independent of PYTHONHASHSEED.
_INT_DTYPES = ("int8", "int16", "int32", "int64", "uint8")

_LOADER_TEMPLATE = '''

import json as _json

_ADAPTER_VERSION = __ADAPTER_VERSION__

class Model(torch.nn.Module):
    """Adapter model: forwards present-case inputs to the official run()."""

    def __init__(self):
        super().__init__()

    def forward(self, *args):
        if len(args) != len(_PARAM_ORDER):
            raise ValueError("official adapter input count does not match parameter order")
        kwargs = {}
        for name, val in zip(_PARAM_ORDER, args):
            if val is not _ABSENT:
                kwargs[name] = val
        return run(**kwargs)

_ABSENT = object()
_INT_DTYPES = __INT_DTYPES__

def _materialize(spec):
    if spec["type"] != "tensor":
        return spec["value"]
    td = getattr(torch, spec["dtype"])
    shape = spec["shape"]
    rng = spec.get("range")
    if spec["dtype"] == "bool":
        return torch.rand(shape) > 0.5
    if spec["dtype"] in _INT_DTYPES:
        if rng is not None:
            return torch.randint(rng[0], rng[1] + 1, shape, dtype=td)
        lo = 0 if spec["dtype"] == "uint8" else -5
        return torch.randint(lo, 5, shape, dtype=td)
    if rng is not None:
        return (torch.rand(shape) * (rng[1] - rng[0]) + rng[0]).to(td)
    return torch.randn(shape, dtype=td)

def _get_case_specs():
    cases = _json.loads(
        __CASES_LITERAL__
    )
    return cases

def get_case_count():
    return len(_get_case_specs())

def _iter_input_groups():
    for case in _get_case_specs():
        by_name = {i["name"]: i for i in case["inputs"]}
        vals = []
        for name in _PARAM_ORDER:
            spec = by_name.get(name)
            vals.append(_materialize(spec) if spec is not None else _ABSENT)
        yield vals

def get_input_groups():
    return list(_iter_input_groups())

def get_input_group(case_index):
    """Zero-based case; preserve preceding-case RNG consumption, bounded memory."""
    if not isinstance(case_index, int) or isinstance(case_index, bool) or case_index < 0:
        raise IndexError("official adapter case index must be a nonnegative integer")
    for index, inputs in enumerate(_iter_input_groups()):
        if index == case_index:
            return inputs
    raise IndexError("official adapter case index is out of range")

def get_init_inputs():
    return []
'''


class OfficialTaskError(ValueError):
    """Raised when an official task directory fails structural validation."""


def combined_digest(definition_bytes: bytes, workload_bytes: bytes) -> str:
    """Bind exact official bytes, with a separator between the two files."""
    hasher = hashlib.sha256()
    hasher.update(definition_bytes)
    hasher.update(b"\0")
    hasher.update(workload_bytes)
    return hasher.hexdigest()


def is_official_task_dir(task_dir: Path | str) -> bool:
    task_dir = Path(task_dir)
    return (task_dir / "definition.json").is_file() and (task_dir / "workload.jsonl").is_file()


def _parse_workload(workload_bytes: bytes, task_name: str, param_order: list[str]) -> list[dict]:
    cases: list[dict] = []
    try:
        lines = workload_bytes.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise OfficialTaskError(f"{task_name}: workload.jsonl is not UTF-8") from exc
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            workload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise OfficialTaskError(f"{task_name}: workload line {line_no}: {exc}") from exc
        if not isinstance(workload, dict) or not isinstance(workload.get("inputs"), dict):
            raise OfficialTaskError(
                f"{task_name}: workload line {line_no}: inputs must be an object"
            )
        inputs: list[dict[str, Any]] = []
        for name, spec in workload["inputs"].items():
            if name not in param_order:
                raise OfficialTaskError(
                    f"{task_name}: workload input {name!r} not declared by definition.inputs"
                )
            if not isinstance(spec, dict):
                raise OfficialTaskError(f"{task_name}: workload input {name!r} must be an object")
            if spec.get("type") == "random" or "shape" in spec:
                dtype = spec.get("dtype")
                if dtype not in _ALLOWED_DTYPES:
                    raise OfficialTaskError(f"{task_name}: unsupported dtype {dtype!r}")
                shape = spec.get("shape")
                if not isinstance(shape, list) or any(
                    not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in shape
                ):
                    raise OfficialTaskError(
                        f"{task_name}: tensor shape must be nonnegative integers"
                    )
                entry: dict[str, Any] = {
                    "name": name,
                    "type": "tensor",
                    "dtype": dtype,
                    "shape": shape,
                }
                if spec.get("range") is not None:
                    rng = spec["range"]
                    if (
                        not isinstance(rng, list)
                        or len(rng) != 2
                        or any(not isinstance(v, (int, float)) or isinstance(v, bool) for v in rng)
                        or rng[0] > rng[1]
                    ):
                        raise OfficialTaskError(f"{task_name}: tensor range must be [low, high]")
                    if dtype in _INT_DTYPES and any(not isinstance(v, int) for v in rng):
                        raise OfficialTaskError(
                            f"{task_name}: integer tensor range must be integers"
                        )
                    entry["range"] = rng
                inputs.append(entry)
            else:
                inputs.append({"name": name, "type": "attr", "value": spec.get("value")})
        cases.append({"inputs": inputs})
    if not cases:
        raise OfficialTaskError(f"{task_name}: workload has no cases")
    return cases


def compose_reference_source(
    definition_bytes: bytes, workload_bytes: bytes, *, problem_id: str
) -> str:
    """Assemble byte-identical source from the same bytes and problem ID."""
    if not problem_id or not problem_id.isidentifier() or len(problem_id) > 128:
        raise OfficialTaskError("problem_id must be a Python identifier of at most 128 characters")
    try:
        definition = json.loads(definition_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OfficialTaskError(f"{problem_id}: definition.json: {exc}") from exc
    if not isinstance(definition, dict):
        raise OfficialTaskError(f"{problem_id}: definition.json must be an object")
    reference_code = definition.get("reference")
    if not isinstance(reference_code, str) or "def run(" not in reference_code:
        raise OfficialTaskError(f"{problem_id}: definition.reference must define run()")
    inputs_decl = definition.get("inputs")
    if not isinstance(inputs_decl, dict) or not inputs_decl:
        raise OfficialTaskError(f"{problem_id}: definition.inputs missing")
    param_order = list(inputs_decl)

    cases = _parse_workload(workload_bytes, problem_id, param_order)
    cases_literal = json.dumps(json.dumps(cases, ensure_ascii=False), ensure_ascii=False)

    # Published artifacts retain this historical module name as a byte-stable
    # format marker. Moving the offline adapter must not rebind catalog hashes.
    header = (
        f'"""Composed by triton_evaluation.official_task {OFFICIAL_ADAPTER_VERSION}\n'
        f"problem_id: {problem_id}\n"
        f"official_digest: {combined_digest(definition_bytes, workload_bytes)}\n"
        "The official reference implementation below is embedded VERBATIM from\n"
        "definition.json; the appended loader is fixed, versioned adapter code.\n"
        '"""\n'
        "import torch\n"
    )
    return (
        header
        + reference_code
        + "\n"
        + f"_PARAM_ORDER = {param_order!r}\n"
        + _LOADER_TEMPLATE.replace("__ADAPTER_VERSION__", repr(OFFICIAL_ADAPTER_VERSION))
        .replace("__INT_DTYPES__", repr(_INT_DTYPES))
        .replace("__CASES_LITERAL__", cases_literal)
    )


def load_task(task_dir: Path | str, *, problem_id: str | None = None) -> dict:
    """Read one official directory; return its artifact and provenance facts."""
    task_dir = Path(task_dir)
    definition_bytes = (task_dir / "definition.json").read_bytes()
    workload_bytes = (task_dir / "workload.jsonl").read_bytes()
    problem_id = problem_id or task_dir.name
    source = compose_reference_source(definition_bytes, workload_bytes, problem_id=problem_id)
    definition = json.loads(definition_bytes.decode("utf-8"))
    return {
        "problem_id": problem_id,
        "official_digest": combined_digest(definition_bytes, workload_bytes),
        "reference_source": source,
        "reference_digest": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "definition_digest": hashlib.sha256(definition_bytes).hexdigest(),
        "workload_digest": hashlib.sha256(workload_bytes).hexdigest(),
        "cases": sum(1 for line in workload_bytes.decode("utf-8").splitlines() if line.strip()),
        "op_name": definition.get("name", task_dir.name),
        "source_dir": str(task_dir),
        "adapter_version": OFFICIAL_ADAPTER_VERSION,
    }
