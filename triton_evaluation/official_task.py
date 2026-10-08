"""Official npu-kernelbench task adapter (debridge-era task loading).

Round-2 task set = official Ascend npu-kernelbench tasks
(``definition.json`` + ``workload.jsonl`` per task directory).  This module
is the single, tested, service-side assembler that turns an official task
directory into the reference module source the evaluation engine already
consumes (``reference_code`` → ``{stem}_torch.py`` → ``Model`` /
``get_input_groups`` / ``get_init_inputs``).

Design contract (freeze-recorded):

- Official files are the single source of truth; the composed module is a
  deterministic, replayable build artifact: ``compose_reference_source`` is
  a pure function — identical inputs produce byte-identical output.
- The official reference implementation is embedded VERBATIM (no parsing,
  no transformation).  Only a fixed, versioned loader is appended.
- Input materialization policy (pinned, mirrors the probe that validated
  20/20 official tasks green on CANN 9.1.1 / torch_npu 2.9.0.post8):

  * tensor, integer dtype, ``range`` present  → ``torch.randint(lo, hi + 1)``
  * tensor, integer dtype, no ``range``       → ``torch.randint(-5, 5)``
  * tensor, float dtype, ``range`` present    → uniform in range, cast to dtype
  * tensor, float dtype, no ``range``         → ``torch.randn(shape, dtype)``
  * tensor, bool                              → ``torch.rand(shape) > 0.5``
  * scalar                                    → value passed through verbatim
  * parameter absent from a case              → not passed (the official
    ``run()`` applies its own Python defaults)

- Cases are embedded as a pure-data JSON literal (audit static-parse
  contract).  Device and seeding are ambient (engine invariant: NPU default
  device, seed 0 per case materialization).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

OFFICIAL_ADAPTER_VERSION = "official-task-v1"

_ALLOWED_DTYPES = {
    "int8", "int16", "int32", "int64", "uint8", "bool",
    "float16", "float32", "float64", "bfloat16",
}

_LOADER_TEMPLATE = '''

import json as _json

_ADAPTER_VERSION = __ADAPTER_VERSION__

class Model(torch.nn.Module):
    """Adapter model: forwards present-case inputs to the official run()."""

    def __init__(self):
        super().__init__()

    def forward(self, *args):
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
        if rng:
            return torch.randint(rng[0], rng[1] + 1, shape, dtype=td)
        return torch.randint(-5, 5, shape, dtype=td)
    if rng:
        return (torch.rand(shape) * (rng[1] - rng[0]) + rng[0]).to(td)
    return torch.randn(shape, dtype=td)

def get_input_groups():
    cases = _json.loads(
        __CASES_LITERAL__
    )
    input_groups = []
    for case in cases:
        by_name = {i["name"]: i for i in case["inputs"]}
        vals = []
        for name in _PARAM_ORDER:
            spec = by_name.get(name)
            vals.append(_materialize(spec) if spec is not None else _ABSENT)
        input_groups.append(vals)
    return input_groups

def get_init_inputs():
    return []
'''


class OfficialTaskError(ValueError):
    """Raised when an official task directory fails validation."""


def combined_digest(definition_bytes: bytes, workload_bytes: bytes) -> str:
    """Digest binding both official files (immutability anchor for catalogs)."""
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
    for line_no, line in enumerate(workload_bytes.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            workload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise OfficialTaskError(f"{task_name}: workload line {line_no}: {exc}") from exc
        inputs: list[dict[str, Any]] = []
        for name, spec in workload.get("inputs", {}).items():
            if name not in param_order:
                raise OfficialTaskError(
                    f"{task_name}: workload input {name!r} not declared by definition.inputs"
                )
            if spec.get("type") == "random" or "shape" in spec:
                dtype = spec.get("dtype")
                if dtype not in _ALLOWED_DTYPES:
                    raise OfficialTaskError(f"{task_name}: unsupported dtype {dtype!r}")
                entry: dict[str, Any] = {
                    "name": name, "type": "tensor", "dtype": dtype,
                    "shape": spec["shape"],
                }
                if spec.get("range") is not None:
                    entry["range"] = spec["range"]
                inputs.append(entry)
            else:
                inputs.append({"name": name, "type": "attr", "value": spec.get("value")})
        cases.append({"inputs": inputs})
    if not cases:
        raise OfficialTaskError(f"{task_name}: workload has no cases")
    return cases


def compose_reference_source(definition_bytes: bytes, workload_bytes: bytes, *,
                             problem_id: str) -> str:
    """Assemble the engine-consumable reference module source.

    Pure and deterministic: byte-identical output for identical inputs.
    """
    try:
        definition = json.loads(definition_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OfficialTaskError(f"{problem_id}: definition.json: {exc}") from exc
    reference_code = definition.get("reference")
    if not isinstance(reference_code, str) or "def run(" not in reference_code:
        raise OfficialTaskError(f"{problem_id}: definition.reference must define run()")
    inputs_decl = definition.get("inputs")
    if not isinstance(inputs_decl, dict) or not inputs_decl:
        raise OfficialTaskError(f"{problem_id}: definition.inputs missing")
    param_order = list(inputs_decl.keys())

    cases = _parse_workload(workload_bytes, problem_id, param_order)
    cases_literal = json.dumps(json.dumps(cases, ensure_ascii=False), ensure_ascii=False)

    header = (
        f'"""Composed by triton_evaluation.official_task {OFFICIAL_ADAPTER_VERSION}\n'
        f"problem_id: {problem_id}\n"
        f"official_digest: {combined_digest(definition_bytes, workload_bytes)}\n"
        'The official reference implementation below is embedded VERBATIM from\n'
        'definition.json; the appended loader is fixed, versioned adapter code.\n'
        '"""\n'
        "import torch\n"
    )
    return (
        header
        + reference_code.rstrip()
        + "\n"
        + f"_PARAM_ORDER = {param_order!r}\n"
        + _LOADER_TEMPLATE
            .replace("__ADAPTER_VERSION__", repr(OFFICIAL_ADAPTER_VERSION))
            .replace("__INT_DTYPES__", repr({"int8", "int16", "int32", "int64", "uint8"}))
            .replace("__CASES_LITERAL__", cases_literal)
    )


def load_task(task_dir: Path | str, *, problem_id: str | None = None) -> dict:
    """Load one official task directory → registry entry payload."""
    task_dir = Path(task_dir)
    definition_bytes = (task_dir / "definition.json").read_bytes()
    workload_bytes = (task_dir / "workload.jsonl").read_bytes()
    problem_id = problem_id or task_dir.name
    source = compose_reference_source(definition_bytes, workload_bytes, problem_id=problem_id)
    definition = json.loads(definition_bytes.decode("utf-8"))
    workload_cases = sum(
        1 for line in workload_bytes.decode("utf-8").splitlines() if line.strip()
    )
    return {
        "problem_id": problem_id,
        "official_digest": combined_digest(definition_bytes, workload_bytes),
        "reference_source": source,
        "reference_digest": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "cases": workload_cases,
        "op_name": definition.get("name", task_dir.name),
        "source_dir": str(task_dir),
        "adapter_version": OFFICIAL_ADAPTER_VERSION,
    }
