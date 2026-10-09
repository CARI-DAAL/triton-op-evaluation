"""The one real NPU workload keeps the official task bytes and all boundary cases."""

import ast
import hashlib
import json
from collections import Counter
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "npu_kernelbench" / "batchmatmul"
OFFICIAL_DIGEST = "57e902365bd9b2fa437cc500cf0d862989b5881f439b2157de7341b127126c73"


def test_task_and_cases_preserve_official_and_archived_identity():
    definition_bytes = (FIXTURE / "definition.json").read_bytes()
    workload_bytes = (FIXTURE / "workload.jsonl").read_bytes()
    manifest = json.loads((FIXTURE / "manifest.json").read_text())
    assert hashlib.sha256(definition_bytes + b"\0" + workload_bytes).hexdigest() == OFFICIAL_DIGEST
    assert manifest["official_digest"]["value"] == OFFICIAL_DIGEST
    assert manifest["official_source"]["commit"] == "0607a29d231cde384c9dfdb8a39f54f91837c327"
    for name, expected in manifest["files"].items():
        data = (FIXTURE / name).read_bytes()
        assert len(data) == expected["size_bytes"], name
        assert hashlib.sha256(data).hexdigest() == expected["sha256"], name
    definition = json.loads(definition_bytes)
    assert definition["name"] == "BatchMatmul"
    assert list(definition["inputs"]) == ["A", "B"]
    assert definition["reference"] == "import torch\n\ndef run(A, B):\n    return torch.bmm(A, B)\n"


def test_all_51_original_cases_retain_order_dtypes_and_extremes():
    cases = [json.loads(line) for line in (FIXTURE / "workload.jsonl").read_text().splitlines()]
    assert [case["uuid"] for case in cases] == [f"BatchMatmul-{i:03}" for i in range(1, 52)]
    counts = Counter()
    for case in cases:
        a, b = case["inputs"]["A"], case["inputs"]["B"]
        assert a["type"] == b["type"] == "random"
        assert a["dtype"] == b["dtype"]
        assert a["shape"][0] == b["shape"][0]
        assert a["shape"][2] == b["shape"][1]
        counts[a["dtype"]] += 1
    assert counts == {"float32": 18, "float16": 18, "bfloat16": 15}
    assert cases[5]["inputs"]["A"]["shape"] == [1, 2, 3]
    assert cases[2]["inputs"]["A"]["shape"] == [2, 7, 15]
    assert cases[43]["inputs"]["A"]["shape"] == [128, 1024, 2048]
    assert cases[50]["inputs"]["A"]["shape"] == [40, 2590, 5181]


def test_candidate_is_the_archived_implementation_not_a_new_reference():
    source = (FIXTURE / "candidate.py").read_bytes()
    assert hashlib.sha256(source).hexdigest() == (
        "ade816bea5998a02e73db672aed90be085470cf41a24ee2a1a05174961bf731c"
    )
    module = ast.parse(source)
    names = {node.name for node in module.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    assert {"ModelNew", "_bmm", "_bmm_kernel"} <= names
    manifest = json.loads((FIXTURE / "manifest.json").read_text())
    assert manifest["generation_policy"]["equivalent"] is False
    assert manifest["generation_policy"]["official_cli_default_seed"] == 200
