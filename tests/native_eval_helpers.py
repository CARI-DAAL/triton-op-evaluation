"""Real catalog and fenced result fixtures shared by tool and HTTP tests."""
import json
from pathlib import Path
from types import SimpleNamespace

from triton_evaluation.api.contract import EvaluationResult
from triton_evaluation.models import AlgorithmVersions, WorkerIdentity, new_worker_id
from triton_evaluation.official_task import load_task

TOKEN = "native-tool-token"
TASK_ID = "official_l3_Demo"
FIXTURES = Path(__file__).parent / "fixtures" / "kernel_eval"


def official_catalog(tmp_path):
    task_dir = tmp_path / "official" / "Demo"
    task_dir.mkdir(parents=True)
    (task_dir / "definition.json").write_text(
        json.dumps(
            {
                "name": "Demo",
                "inputs": {"x": {"dtype": "float32"}},
                "reference": "def run(x):\n    return x + 1\n",
            }
        )
    )
    (task_dir / "workload.jsonl").write_text(
        json.dumps({"inputs": {"x": {"type": "random", "dtype": "float32", "shape": [2]}}}) + "\n"
    )
    task = load_task(task_dir, problem_id=TASK_ID)
    root = tmp_path / "catalog"
    refs = root / "references"
    refs.mkdir(parents=True)
    relative = f"references/{TASK_ID}.py"
    (root / relative).write_text(task["reference_source"])
    provenance = {
        "schema_version": "1.0",
        "manifest_id": "official-release",
        "problem_id": TASK_ID,
        "reference_digest": task["reference_digest"],
        "official_digest": task["official_digest"],
        "op_name": "Demo",
        "adapter_version": task["adapter_version"],
        "cases": 1,
    }
    (root / (relative + ".PROVENANCE.json")).write_text(json.dumps(provenance))
    manifest = root / "catalog.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "entries": [
                    {
                        "manifest_id": "official-release",
                        "problem_id": TASK_ID,
                        "reference_digest": task["reference_digest"],
                        "reference_path": relative,
                    }
                ],
            }
        )
    )
    files = SimpleNamespace(root=root, manifest_path=manifest)
    return files, task


def outcome(name, settings):
    fixture = json.loads((FIXTURES / f"{name}.input.json").read_bytes())
    result = EvaluationResult.model_validate(fixture["resource"]["result"])
    # The terminal worker result belongs to this deployment/official trial.
    from conftest import make_provenance

    return result.model_copy(update={"provenance": make_provenance(settings)})



def worker_identity(settings):
    return WorkerIdentity(
        new_worker_id(), settings.evaluator_build,
        AlgorithmVersions(
            correctness=settings.effective_correctness_algorithm_version,
            decoy=settings.decoy_algorithm_version,
            benchmark=settings.benchmark_algorithm_version,
            profiler=settings.profiler_algorithm_version,
        ),
    )
