"""Real catalog and fenced result fixtures shared by tool and HTTP tests."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from triton_evaluation.api.contract import EvaluationResult
from triton_evaluation.service.models import AlgorithmVersions, WorkerIdentity, new_worker_id

TOKEN = "native-tool-token"
TASK_ID = "custom-add/v1"
FIXTURES = Path(__file__).parent / "fixtures" / "kernel_eval"


def task_catalog(tmp_path):
    """Publish plain reference source without a dataset adapter or sidecar."""
    source = (
        "import torch\n\n"
        "class Model(torch.nn.Module):\n"
        "    def forward(self, x):\n"
        "        return x + 1\n\n"
        "def get_inputs():\n"
        "    return [torch.ones(2)]\n"
    )
    task = {
        "reference_source": source,
        "reference_digest": hashlib.sha256(source.encode("utf-8")).hexdigest(),
    }
    root = tmp_path / "catalog"
    refs = root / "references"
    refs.mkdir(parents=True)
    relative = "references/custom_add.py"
    task["reference_path"] = relative
    (root / relative).write_text(source, encoding="utf-8")
    manifest = root / "catalog.json"
    manifest.write_text(
        json.dumps({
            "schema_version": "1.0",
            "entries": [{
                "manifest_id": "custom-release",
                "problem_id": TASK_ID,
                "reference_digest": task["reference_digest"],
                "reference_path": relative,
            }],
        }),
        encoding="utf-8",
    )
    files = SimpleNamespace(root=root, manifest_path=manifest)
    return files, task


def outcome(name, settings):
    fixture = json.loads((FIXTURES / f"{name}.input.json").read_bytes())
    result = EvaluationResult.model_validate(fixture["resource"]["result"])
    # The terminal worker result belongs to this deployment/trial.
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
