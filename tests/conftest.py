import json
from types import SimpleNamespace

import fakeredis
import pytest

from triton_evaluation.api.contract_artifacts import sha256_hex
from triton_evaluation.broker import Broker
from triton_evaluation.config import Settings


@pytest.fixture
def settings(tmp_path):
    return Settings(engine_dir=str(tmp_path), dequeue_block_seconds=1)


@pytest.fixture
def broker(settings):
    client = fakeredis.FakeStrictRedis(decode_responses=True)
    return Broker(client, settings)


CATALOG_REFERENCE_CODE = "import torch\n\n\nclass Model(torch.nn.Module):\n    pass\n"


@pytest.fixture
def release_catalog_files(tmp_path):
    """One valid release catalog root: the manifest plus the addmm-1 reference artifact."""
    root = tmp_path / "catalog"
    references = root / "references"
    references.mkdir(parents=True)
    (references / "addmm-1.py").write_text(CATALOG_REFERENCE_CODE, encoding="utf-8")
    reference_digest = sha256_hex(CATALOG_REFERENCE_CODE.encode("utf-8"))
    manifest_path = root / "catalog.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "entries": [
                    {
                        "manifest_id": "release-1",
                        "problem_id": "addmm-1",
                        "reference_digest": reference_digest,
                        "reference_path": "references/addmm-1.py",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return SimpleNamespace(
        root=root,
        manifest_path=manifest_path,
        reference_code=CATALOG_REFERENCE_CODE,
        reference_digest=reference_digest,
    )
