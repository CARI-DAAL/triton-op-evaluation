"""The standalone CLI speaks the real HTTP admission and recovery contract."""

import asyncio
import json
import shutil
import sys
from pathlib import Path

import httpx
import pytest
from test_api import TRIAL_TOKEN, build_app, seed_worker
from test_durable_http import claim_and_complete
from test_native_listeners import free_port, wait_for_liveness

from triton_evaluation.api.listeners import serve_app

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def test_isolated_client_submits_replays_and_recovers_real_resource(tmp_path):
    app = build_app(tmp_path, None)
    seed_worker(app)
    app.settings.api_host = "127.0.0.1"
    app.settings.api_port = free_port()
    base = f"http://127.0.0.1:{app.settings.api_port}"
    owner = asyncio.create_task(serve_app(app.app, app.settings, install_signal_handlers=False))
    plugin = Path(__file__).parents[1] / "plugins/triton-eval/skills/triton-eval/scripts"
    script = tmp_path / "client.py"
    shutil.copyfile(plugin / "triton_eval.py", script)
    token = tmp_path / "token"
    token.write_text(TRIAL_TOKEN)
    token.chmod(0o600)
    torch_source = tmp_path / "torch_impl.py"
    candidate = tmp_path / "triton_impl.py"
    cases = tmp_path / "cases.json"
    marker = tmp_path / "executed-during-admission"
    source = f"from pathlib import Path\nPath({str(marker)!r}).touch()\ndef run(x): return x\n"
    torch_source.write_text(json.dumps({
        "axes": {}, "inputs": {"x": {}}, "outputs": {}, "reference": source,
    }))
    candidate.write_text(source)
    cases.write_text(json.dumps([{
        "uuid": "test", "axes": None,
        "inputs": {"x": {"type": "random", "shape": [2], "dtype": "float32"}},
    }]))
    handle = tmp_path / "handle.json"

    async def invoke(*arguments):
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-I", "-S", str(script), *arguments,
            "--url", base, "--capability-file", str(token),
            cwd=tmp_path, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)
        finally:
            if process.returncode is None:
                process.kill()
                await process.communicate()
        assert process.returncode == 0, stderr.decode()
        assert TRIAL_TOKEN.encode() not in stdout + stderr
        return json.loads(stdout)

    submit = (
        "verify", "--definition", str(torch_source), "--triton", str(candidate),
        "--workloads", str(cases), "--idempotency-key", "cli-integration",
        "--session-id", "cli-session", "--save-handle", str(handle),
    )
    try:
        async with httpx.AsyncClient(timeout=1) as client:
            await wait_for_liveness(client, base, owner)
        queued = await invoke(*submit, "--no-wait")
        assert queued["state"] == "queued"
        assert not marker.exists()
        saved = json.loads(handle.read_text())
        assert saved["evaluation_id"] == queued["evaluation_id"]
        assert saved["session_id"] == "cli-session"
        claim_and_complete(app)
        completed = await invoke(*submit)
        assert completed["evaluation_id"] == queued["evaluation_id"]
        assert completed["state"] == "completed"
        assert completed["result"]["correctness"]["data"]["passed"] is True
        recovered = await invoke("get", queued["evaluation_id"])
        assert recovered == completed
        assert not marker.exists()
    finally:
        owner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(owner, timeout=5)
        await app.http.aclose()
