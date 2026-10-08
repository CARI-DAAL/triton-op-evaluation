"""Real local HTTP/TLS listeners share one service lifespan and shut down together."""

import asyncio
import socket
import ssl
import subprocess
from contextlib import asynccontextmanager

import httpx
import pytest
from test_kernel_eval_api import build_native_app

from triton_evaluation.api.listeners import serve_app
from triton_evaluation.config import Settings

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def native_app(tmp_path):
    return build_native_app(tmp_path)


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def certificate(tmp_path):
    cert = tmp_path / "native.crt"
    key = tmp_path / "native.key"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            "1",
            "-subj",
            "/CN=localhost",
            "-addext",
            "subjectAltName=DNS:localhost,IP:127.0.0.1",
        ],
        check=True,
        capture_output=True,
    )
    key.chmod(0o600)
    return cert, key


async def wait_for_liveness(client, url, owner):
    for _ in range(200):
        if owner.done():
            await owner
            pytest.fail("listener owner exited before accepting traffic")
        try:
            response = await client.get(url + "/livez")
            if response.status_code == 200:
                return response
        except httpx.ConnectError:
            pass
        await asyncio.sleep(0.01)
    pytest.fail("listener never accepted local traffic")


async def test_http_https_share_lifespan_and_cleanup_on_cancellation(native_app, certificate):
    cert, key = certificate
    settings = native_app.settings
    settings.api_host = "127.0.0.1"
    settings.api_port = free_port()
    settings.api_tls_port = free_port()
    settings.api_tls_certfile = cert
    settings.api_tls_keyfile = key
    app = native_app.app
    calls = []
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def counted_lifespan(instance):
        calls.append("startup")
        try:
            async with original_lifespan(instance):
                yield
        finally:
            calls.append("shutdown")

    app.router.lifespan_context = counted_lifespan
    owner = asyncio.create_task(serve_app(app, settings, install_signal_handlers=False))
    context = ssl.create_default_context(cafile=str(cert))
    try:
        async with httpx.AsyncClient(verify=context, timeout=1) as client:
            http = await wait_for_liveness(client, f"http://127.0.0.1:{settings.api_port}", owner)
            https = await wait_for_liveness(
                client, f"https://127.0.0.1:{settings.api_tls_port}", owner
            )
            assert http.json() == https.json() == {"status": "ok"}
            rejected = await client.post(
                f"https://127.0.0.1:{settings.api_tls_port}/verify", json={}
            )
            assert rejected.status_code == 401
            assert rejected.json()["error"]["code"] == "UNAUTHENTICATED"
            assert rejected.json()["request_id"] == rejected.headers["X-Request-ID"]
            assert app.state.context.instance is native_app.app.state.context.instance
            assert calls == ["startup"]
    finally:
        owner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(owner, timeout=5)
    assert calls == ["startup", "shutdown"]
    assert app.state.maintenance_stop.is_set()
    for port in (settings.api_port, settings.api_tls_port):
        with socket.socket() as sock:
            assert sock.connect_ex(("127.0.0.1", port)) != 0


async def test_tls_bind_failure_closes_reserved_http_socket(native_app, certificate):
    cert, key = certificate
    settings = native_app.settings
    settings.api_port = free_port()
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        settings.api_tls_port = occupied.getsockname()[1]
        settings.api_tls_certfile = cert
        settings.api_tls_keyfile = key
        with pytest.raises(SystemExit):
            await serve_app(native_app.app, settings, install_signal_handlers=False)
    assert not hasattr(native_app.app.state, "maintenance_stop")
    with socket.socket() as sock:
        assert sock.connect_ex(("127.0.0.1", settings.api_port)) != 0


async def test_bad_tls_key_starts_no_listener(native_app, certificate):
    cert, _key = certificate
    settings = native_app.settings
    settings.api_port = free_port()
    settings.api_tls_port = free_port()
    settings.api_tls_certfile = cert
    settings.api_tls_keyfile = cert.parent / "missing-key"
    with pytest.raises(FileNotFoundError):
        await serve_app(native_app.app, settings, install_signal_handlers=False)
    assert not hasattr(native_app.app.state, "maintenance_stop")
    with socket.socket() as sock:
        assert sock.connect_ex(("127.0.0.1", settings.api_port)) != 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"api_tls_port": 12909},
        {"api_tls_certfile": "/tmp/cert"},
        {
            "api_port": 12907,
            "api_tls_port": 12907,
            "api_tls_certfile": "/tmp/cert",
            "api_tls_keyfile": "/tmp/key",
        },
    ],
)
def test_partial_or_conflicting_tls_configuration_is_rejected(overrides):
    with pytest.raises(ValueError, match="native"):
        Settings(**overrides)
