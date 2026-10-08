"""Native HTTP/HTTPS listeners share one app, instance and lifespan in one PID."""

import asyncio
import signal
from contextlib import contextmanager

import uvicorn


class _OwnedServer(uvicorn.Server):
    @contextmanager
    def capture_signals(self):
        # The owner handles one signal for both listeners. Independent
        # uvicorn signal capture would leave the other server running.
        yield


async def serve_app(app, settings, *, install_signal_handlers: bool = True) -> None:
    configs = [
        uvicorn.Config(
            app,
            host=settings.api_host,
            port=settings.api_port,
            lifespan="off",
            log_level="warning",
        )
    ]
    if settings.api_tls_port is not None:
        configs.append(
            uvicorn.Config(
                app,
                host=settings.api_tls_host,
                port=settings.api_tls_port,
                lifespan="off",
                log_level="warning",
                ssl_certfile=str(settings.api_tls_certfile),
                ssl_keyfile=str(settings.api_tls_keyfile),
            )
        )
    servers = [_OwnedServer(config) for config in configs]
    loop = asyncio.get_running_loop()

    def shutdown():
        for server in servers:
            server.should_exit = True

    installed = []
    if install_signal_handlers:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, shutdown)
            installed.append(sig)

    sockets = []
    tasks = []
    try:
        # Validate both TLS configuration and bindings before either listener
        # starts. A occupied TLS port must never leave an HTTP-only service.
        for config in configs:
            config.load()
        for config in configs:
            sockets.append(config.bind_socket())
        # Storage checks and maintenance run exactly once.
        async with app.router.lifespan_context(app):
            tasks = [
                asyncio.create_task(server.serve(sockets=[sock]))
                for server, sock in zip(servers, sockets, strict=True)
            ]
            try:
                # Python 3.10 is supported by the deployed service. Own and
                # join every task explicitly (TaskGroup requires 3.11).
                done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
            finally:
                shutdown()
                await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        shutdown()
        for sock in sockets:
            sock.close()
        for sig in installed:
            loop.remove_signal_handler(sig)
