"""Request-ID and body-limit middleware.

The request ID is accepted inbound only when it matches
``[A-Za-z0-9._:-]{1,64}``; otherwise ``rq_`` plus 32 UUID4 hex characters is
generated. Actual request bytes are counted from the stream (never trusting
``Content-Length``) and rejected above 4 MiB before the app parses anything,
with a schema 2.1 ``ErrorEnvelope`` body.
"""
import re
import uuid

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .contract import ErrorEnvelope, ErrorInfo

REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
MAX_BODY_BYTES = 4 * 1024 * 1024


def new_request_id() -> str:
    return "rq_" + uuid.uuid4().hex


def resolve_request_id(headers: Headers) -> str:
    inbound = headers.get("X-Request-ID")
    if inbound is not None and REQUEST_ID_RE.fullmatch(inbound) is not None:
        return inbound
    return new_request_id()


def _limit_exceeded_response(request_id: str) -> Response:
    envelope = ErrorEnvelope(
        schema_version="2.1",
        request_id=request_id,
        error=ErrorInfo(
            code="LIMIT_EXCEEDED",
            phase="request",
            check=None,
            retryable=False,
            message="request body exceeds the configured size limit",
        ),
    )
    return Response(
        content=envelope.model_dump_json(exclude_none=False),
        status_code=413,
        media_type="application/json",
        headers={"X-Request-ID": request_id},
    )


class RequestIdBodyLimitMiddleware:
    """Resolve the request ID, cap the request body, and echo the ID."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = resolve_request_id(Headers(scope=scope))
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message).append("X-Request-ID", request_id)
            await send(message)

        # Read the body once, counting actual bytes; stop the moment the cap
        # is exceeded rather than buffering an unbounded stream.
        buffered = bytearray()
        exceeded = False
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            buffered.extend(message.get("body", b""))
            if len(buffered) > MAX_BODY_BYTES:
                exceeded = True
                break
            if not message.get("more_body", False):
                break

        if exceeded:
            await _limit_exceeded_response(request_id)(scope, receive, send)
            return

        body = bytes(buffered)
        replayed = False

        async def replay_receive() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay_receive, send_with_id)
