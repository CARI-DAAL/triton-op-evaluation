"""Stable error codes and the API error boundary.

Every non-2xx response is a single-line, byte-bounded, redacted
``ErrorEnvelope``. The initial code policy is frozen here: a code's phase may
be narrowed to its failing check at the resource layer, but its retryable
value never varies by call site.
"""
from collections.abc import Mapping
from dataclasses import dataclass, field

from ..api.contract import Check, ErrorInfo

RETRY_AFTER_ONE = {"Retry-After": "1"}


@dataclass(frozen=True)
class ApiError(Exception):
    http_status: int
    info: ErrorInfo
    headers: Mapping[str, str] = field(default_factory=dict)


def _error(code: str, phase: str, retryable: bool, message: str,
           check: Check | None = None) -> ErrorInfo:
    return ErrorInfo(
        code=code, phase=phase, check=check, retryable=retryable, message=message
    )


def unauthenticated() -> ApiError:
    return ApiError(
        401,
        _error("UNAUTHENTICATED", "auth", False, "missing or invalid bearer credential"),
        {"WWW-Authenticate": "Bearer"},
    )


def forbidden(message="caller lacks the required scope") -> ApiError:
    return ApiError(403, _error("FORBIDDEN", "auth", False, message))


def manifest_binding_failed(message="reference does not match the bound manifest entry"):
    return ApiError(403, _error("MANIFEST_BINDING_FAILED", "admission", False, message))


def invalid_request(message: str) -> ApiError:
    return ApiError(400, _error("INVALID_REQUEST", "request", False, message))


def invalid_request_schema(message: str) -> ApiError:
    return ApiError(422, _error("INVALID_REQUEST", "request", False, message))


def not_found() -> ApiError:
    return ApiError(404, _error("NOT_FOUND", "request", False, "resource not found"))


def idempotency_conflict() -> ApiError:
    return ApiError(
        409,
        _error(
            "IDEMPOTENCY_CONFLICT", "admission", False,
            "idempotency key was used with a different request",
        ),
    )


def limit_exceeded(message: str) -> ApiError:
    return ApiError(413, _error("LIMIT_EXCEEDED", "request", False, message))


def rate_limited() -> ApiError:
    return ApiError(
        429,
        _error("RATE_LIMITED", "admission", True, "caller rate limit exceeded"),
        RETRY_AFTER_ONE,
    )


def not_ready(message="service is not ready to accept evaluations") -> ApiError:
    return ApiError(
        503, _error("NOT_READY", "admission", True, message), RETRY_AFTER_ONE
    )


def queue_unavailable() -> ApiError:
    return ApiError(
        503,
        _error("QUEUE_UNAVAILABLE", "queue", True, "durable queue is temporarily unavailable"),
        RETRY_AFTER_ONE,
    )


def internal(message="unexpected internal error") -> ApiError:
    return ApiError(500, _error("INTERNAL", "storage", False, message))
