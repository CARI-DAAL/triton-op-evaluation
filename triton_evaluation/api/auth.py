"""Capability authentication dependencies.

Authentication precedes body parsing: these dependencies resolve the caller
from the Authorization header before any JSON is read. Credential failures
are indistinguishable by design.
"""
from fastapi import Depends, Request

from ..security import Caller, CapabilityRegistry
from . import errors
from .dependencies import get_context

SUBMIT_SCOPES = ("trial:submit", "grade:submit")
READ_SCOPES = ("trial:read", "grade:read", "observer:read")


def authenticate(request: Request, registry: CapabilityRegistry) -> Caller:
    try:
        return registry.authenticate(request.headers.get("Authorization"))
    except Exception:
        raise errors.unauthenticated() from None


def require_submit_caller(
    request: Request, context=Depends(get_context)
) -> Caller:
    caller = authenticate(request, context.registry)
    if not any(scope in caller.scopes for scope in SUBMIT_SCOPES):
        raise errors.forbidden("caller lacks any submit scope")
    return caller


def require_read_caller(request: Request, context=Depends(get_context)) -> Caller:
    caller = authenticate(request, context.registry)
    if not any(scope in caller.scopes for scope in READ_SCOPES):
        raise errors.forbidden("caller lacks any read scope")
    return caller
