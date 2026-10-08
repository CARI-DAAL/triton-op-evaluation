"""Bearer dependencies resolve capabilities before admitting any work."""
from typing import Annotated

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..security import Caller, CapabilityRegistry
from . import errors
from .dependencies import AppContext, get_context

capability_bearer = HTTPBearer(
    scheme_name="CapabilityBearer", auto_error=False,
    description="Capability token; submit/read/observer scope depends on the operation.",
)
tool_bearer = HTTPBearer(
    scheme_name="KernelToolBearer", auto_error=False,
    description="Dedicated kernel tool token, separate from capability credentials.",
)

SUBMIT_SCOPES = ("trial:submit", "grade:submit")
READ_SCOPES = ("trial:read", "grade:read", "observer:read")


def authenticate(
    credential: HTTPAuthorizationCredentials | None, registry: CapabilityRegistry,
) -> Caller:
    try:
        header = f"Bearer {credential.credentials}" if credential else None
        return registry.authenticate(header)
    except Exception:
        raise errors.unauthenticated() from None


def require_caller(
    context: Annotated[AppContext, Depends(get_context)],
    credential: Annotated[HTTPAuthorizationCredentials | None, Depends(capability_bearer)],
) -> Caller:
    return authenticate(credential, context.registry)


def require_submit_caller(caller: Annotated[Caller, Depends(require_caller)]) -> Caller:
    if not any(scope in caller.scopes for scope in SUBMIT_SCOPES):
        raise errors.forbidden("caller lacks any submit scope")
    return caller


def require_read_caller(caller: Annotated[Caller, Depends(require_caller)]) -> Caller:
    if not any(scope in caller.scopes for scope in READ_SCOPES):
        raise errors.forbidden("caller lacks any read scope")
    return caller


def require_observer(caller: Annotated[Caller, Depends(require_caller)]) -> Caller:
    if "observer:read" not in caller.scopes:
        raise errors.forbidden("metrics require observer scope")
    return caller
