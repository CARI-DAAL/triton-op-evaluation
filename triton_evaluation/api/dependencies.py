"""App context access for the durable API."""
from dataclasses import dataclass, field

from fastapi import Request

from ..service.instance import EvaluationInstance
from ..service.security import CapabilityRegistry
from ..versioning import ReleaseIdentity, unverified_identity


@dataclass
class AppContext:
    instance: EvaluationInstance
    registry: CapabilityRegistry
    # In-process assembly that verified nothing says so; formal startup paths
    # pass the identity they verified.
    release: ReleaseIdentity = field(default_factory=unverified_identity)


def get_context(request: Request) -> AppContext:
    """Return the process-wide context stashed on app.state by create_app."""
    return request.app.state.context


def get_instance(request: Request) -> EvaluationInstance:
    return get_context(request).instance
