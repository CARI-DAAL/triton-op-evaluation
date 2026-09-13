"""App context access for the durable API."""
from dataclasses import dataclass

from fastapi import Request

from ..instance import EvaluationInstance
from ..security import CapabilityRegistry


@dataclass
class AppContext:
    instance: EvaluationInstance
    registry: CapabilityRegistry


def get_context(request: Request) -> AppContext:
    """Return the process-wide context stashed on app.state by create_app."""
    return request.app.state.context


def get_instance(request: Request) -> EvaluationInstance:
    return get_context(request).instance
