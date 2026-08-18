"""App context access for the durable API."""
from dataclasses import dataclass

from fastapi import Request

from ..broker import EvaluationStore
from ..catalog import ReleaseCatalog
from ..config import Settings
from ..security import CapabilityRegistry


@dataclass
class AppContext:
    settings: Settings
    store: EvaluationStore
    registry: CapabilityRegistry
    catalog: ReleaseCatalog
    metrics: object | None = None
    artifacts_verified: bool = False


def get_context(request: Request) -> AppContext:
    """Return the process-wide context stashed on app.state by create_app."""
    return request.app.state.context
