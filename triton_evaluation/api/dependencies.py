from fastapi import Request

from ..broker import Broker


def get_broker(request: Request) -> Broker:
    """Return the process-wide Broker stashed on app.state by create_app."""
    return request.app.state.broker
