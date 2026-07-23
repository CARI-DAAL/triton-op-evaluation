from contextlib import asynccontextmanager

from fastapi import FastAPI

from ..broker import Broker
from .routes import router


def create_app(broker: Broker) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Fail fast if Redis is unreachable at startup; release the pool on shutdown.
        broker.ping()
        yield
        broker.close()

    app = FastAPI(title="triton-evaluation", lifespan=lifespan)
    app.state.broker = broker
    app.include_router(router)
    return app
