"""Shared HTTP values used directly by FastAPI parameters and responses."""

from typing import Annotated, Literal

from pydantic import Field

from .contract import StrictModel

SessionId = Annotated[str, Field(pattern=r"^[A-Za-z0-9._:-]{1,128}$")]
IdempotencyKey = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[!-~]+$")]


class Liveness(StrictModel):
    status: Literal["ok"]


class Readiness(StrictModel):
    ready: bool
