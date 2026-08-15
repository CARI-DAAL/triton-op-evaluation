"""Public API package.

``triton_evaluation.api.contract`` is the standalone wire-contract surface and
must stay importable without the app stack, so ``create_app`` is loaded lazily.
"""
import typing

if typing.TYPE_CHECKING:
    from .app import create_app

__all__ = ["create_app"]


def __getattr__(name: str) -> typing.Any:
    if name == "create_app":
        from .app import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
