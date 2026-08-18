"""Static bearer-capability authentication (design section 3.1).

Capabilities are provisioned out of band as one mounted secret file with the
fixed shape ``{"capabilities": [{"token", "caller_id", "caller_class",
"scopes"}]}``. There is no login or token-minting surface: this module only
parses the secret file and authenticates presented Bearer tokens. Raw token
text is hashed to SHA-256 at load time and never stored, logged, or included
in exceptions; every credential failure returns the same generic
``UNAUTHENTICATED`` error.
"""
import hashlib
import json
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError

from triton_evaluation.api.contract import StrictModel

CallerClass = Literal["trial", "trusted", "observer"]
Scope = Literal[
    "trial:submit", "trial:read", "grade:submit", "grade:read", "observer:read"
]

CALLER_ID_PATTERN = r"^[A-Za-z0-9._:-]{1,128}$"

_ALLOWED_SCOPES: dict[CallerClass, frozenset] = {
    "trial": frozenset({"trial:submit", "trial:read"}),
    "trusted": frozenset({"grade:submit", "grade:read", "observer:read"}),
    "observer": frozenset({"observer:read", "grade:submit", "grade:read"}),
}


class UnauthenticatedError(Exception):
    """Public ``UNAUTHENTICATED`` rejection, identical for every credential failure."""

    code = "UNAUTHENTICATED"
    phase = "auth"
    retryable = False

    def __init__(self) -> None:
        super().__init__("missing or invalid bearer credential")


@dataclass(frozen=True)
class Caller:
    caller_id: str
    caller_class: CallerClass
    scopes: frozenset[Scope]


class _CapabilityEntry(StrictModel):
    token: str = Field(min_length=1)
    caller_id: str = Field(pattern=CALLER_ID_PATTERN)
    caller_class: CallerClass
    scopes: list[Scope] = Field(min_length=1)


class _CapabilityFile(StrictModel):
    capabilities: list[_CapabilityEntry] = Field(min_length=1)


@dataclass(frozen=True)
class _TokenBinding:
    digest: bytes
    caller: Caller


class CapabilityRegistry:
    """Authenticated view over the mounted capability secret.

    Only SHA-256 token digests are held in memory; ``authenticate`` compares
    every configured digest with ``secrets.compare_digest`` so lookup time
    does not reveal which entry matched.
    """

    def __init__(self, bindings: tuple[_TokenBinding, ...]) -> None:
        self._bindings = bindings

    @classmethod
    def from_secret_file(cls, path: Path) -> "CapabilityRegistry":
        """Load and validate the mounted secret, rejecting insecure files.

        Load-time failures are configuration errors (``ValueError``) whose
        messages name the violated rule but never echo token text.
        """
        path = Path(path)
        try:
            file_stat = path.stat()
        except OSError:
            raise ValueError("capability secret file is unavailable") from None
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError("capability secret path is not a regular file")
        if file_stat.st_mode & 0o077:
            raise ValueError("capability secret file mode must not be broader than 0600")
        try:
            payload = json.loads(path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError("capability secret file is not valid UTF-8 JSON") from None
        try:
            parsed = _CapabilityFile.model_validate(payload)
        except ValidationError:
            raise ValueError(
                "capability secret file failed schema validation"
            ) from None
        return cls(cls._bind(parsed.capabilities))

    @staticmethod
    def _bind(entries: list[_CapabilityEntry]) -> tuple[_TokenBinding, ...]:
        callers: dict[str, Caller] = {}
        seen_digests: set[bytes] = set()
        bindings: list[_TokenBinding] = []
        for entry in entries:
            scopes = frozenset(entry.scopes)
            if len(scopes) != len(entry.scopes):
                raise ValueError("capability scopes must not contain duplicates")
            if not scopes <= _ALLOWED_SCOPES[entry.caller_class]:
                raise ValueError(
                    f"scopes are not grantable to {entry.caller_class} callers"
                )
            if entry.caller_class == "observer" and "observer:read" not in scopes:
                raise ValueError("observer callers must hold observer:read")
            digest = hashlib.sha256(entry.token.encode("utf-8")).digest()
            if digest in seen_digests:
                raise ValueError("capability tokens must be unique")
            seen_digests.add(digest)
            caller = Caller(
                caller_id=entry.caller_id,
                caller_class=entry.caller_class,
                scopes=scopes,
            )
            existing = callers.get(entry.caller_id)
            if existing is None:
                callers[entry.caller_id] = caller
            elif existing.caller_class != caller.caller_class or (
                existing.scopes != caller.scopes
            ):
                raise ValueError("conflicting definitions for one caller_id")
            else:
                # Rotation: a second token for an identical caller definition.
                caller = existing
            bindings.append(_TokenBinding(digest=digest, caller=caller))
        return tuple(bindings)

    def authenticate(self, authorization: str | None) -> Caller:
        """Resolve exactly one ``Bearer`` token to its caller.

        Every failure — missing, malformed, or unknown credential — raises the
        same generic :class:`UnauthenticatedError`.
        """
        token = self._parse_authorization(authorization)
        presented = hashlib.sha256(token.encode("utf-8")).digest()
        matched: Caller | None = None
        for binding in self._bindings:
            if secrets.compare_digest(presented, binding.digest):
                matched = binding.caller
        if matched is None:
            raise UnauthenticatedError()
        return matched

    @staticmethod
    def _parse_authorization(authorization: str | None) -> str:
        if authorization is None:
            raise UnauthenticatedError()
        parts = authorization.split(" ")
        if len(parts) != 2 or parts[0] != "Bearer" or not parts[1]:
            raise UnauthenticatedError()
        return parts[1]
