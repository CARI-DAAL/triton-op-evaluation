"""Canonical JSON serialization and digests for durable evaluation identity.

RFC 8785 bytes and its numeric domain must survive a store reload unchanged.
These runtime operations are independent of HTTP schemas and test fixtures.
"""
import hashlib
import json
import math

import rfc8785


def canonical_json_bytes(value: object) -> bytes:
    return rfc8785.dumps(value)


_SAFE_INT_MAX = 2**53 - 1


def _parse_canonical_number(literal: str) -> int | float:
    """Parse a canonical-JSON number literal with the serializer's own rule.

    ``rfc8785.dumps`` writes every double below 1e21 in plain digit form --
    including values far outside the 2^53 integer safe domain
    (1.8517161196398838e+18 serializes as 1851716119639883800). A plain
    ``json.loads`` reads that literal back as an **int**, and re-serializing
    the reloaded structure then raises IntegerDomainError -- our canonical
    output failed to round-trip through our own store.

    Literals inside the safe domain stay exact ints; larger ones become the
    nearest double (the only thing canonical serialization could have
    produced); anything past the double domain is rejected. Mirrors the
    platform-side rule in harbor.verifier.triton.json_contract.
    """
    value = int(literal)
    if -_SAFE_INT_MAX <= value <= _SAFE_INT_MAX:
        return value
    as_double = float(literal)
    if not math.isfinite(as_double):
        raise ValueError(f"number outside the double domain: {literal}")
    return as_double


def load_canonical(data: bytes | str) -> object:
    """Reload canonical JSON produced by :func:`canonical_json_bytes`.

    Persisted canonical bytes must be parsed with the serializer's number
    domain, otherwise the reload is lossy (digit-form doubles become ints)
    and the next canonical write of the same structure fails.
    """
    return json.loads(data, parse_int=_parse_canonical_number)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
