"""Canonical admission for evaluation requests (design sections 3.1-3.2).

``admit()`` authorizes the caller's purpose against its scopes, binds grade
requests to the trusted release catalog, and canonicalizes the validated
model: the wire contract's model validators have already filled defaults and
ordered checks/case indices, so ``model_dump(mode="json",
exclude_none=False)`` plus RFC 8785 canonicalization yields the exact bytes
whose SHA-256 is ``request_digest``. Transport metadata (``wait_seconds``,
``X-Session-ID``, headers) never enters the canonical form, and source bytes
are preserved verbatim.
"""
from collections.abc import Sequence
from dataclasses import dataclass

from triton_evaluation.api.contract import Check, EvaluationRequest
from triton_evaluation.api.contract_artifacts import canonical_json_bytes, sha256_hex
from triton_evaluation.catalog import CatalogBinding, ReleaseCatalog
from triton_evaluation.security import Caller


class ForbiddenError(Exception):
    """Public ``FORBIDDEN`` rejection for purpose/scope mismatches."""

    code = "FORBIDDEN"
    phase = "auth"
    retryable = False

    def __init__(self) -> None:
        super().__init__("caller lacks the required scope")


@dataclass(frozen=True)
class CanonicalAdmission:
    canonical_request: bytes
    request_digest: str
    reference_digest: str
    requested_checks: Sequence[Check]
    catalog_binding: CatalogBinding | None


def admit(
    request: EvaluationRequest, caller: Caller, catalog: ReleaseCatalog
) -> CanonicalAdmission:
    """Authorize purpose, validate grade binding, and canonicalize the model.

    Trial and rollout requests require ``trial:submit`` and never touch the
    catalog; grade requests require ``grade:submit`` and a successful catalog
    binding.
    """
    binding: CatalogBinding | None = None
    if request.purpose in ("trial", "rollout"):
        if "trial:submit" not in caller.scopes:
            raise ForbiddenError()
    else:
        if "grade:submit" not in caller.scopes:
            raise ForbiddenError()
        binding = catalog.bind(request.grade_context, request.reference_code)
    dumped = request.model_dump(mode="json", exclude_none=False)
    canonical = canonical_json_bytes(dumped)
    return CanonicalAdmission(
        canonical_request=canonical,
        request_digest=sha256_hex(canonical),
        reference_digest=sha256_hex(request.reference_code.encode("utf-8")),
        requested_checks=tuple(request.checks),
        catalog_binding=binding,
    )
