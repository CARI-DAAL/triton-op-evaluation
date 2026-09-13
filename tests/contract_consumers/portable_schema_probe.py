#!/usr/bin/env python
"""Portable evaluation-api 2.1 schema consumer (framework-free).

Stdlib only plus the sole narrow direct dependency ``jsonschema`` (whose
resolver runtime dependency ``referencing`` is also permitted). It never
imports Eval Hub, FastAPI, Pydantic, Redis, or ``triton_evaluation``, and it
contains no handwritten Evaluation field/type validator: every wire decision
comes from the three frozen ``$defs`` entrypoints of the published schema.
"""
import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import jsonschema
from referencing import Registry, Resource

_FORBIDDEN_MODULES = (
    "fastapi", "pydantic", "redis", "triton_evaluation", "uvicorn", "httpx",
    "starlette", "anyio",
)


def strict_json_loads(raw: bytes):
    """Strict UTF-8 JSON: duplicate keys and non-finite constants rejected."""
    def reject_constant(token):
        raise ValueError(f"nonstandard JSON constant: {token}")

    def no_duplicates(pairs):
        seen = {}
        for key, value in pairs:
            if key in seen:
                raise ValueError(f"duplicate JSON key: {key}")
            seen[key] = value
        return seen

    return json.loads(
        raw.decode("utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=no_duplicates,
    )


def verify_contract(root: Path) -> dict:
    """Recompute both pinned digests from the local artifact copy."""
    contract = strict_json_loads((root / "contract.json").read_bytes())
    schema_digest = hashlib.sha256((root / "schema.json").read_bytes()).hexdigest()
    index_digest = hashlib.sha256(
        (root / "fixtures" / "index.json").read_bytes()
    ).hexdigest()
    if contract["evaluation_api_schema_digest"] != schema_digest:
        raise SystemExit("schema digest mismatch")
    if contract["evaluation_api_golden_fixture_digest"] != index_digest:
        raise SystemExit("fixture index digest mismatch")
    index = strict_json_loads((root / "fixtures" / "index.json").read_bytes())
    for entry in index["fixtures"]:
        path = root / entry["path"]
        if not path.is_file():
            raise SystemExit(f"missing pinned fixture: {entry['name']}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise SystemExit(f"fixture digest mismatch: {entry['name']}")
    return strict_json_loads((root / "schema.json").read_bytes())


def build_validators(schema: dict):
    """Three separate Draft 2020-12 validators over $ref wrappers."""
    root_id = schema["$id"]
    registry = Registry().with_resource(root_id, Resource.from_contents(schema))
    validators = {}
    for name in ("EvaluationRequest", "EvaluationResource", "ErrorEnvelope"):
        validators[name] = jsonschema.Draft202012Validator(
            {"$ref": f"{root_id}#/$defs/{name}"}, registry=registry
        )
    return validators


class Consumer:
    def __init__(self, base_url: str, token: str, session_id: str, key: str):
        self.base = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {token}",
            "X-Session-ID": session_id,
            "Idempotency-Key": key,
            "Content-Type": "application/json",
        }
        self.evaluation_id = None

    def _request(self, method, path, body_bytes=None):
        request = urllib.request.Request(
            f"{self.base}{path}", data=body_bytes, method=method,
            headers=self.headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                return response.status, response.read(), dict(response.headers)
        except urllib.error.HTTPError as error:
            # Non-2xx bodies still carry the ErrorEnvelope to validate.
            return error.code, error.read(), dict(error.headers)

    def submit(self, body_bytes):
        """POST through the request validator; persist only a valid 200/202."""
        validators = self.validators
        payload = strict_json_loads(body_bytes)
        validators["EvaluationRequest"].validate(payload)
        status, raw, headers = self._request("POST", "/evaluations", body_bytes)
        body = strict_json_loads(raw)
        if 200 <= status < 300:
            validators["EvaluationResource"].validate(body)
            if not self._consistent(status, body, headers):
                raise SystemExit("inconsistent resource response")
            self.evaluation_id = body["evaluation_id"]
        else:
            validators["ErrorEnvelope"].validate(body)
            if body.get("error", {}).get("code") is None:
                raise SystemExit("error envelope without a code")
        return status, body

    @staticmethod
    def _consistent(status, body, headers) -> bool:
        evaluation_id = body.get("evaluation_id", "")
        if not isinstance(evaluation_id, str) or not evaluation_id.startswith("ev_"):
            return False
        if status == 200:
            return body.get("state") in ("completed", "failed", "timed_out")
        if status == 202:
            if body.get("state") not in ("queued", "running"):
                return False
            return headers.get("Location", headers.get("location")) == (
                f"/evaluations/{evaluation_id}"
            )
        return False

    def poll(self, max_backoff=5.0, deadline_seconds=600.0):
        """GET with capped exponential backoff until terminal."""
        if self.evaluation_id is None:
            raise SystemExit("no durable ID to poll")
        backoff = 0.5
        deadline = time.monotonic() + deadline_seconds
        while True:
            status, raw, _ = self._request(
                "GET", f"/evaluations/{self.evaluation_id}"
            )
            body = strict_json_loads(raw)
            if 200 <= status < 300:
                self.validators["EvaluationResource"].validate(body)
                if body.get("state") in ("completed", "failed", "timed_out"):
                    return status, body
            else:
                self.validators["ErrorEnvelope"].validate(body)
                if status == 404:
                    raise SystemExit("durable ID not visible")
            if time.monotonic() >= deadline:
                return status, body
            time.sleep(min(backoff, max_backoff))
            backoff = min(backoff * 2, max_backoff)


def audit_imports():
    """The probe must never import a service framework or service package.

    Allowed: the Python standard library plus ``jsonschema`` and its resolver
    runtime dependency ``referencing`` (with their transitive deps).
    """
    offenders = sorted(
        name for name in sys.modules
        if name.split(".")[0] in _FORBIDDEN_MODULES
    )
    return offenders


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-root", default="contracts/evaluation-api/2.1")
    parser.add_argument("--url", default="http://127.0.0.1:11907")
    parser.add_argument("--token", required=True)
    parser.add_argument("--session-id", default="portable-probe")
    parser.add_argument("--idempotency-key", default="portable-probe-1")
    parser.add_argument("--body", help="path to the request JSON to submit")
    parser.add_argument("--audit-imports", action="store_true")
    args = parser.parse_args(argv)

    schema = verify_contract(Path(args.contract_root))
    consumer = Consumer(args.url, args.token, args.session_id, args.idempotency_key)
    consumer.validators = build_validators(schema)

    if args.audit_imports:
        offenders = audit_imports()
        if offenders:
            print(f"disallowed modules: {offenders}", file=sys.stderr)
            return 1
        print("import audit clean")
        return 0

    if args.body:
        body_bytes = Path(args.body).read_bytes()
        status, body = consumer.submit(body_bytes)
        print(json.dumps({"status": status, "evaluation_id": consumer.evaluation_id}))
        if consumer.evaluation_id:
            status, terminal = consumer.poll()
            print(json.dumps({"status": status, "state": terminal.get("state")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
