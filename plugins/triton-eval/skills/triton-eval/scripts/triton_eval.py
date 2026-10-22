#!/usr/bin/env python3
"""Submit operator evaluations and recover durable resources using only the standard library."""

from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import queue
import re
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

# This client's own release number, independent of the service release and
# equal to the version declared by the plugin manifests shipping it.
__version__ = "0.4.0"

TERMINAL = {"completed", "failed", "timed_out"}
STATES = TERMINAL | {"queued", "running"}
ID = re.compile(r"ev_[0-9a-f]{32}\Z")
RETRYABLE = {429, 502, 503, 504}


class ClientError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


class Parser(argparse.ArgumentParser):
    def error(self, message):
        # argparse's message can contain arbitrary argument values; never echo it.
        raise ClientError("INVALID_ARGUMENT", "invalid arguments; use --help for usage")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def positive(value: str) -> float:
    try:
        result = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("expected a positive number") from None
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("expected a finite positive number")
    return result


def case_indices(value: str) -> list[int]:
    try:
        values = [int(item) for item in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError("expected comma-separated case indices") from None
    if not values or len(values) > 128 or min(values) < 0 or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("case indices must be unique nonnegative integers")
    return sorted(values)


def arguments(argv):
    parser = Parser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for action in ("verify", "benchmark", "profile", "evaluate", "get", "progress"):
        command = commands.add_parser(action)
        command.add_argument("--url", default=os.environ.get("EVALUATION_URL"))
        command.add_argument(
            "--capability-file", type=Path, default=os.environ.get("EVALUATION_CAPABILITY_FILE")
        )
        command.add_argument("--timeout", type=positive, default=3660.0)
        command.add_argument("--poll-interval", type=positive, default=1.0)
        command.add_argument("--no-wait", action="store_true")
        command.add_argument(
            "--wait-for", choices=("compile", "correctness", "performance", "profile"),
            help="return the progress snapshot once this check has evidence",
        )
        command.add_argument("--save-handle", type=Path)
        if action in {"get", "progress"}:
            command.add_argument("evaluation_id")
        else:
            command.add_argument("--definition", type=Path, required=True)
            command.add_argument("--custom-inputs", type=Path)
            command.add_argument("--seed", type=int, default=200)
            command.add_argument("--triton-only", action="store_true")
            command.add_argument("--triton", type=Path)
            command.add_argument("--workloads", type=Path, required=True)
            command.add_argument("--idempotency-key")
            command.add_argument("--session-id")
            command.add_argument("--deadline-seconds", type=int)
        if action in {"verify", "benchmark", "profile"}:
            command.add_argument("--case-indices", type=case_indices)
        if action == "evaluate":
            command.add_argument(
                "--checks", nargs="+", choices=("compile", "correctness", "performance", "profile"),
            )
            for check_name in ("correctness", "performance", "profile"):
                command.add_argument("--" + check_name + "-cases", type=case_indices)
        if action in {"verify", "benchmark", "evaluate"}:
            command.add_argument("--stop-on", choices=("complete", "first_failure"))
            command.add_argument("--correctness-from")
        if action in {"profile", "evaluate"}:
            command.add_argument("--profile-mode", choices=("op", "deep", "trace"))
            command.add_argument("--metric-groups", nargs="+", choices=(
                "PipeUtilization", "ArithmeticUtilization", "Memory", "MemoryL0",
                "MemoryUB", "L2Cache", "ResourceConflictRatio",
            ))
            command.add_argument("--no-per-core", action="store_true")
            command.add_argument("--max-kernels-per-case", type=int)
        if action in {"benchmark", "evaluate"}:
            command.add_argument("--warmup", type=int)
            command.add_argument("--repeats", type=int)
            command.add_argument(
                "--measurement-method", "--method", choices=("event", "profiler"),
                help=("timing backend (service default: event); "
                      "profiler returns available pipe counters"),
            )
            cache = command.add_mutually_exclusive_group()
            cache.add_argument(
                "--use-baseline-cache", dest="use_baseline_cache", action="store_true",
                help="reuse Torch baseline timing for a two-source comparison (default)",
            )
            cache.add_argument(
                "--no-baseline-cache", dest="use_baseline_cache", action="store_false",
                help=("bypass comparison baseline caching; "
                      "single-source measurements are always fresh"),
            )
            command.set_defaults(use_baseline_cache=True)
    discovery = commands.add_parser("version")
    discovery.add_argument("--url", default=os.environ.get("EVALUATION_URL"))
    discovery.add_argument(
        "--capability-file", type=Path, default=os.environ.get("EVALUATION_CAPABILITY_FILE")
    )
    discovery.add_argument("--timeout", type=positive, default=30.0)
    args = parser.parse_args(argv)
    if args.command == "version":
        return args
    if args.command not in {"get", "progress"} and args.triton_only and args.triton is None:
        raise ClientError("INVALID_ARGUMENT", "--triton-only requires --triton")
    if args.wait_for and (args.no_wait or args.command == "progress"):
        raise ClientError("INVALID_ARGUMENT", "--wait-for requires a waiting evaluation or get")
    return args


def apply_controls(payload, args):
    """Forward explicit capability controls; omitted options keep service defaults."""
    if args.command == "evaluate" and args.checks is not None:
        payload["checks"] = args.checks
    if args.deadline_seconds is not None:
        if not 10 <= args.deadline_seconds <= 3600:
            raise ClientError("INVALID_ARGUMENT", "deadline must be between 10 and 3600 seconds")
        payload["deadline_seconds"] = args.deadline_seconds
    maps = {
        "correctness": {"stop_on": "stop_on", "correctness_cases": "case_indices"},
        "performance": {
            "performance_cases": "case_indices", "warmup": "warmup", "repeats": "repeats",
            "measurement_method": "method",
        },
        "profile": {
            "profile_cases": "case_indices", "profile_mode": "mode",
            "metric_groups": "metric_groups",
            "max_kernels_per_case": "max_kernels_per_case",
        },
    }
    shortcut = {"verify": "correctness", "benchmark": "performance", "profile": "profile"}
    for check_name, fields in maps.items():
        params = {
            field: getattr(args, option)
            for option, field in fields.items() if getattr(args, option, None) is not None
        }
        if shortcut.get(args.command) == check_name and args.case_indices is not None:
            params["case_indices"] = args.case_indices
        if check_name == "performance" and getattr(args, "use_baseline_cache", True) is False:
            params["use_baseline_cache"] = False
        if check_name == "profile" and getattr(args, "no_per_core", False):
            params["include_per_core"] = False
        if params:
            payload[check_name] = params
    if getattr(args, "correctness_from", None) is not None:
        if not ID.fullmatch(args.correctness_from):
            raise ClientError("INVALID_ARGUMENT", "correctness evidence ID has an invalid format")
        payload["correctness_from"] = args.correctness_from


def base_url(value):
    try:
        parsed = urllib.parse.urlsplit(value or "")
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or any(char.isspace() or ord(char) < 32 for char in value)
            or "?" in value
            or "#" in value
        ):
            raise ValueError
        _ = parsed.port  # Reject malformed or out-of-range ports.
    except (ValueError, TypeError):
        raise ClientError(
            "INVALID_URL", "URL must be HTTP(S), without credentials, query or fragment"
        ) from None
    return value.rstrip("/")


def read_text(path: Path, label: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        raise ClientError("INVALID_INPUT", f"cannot read {label} as UTF-8") from None


def json_value(data):
    try:
        return json.loads(data, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (ValueError, UnicodeError):
        raise ClientError("INVALID_JSON", "received or supplied JSON is invalid") from None


def validate_resource(value):
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("evaluation_id"), str)
        or not ID.fullmatch(value["evaluation_id"])
        or not isinstance(value.get("state"), str)
        or value["state"] not in STATES
    ):
        raise ClientError("INVALID_RESPONSE", "service returned an invalid evaluation resource")
    if value["state"] in TERMINAL and not isinstance(value.get("result"), dict):
        raise ClientError("INVALID_RESPONSE", "terminal resource lacks evaluation results")
    return value


def validate_progress(value):
    if not isinstance(value, dict) or not isinstance(value.get("completed_checks"), dict):
        raise ClientError("INVALID_RESPONSE", "progress lacks check evidence")
    for name in ("resource_revision", "progress_revision"):
        if type(value.get(name)) is not int or value[name] < 0:
            raise ClientError("INVALID_RESPONSE", "progress lacks a valid revision")
    validate_resource({**value, "result": value["completed_checks"]})
    return value


def validate_version(value):
    """Accept a version document, tolerating fields this client does not read."""
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("service_version"), str)
        or not isinstance(value.get("default_api_version"), str)
        or not isinstance(value.get("api_versions"), list)
        or not value["api_versions"]
        or any(not isinstance(item, str) for item in value["api_versions"])
    ):
        raise ClientError("INVALID_RESPONSE", "service returned an invalid version document")
    return value


def exit_status(resource):
    if resource["state"] not in TERMINAL:
        return 0  # Accepted work is not an assertion that any check passed.
    if resource["state"] != "completed":
        return 2
    result = resource["result"]
    checks = [result.get(name) for name in ("compile", "correctness", "performance", "profile")]
    present = [check for check in checks if check is not None]
    if not present or any(not isinstance(check, dict) for check in present):
        raise ClientError("INVALID_RESPONSE", "terminal resource has invalid check results")
    for check in present:
        if not isinstance(check.get("status"), str) or check["status"] not in {
            "succeeded",
            "skipped",
            "failed",
            "timeout",
        }:
            raise ClientError("INVALID_RESPONSE", "terminal resource has an invalid check status")
        if check["status"] == "succeeded" and not isinstance(check.get("data"), dict):
            raise ClientError("INVALID_RESPONSE", "successful check lacks result data")
    if any(check["status"] in {"failed", "timeout"} for check in present):
        return 2
    if not any(check["status"] == "succeeded" for check in present):
        raise ClientError("INVALID_RESPONSE", "terminal resource has no executed check evidence")
    correctness = result.get("correctness")
    if correctness and correctness["status"] == "succeeded":
        if not isinstance(correctness["data"].get("passed"), bool):
            raise ClientError("INVALID_RESPONSE", "correctness result lacks a boolean verdict")
        if correctness["data"].get("outcome") == "incomplete":
            return 2
    compile_check = result.get("compile")
    if compile_check and compile_check["status"] == "succeeded":
        if compile_check["data"].get("outcome") not in ("compiled", "compile_error"):
            raise ClientError("INVALID_RESPONSE", "compile result lacks a recognized verdict")
    for check in present:
        data = check.get("data")
        if isinstance(data, dict) and (
            data.get("passed") is False or data.get("outcome") == "compile_error"
        ):
            return 1
    return 2 if any(check["status"] == "skipped" for check in present) else 0


def redact(value, token):
    if not token:
        return value
    if isinstance(value, str):
        return value.replace(token, "[REDACTED]")
    if isinstance(value, list):
        return [redact(item, token) for item in value]
    if isinstance(value, dict):
        return {redact(key, token): redact(item, token) for key, item in value.items()}
    return value


class Client:
    def __init__(self, args):
        self.args = args
        self.url = base_url(args.url)
        self.key = getattr(args, "idempotency_key", None)
        self.evaluation_id = getattr(args, "evaluation_id", None)
        self.token = ""
        self.deadline = time.monotonic() + args.timeout
        self.opener = urllib.request.build_opener(NoRedirect())

    def remaining(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise ClientError(
                "CLIENT_TIMEOUT", "client wait expired; background work is not cancelled"
            )
        return remaining

    def pause(self, duration):
        time.sleep(min(duration, self.remaining()))
        self.remaining()

    def handle(self):
        return {
            "url": self.url,
            "evaluation_id": self.evaluation_id,
            "idempotency_key": self.key,
            "session_id": getattr(self.args, "session_id", None),
        }

    def progress(self, message):
        print(redact(message, self.token), file=sys.stderr)

    def save_handle(self):
        if self.args.save_handle is None:
            return
        temporary = None
        try:
            target = self.args.save_handle
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=target.parent, prefix=".triton-eval-", delete=False
            ) as stream:
                temporary = Path(stream.name)
                os.fchmod(stream.fileno(), 0o600)
                json.dump(redact(self.handle(), self.token), stream)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        except OSError:
            raise ClientError("HANDLE_WRITE_FAILED", "cannot save recovery handle") from None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def request(self, method, path, body=None):
        headers = {"Authorization": f"Bearer {self.token}", "Accept": "application/json"}
        if body is not None:
            headers.update({"Content-Type": "application/json", "Idempotency-Key": self.key})
            if self.args.session_id is not None:
                headers["X-Session-ID"] = self.args.session_id
        for attempt in range(3):
            try:
                return self.exchange(method, path, headers, body)
            except urllib.error.HTTPError as error:
                status = error.code
                error.close()
                if 300 <= status < 400:
                    raise ClientError(
                        "REDIRECT_REFUSED", "service redirects are not followed"
                    ) from None
                if status not in RETRYABLE or attempt == 2:
                    raise ClientError("HTTP_ERROR", f"service returned HTTP {status}") from None
            except (urllib.error.URLError, OSError, http.client.HTTPException):
                self.remaining()
                if attempt == 2:
                    raise ClientError(
                        "TRANSPORT_ERROR", "service request failed; retry using the same key"
                    ) from None
            self.progress("Retrying service request with the same recovery context.")
            self.pause(0.1 * (attempt + 1))
        raise AssertionError("unreachable")

    def exchange(self, method, path, headers, body):
        # A socket timeout alone can be reset by a slowly streaming response.
        # Bound the complete exchange too; a timed-out POST is never cancelled
        # or retried with a new key, and the daemon cannot hold the CLI open.
        completed = queue.Queue(maxsize=1)

        def network():
            try:
                request = urllib.request.Request(
                    self.url + path, data=body, headers=headers, method=method
                )
                with self.opener.open(request, timeout=min(30.0, self.remaining())) as response:
                    if response.status not in {200, 202}:
                        raise ClientError("INVALID_RESPONSE", "unexpected service response status")
                    validate = (
                        validate_version if path == "/version"
                        else validate_progress if path.endswith("/progress")
                        else validate_resource
                    )
                    value = validate(json_value(response.read()))
                completed.put((value, None))
            except Exception as error:
                completed.put((None, error))

        threading.Thread(target=network, daemon=True).start()
        try:
            value, error = completed.get(timeout=min(30.0, self.remaining()))
        except queue.Empty:
            self.remaining()
            raise TimeoutError from None
        if error is not None:
            raise error
        return value

    def run(self):
        if self.args.capability_file is None:
            raise ClientError("INVALID_INPUT", "a capability file is required")
        self.token = read_text(self.args.capability_file, "capability file").strip()
        if not self.token or any(not 33 <= ord(char) <= 126 for char in self.token):
            raise ClientError("INVALID_INPUT", "capability file must contain one ASCII token")
        if self.args.command == "version":
            # Identity discovery: the service answers even with no worker able
            # to serve, so this never waits or polls.
            return {"client_version": __version__,
                    "service": self.request("GET", "/version")}, 0
        if self.args.command in {"get", "progress"}:
            if not ID.fullmatch(self.evaluation_id):
                raise ClientError("INVALID_ARGUMENT", "evaluation ID has an invalid format")
            suffix = "/progress" if self.args.command == "progress" else ""
            resource = self.request("GET", "/evaluations/" + self.evaluation_id + suffix)
            if suffix:
                if resource["evaluation_id"] != self.evaluation_id:
                    raise ClientError("INVALID_RESPONSE", "service changed the evaluation ID")
                self.save_handle()
                return resource, 0
        else:
            definition = json_value(read_text(self.args.definition, "definition file"))
            if not isinstance(definition, dict):
                raise ClientError("INVALID_INPUT", "definition file must contain a JSON object")
            original_reference = definition.get("reference")
            if self.args.triton_only:
                definition.pop("reference", None)
            reference = definition.get("reference")
            if not reference and self.args.triton is None:
                raise ClientError("INVALID_ARGUMENT", "provide definition.reference or --triton")
            if self.args.command == "verify" and (not reference or self.args.triton is None):
                raise ClientError(
                    "INVALID_ARGUMENT", "verify requires definition.reference and --triton"
                )
            text = read_text(self.args.workloads, "workloads file").strip()
            workloads = (
                json_value(text) if text.startswith("[") else
                [json_value(line) for line in text.splitlines() if line.strip()]
            )
            if not isinstance(workloads, list) or not workloads or any(
                not isinstance(workload, dict) or not isinstance(workload.get("inputs"), dict)
                for workload in workloads
            ):
                raise ClientError("INVALID_INPUT", "workloads must be nonempty named input objects")
            payload = {"definition": definition, "workloads": workloads, "seed": self.args.seed}
            if self.args.triton is not None:
                payload["triton_code"] = read_text(self.args.triton, "triton source")
            if self.args.custom_inputs is not None:
                payload["custom_inputs_code"] = read_text(
                    self.args.custom_inputs, "custom inputs source"
                )
            uses_custom = any(
                isinstance(spec, dict) and spec.get("type") == "custom"
                for workload in workloads for spec in workload["inputs"].values()
            )
            if uses_custom:
                if not definition.get("custom_inputs_entrypoint"):
                    raise ClientError(
                        "INVALID_INPUT", "custom inputs require custom_inputs_entrypoint"
                    )
                if (self.args.triton_only and self.args.custom_inputs is None
                        and original_reference):
                    payload["custom_inputs_code"] = original_reference
                if not (reference or payload.get("custom_inputs_code")):
                    raise ClientError(
                        "INVALID_INPUT", "custom inputs require reference or --custom-inputs source"
                    )
            apply_controls(payload, self.args)
            expected_checks = {
                "verify": ["correctness"],
                "benchmark": (
                    ["correctness", "performance"] if reference and self.args.triton
                    else ["performance"]
                ),
                "profile": ["profile"],
            }.get(self.args.command, payload.get("checks") or (
                ["correctness", "performance"] if reference and self.args.triton
                else ["performance"]
            ))
            if self.args.wait_for is not None and self.args.wait_for not in expected_checks:
                raise ClientError("INVALID_ARGUMENT", "--wait-for must name a requested check")
            if self.key is None:
                self.key = uuid.uuid4().hex
            if not re.fullmatch(r"[!-~]{1,128}", self.key):
                raise ClientError("INVALID_ARGUMENT", "idempotency key has an invalid format")
            if self.args.session_id is not None and not re.fullmatch(
                r"[A-Za-z0-9._:-]{1,128}", self.args.session_id
            ):
                raise ClientError("INVALID_ARGUMENT", "session ID has an invalid format")
            self.progress("Idempotency-Key: " + self.key)
            self.save_handle()
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            endpoint = "evaluations" if self.args.command == "evaluate" else self.args.command
            resource = self.request("POST", "/" + endpoint, body)
        while True:
            if self.evaluation_id is not None and resource["evaluation_id"] != self.evaluation_id:
                raise ClientError("INVALID_RESPONSE", "service changed the evaluation ID")
            self.evaluation_id = resource["evaluation_id"]
            self.save_handle()
            self.progress("Evaluation " + self.evaluation_id + ": " + resource["state"])
            if self.args.wait_for is not None and resource["state"] not in TERMINAL:
                snapshot = self.request("GET", "/evaluations/" + self.evaluation_id + "/progress")
                if snapshot["evaluation_id"] != self.evaluation_id:
                    raise ClientError("INVALID_RESPONSE", "service changed the evaluation ID")
                evidence = snapshot["completed_checks"].get(self.args.wait_for)
                partial_performance = (
                    self.args.wait_for == "performance"
                    and isinstance(evidence, dict)
                    and evidence.get("status") == "succeeded"
                    and isinstance(evidence.get("data"), dict)
                    and evidence["data"].get("complete", True) is False
                )
                if evidence is not None and not partial_performance:
                    status = exit_status({
                        "state": "completed", "result": {self.args.wait_for: evidence},
                    })
                    return snapshot, status
            if resource["state"] in TERMINAL or self.args.no_wait:
                status = exit_status(resource)
                single_input = self.args.command != "get" and (
                    not reference or self.args.triton is None
                )
                result = resource.get("result") or {}
                unverified = any(
                    isinstance(result.get(name), dict)
                    and isinstance(result[name].get("data"), dict)
                    and result[name]["data"].get("correctness_verified") is False
                    for name in ("performance", "profile")
                )
                if single_input or unverified:
                    self.progress("Single-implementation measurement: correctness not verified.")
                return resource, status
            self.pause(self.args.poll_interval)
            resource = self.request("GET", "/evaluations/" + self.evaluation_id)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv in (["--version"], ["-V"]):
        # Offline identity: a copied or packaged CLI reports its own version
        # without a service, a capability token or the service package.
        print(json.dumps({"client_version": __version__}))
        return 0
    client = None
    try:
        client = Client(arguments(argv))
        output, status = client.run()
    except KeyboardInterrupt:
        output = {
            "error": {
                "code": "INTERRUPTED",
                "message": "interrupted; background work is not cancelled",
            }
        }
        status = 130
    except (ValueError, UnicodeError):
        output = {"error": {"code": "INVALID_INPUT", "message": "input is not valid UTF-8 JSON"}}
        status = 2
    except ClientError as error:
        output = {"error": {"code": error.code, "message": error.message}}
        status = 2
    if status in {2, 130} and "error" in output and client is not None:
        output.update(client.handle())
    if client is not None:
        output = redact(output, client.token)
    print(json.dumps(output, ensure_ascii=False, allow_nan=False))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
