"""Bounded Redis-backed internal metrics.

Counters and histograms live in Redis hashes so the API and worker processes
share one view. Metric names, label names, AND label values are checked
against a compile-time allowlist — arbitrary caller, session, or evaluation
values are never accepted as labels; callers aggregate to the bounded
``caller_class`` values.

Note: ``GradeRecord scored/unscored`` metrics belong to Eval Hub. This
service never creates GradeRecords; it exposes the purpose, grade
eligibility, domain outcome, and check-availability dimensions Hub needs.
"""
import logging
import re

logger = logging.getLogger(__name__)

DURATION_BUCKETS = (0.01, 0.1, 1, 5, 15, 60, 300, 900, 3600)
COVERAGE_BUCKETS = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)

_ERROR_CODES = (
    "INVALID_REQUEST", "UNAUTHENTICATED", "FORBIDDEN", "MANIFEST_BINDING_FAILED",
    "NOT_FOUND", "IDEMPOTENCY_CONFLICT", "LIMIT_EXCEEDED", "RATE_LIMITED",
    "NOT_READY", "QUEUE_UNAVAILABLE", "WORKER_LOST", "EVALUATOR_EXECUTION_FAILED",
    "INVALID_ENGINE_OUTPUT", "DEADLINE_EXCEEDED", "INTERNAL",
)

LABEL_VALUES = {
    "caller_class": ("trial", "trusted", "observer"),
    "decision": ("created", "replayed", "conflict", "rate_limited", "limit_exceeded"),
    "state": ("queued", "running", "completed", "failed", "timed_out"),
    "check": ("correctness", "performance", "profile"),
    "status": ("succeeded", "failed", "timeout", "skipped"),
    "skip_code": ("DEPENDENCY_NOT_PASSED", "PRIOR_CHECK_FAILED", "RESOURCE_DEADLINE"),
    "code": _ERROR_CODES,
    "domain_outcome": (
        "correct", "wrong", "compile_error", "runtime_error", "execution_timeout",
    ),
    "reason": ("UNSUPPORTED_CASE", "NO_STABLE_MEASUREMENT"),
    "outcome": (
        "renewed", "rejected", "won", "lost", "requeued", "failed", "bound",
        "rejected_binding", "compatible", "incompatible",
    ),
    "route": ("evaluations_post", "evaluations_get", "verify", "profile", "tasks"),
    "purpose": ("trial", "grade"),
    "eligible": ("true", "false"),
    "decoy": ("true", "false"),
}

ALLOWED_METRICS = {
    # API surface
    "evaluation_admissions_total": ("decision", "caller_class"),
    "idempotency_replays_total": ("caller_class",),
    "api_error_responses_total": ("code",),
    "auth_failures_total": ("code",),
    "legacy_route_requests_total": ("route",),
    # Resource lifecycle
    "resource_terminal_transitions_total": ("state", "caller_class"),
    "resource_terminal_duration_seconds": ("state",),
    "resource_deadline_expiries_total": ("state",),
    "resource_state_gauge": ("state",),
    # Check outcomes
    "check_outcomes_total": ("check", "status"),
    "check_skips_total": ("check", "skip_code"),
    "check_errors_total": ("check", "code"),
    "correctness_domain_outcomes_total": ("domain_outcome",),
    "decoy_detections_total": ("decoy",),
    "grade_eligibility_total": ("eligible", "purpose"),
    "performance_unmeasured_total": ("reason",),
    "performance_coverage": (),  # histogram over coverage buckets
    # Worker/lease dimensions
    "worker_heartbeats_total": (),
    "worker_attempts_total": (),
    "lease_renewals_total": ("outcome",),
    "fenced_write_rejections_total": (),
    "terminal_commit_cas_total": ("outcome",),
    "reaped_leases_total": ("outcome",),
    "worker_compatibility_total": ("outcome",),
    # Shaping/catalog
    "shaping_failures_total": ("check",),
    "catalog_binding_outcomes_total": ("outcome",),
}

_HISTOGRAMS = {
    "resource_terminal_duration_seconds": DURATION_BUCKETS,
    "performance_coverage": COVERAGE_BUCKETS,
}
_GAUGE_NAMES = ("resource_state_gauge",)

_METRIC_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _validate(name: str, labels: dict) -> None:
    allowed_labels = ALLOWED_METRICS.get(name)
    if allowed_labels is None:
        raise ValueError(f"metric name is not allowlisted: {name}")
    if set(labels) != set(allowed_labels):
        raise ValueError(f"label set does not match the allowlist for {name}")
    for label, value in labels.items():
        if value not in LABEL_VALUES[label]:
            raise ValueError(f"label value is not allowlisted: {label}={value}")


def _field(labels: dict) -> str:
    if not labels:
        return "total"
    return "|".join(f"{key}={labels[key]}" for key in sorted(labels))


class RedisMetrics:
    """Counters and histograms in Redis hashes, rendered as OpenMetrics."""

    def __init__(self, client, prefix: str):
        self._r = client
        self._prefix = prefix

    def _key(self, name: str) -> str:
        return f"{self._prefix}:metrics:data:{name}"

    def increment(self, name: str, labels, amount: int = 1) -> None:
        try:
            _validate(name, dict(labels))
        except ValueError:
            raise
        except Exception as error:  # noqa: BLE001 — defensive only
            raise ValueError(str(error)) from None
        try:
            self._r.hincrby(self._key(name), _field(dict(labels)), int(amount))
        except Exception as error:  # noqa: BLE001 — never affect a transition
            logger.warning("metrics write failed (%s)", type(error).__name__)

    def observe(self, name: str, labels, value: float) -> None:
        labels = dict(labels)
        _validate(name, labels)
        buckets = _HISTOGRAMS[name]
        field = _field(labels)
        try:
            pipeline = self._r.pipeline()
            base = self._key(name)
            pipeline.hincrby(base, f"{field}|_count", 1)
            pipeline.hincrbyfloat(base, f"{field}|_sum", float(value))
            for bucket in buckets:
                if value <= bucket:
                    pipeline.hincrby(base, f"{field}|le={bucket}", 1)
            pipeline.execute()
        except Exception as error:  # noqa: BLE001 — never affect a transition
            logger.warning("metrics write failed (%s)", type(error).__name__)

    def render_openmetrics(self) -> bytes:
        lines: list[str] = []
        for name in sorted(ALLOWED_METRICS):
            stored = self._r.hgetall(self._key(name))
            if not stored:
                continue
            if name in _HISTOGRAMS:
                lines.append(f"# TYPE {name} histogram")
                entries = {}
                for raw_field, count in stored.items():
                    text = raw_field.decode() if isinstance(raw_field, bytes) else raw_field
                    label_part, _, tail = text.rpartition("|")
                    if label_part == "total":
                        label_part = ""
                    value = count.decode() if isinstance(count, bytes) else str(count)
                    entries.setdefault(label_part, []).append((tail, value))
                for label_part in sorted(entries):
                    suffixes = dict(entries[label_part])
                    rendered = _openmetrics_labels(label_part)
                    prefix = f"{name}_bucket{rendered[:-1]}," if rendered else (
                        f"{name}_bucket{{"
                    )
                    for bucket in _HISTOGRAMS[name]:
                        key = f"le={bucket}"
                        if key in suffixes:
                            lines.append(f'{prefix}le="{bucket}"}} {suffixes[key]}')
                    lines.append(f"{name}_count{rendered} {suffixes.get('_count', 0)}")
                    lines.append(f"{name}_sum{rendered} {suffixes.get('_sum', 0)}")
            else:
                kind = "gauge" if name in _GAUGE_NAMES else "counter"
                lines.append(f"# TYPE {name} {kind}")
                for raw_field, count in sorted(stored.items()):
                    field = (
                        raw_field.decode() if isinstance(raw_field, bytes) else raw_field
                    )
                    lines.append(f"{name}{_openmetrics_labels(field)} {int(count)}")
        lines.append("# EOF")
        return ("\n".join(lines) + "\n").encode("utf-8")


def _openmetrics_labels(field: str) -> str:
    """Render a stored ``k=v|k2=v2`` field as OpenMetrics label syntax."""
    if not field or field == "total":
        return ""
    pairs = sorted(part.split("=", 1) for part in field.split("|"))
    return "{" + ",".join(f'{key}="{value}"' for key, value in pairs) + "}"
