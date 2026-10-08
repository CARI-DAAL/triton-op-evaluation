# 0002: Durable admission before execution, one owner for terminal evidence

Status: accepted for the v2.1 cutover.

An evaluation is a durable request for evidence. Accepting that request and
executing its candidate are different operations. Admission must establish
identity, idempotency, quotas and deadline before candidate code can run.
Every candidate execution belongs to a fenced attempt, including syntax and
import prechecks. A rejected or replayed submission must not run candidate code.

Previously the HTTP adapter ran a precheck before durable admission. Concurrent
requests with the same key could import a candidate twice, rejected requests
could already have executed it, and the resource deadline did not cover that
work. Compile-only requests also bypassed the worker lifecycle through a second
terminal publication path. Moving functions without changing these ownership
rules would preserve those defects.

The EvaluationInstance module owns submit/recover operations, admission policy,
dependency readiness. HTTP is an adapter for decoding,
authentication and transport responses. The durable store owns the state machine;
workers own attempts; one evidence module completes results for normal execution,
protocol failures, deadlines and recovery. Pinned identity belongs to the domain,
so recovery does not depend on an evaluator implementation.

Compile-only requests use CPU workers. Requests containing any device check use
one NPU worker for their entire attempt, including an explicit compile check.
Worker compatibility includes execution class, evaluator build and algorithm
versions. Both classes retain interactive-before-bulk queue priority. CPU workers
never acquire card ownership or participate in card health recovery.

We considered a CPU stage followed by an NPU stage for every mixed request. That
would need durable stage handoff, evidence transfer, separate failure ownership
and more retry states. No measured capacity need justifies that machinery here.
The chosen design spends some NPU worker occupancy on explicit mixed prechecks
and keeps one attempt responsible for all of its evidence. The execution class
is a scheduling decision, not an additional container security boundary.

Only explicit compile requests parse and import a candidate. A precheck never
calls an input factory and cannot fabricate correctness evidence or case counts.
Correctness always executes the verifier. Verdict reuse remains scoped to
syntax/import evidence under ADR 0001; it is not backend JIT or result caching.

Completion retains valid requested outcomes and pinned grade identity. The first
missing check receives the ending failure; later missing checks are skipped.
Missing evidence on a normal return is an invalid engine result. Infrastructure
failure cannot produce grade eligibility. Eligible grades require a complete
execution environment and scoring evidence, including legitimate measured
candidate failures. Replacement attempts begin with fresh evidence.

The public v2.1 schema is unchanged. Compile-only POST now follows the asynchronous
resource lifecycle and requires compatible CPU capacity. An explicit HTTP wait
may still return a completed resource. Readiness reports both execution classes;
submission checks only the class required by that evaluation.
