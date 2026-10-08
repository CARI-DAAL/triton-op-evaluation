# 0001: Evaluation instance ownership and measured results

Status: accepted for instance ownership. The original reference-cache scope below
is historical; the current opt-in policy is documented in
[reference caches](../operations/reference-cache.md).

One evaluation instance owns admission, durable IDs, leases, card health and
result recovery. Each machine runs independent Redis state. The consumer
selects an instance once per session and retains the owner in every recovery
handle. It does not turn an ambiguous submission into work on a second host.

A shared cross-host Redis would require host-qualified card identities and
cross-host recovery authority. Independent instances keep the physical-card
recovery module valid, at the cost of per-instance quotas and no automatic
reassignment of accepted evaluations. Stable rendezvous selection distributes
new sessions; it is not a load-aware or high-availability scheduler.

The wire interface is evaluations with requested checks. Session verify,
benchmark and profile remain caller intents where they own budget semantics;
they are not alternate transport protocols. A deep consumer module owns
capabilities, request identity, retry, polling and evidence validation.

Compile verdict reuse is limited to syntax/import prechecks. A positive
precheck is not successful on-device compilation. Backend failures may depend
on input shapes, dtypes or reference behavior and cannot populate a kernel-only
negative cache. Publication precedence and TTL belong to one atomic cache
operation; absent toolchain identity disables reuse.

New logical evaluations execute correctness and performance again. Sharing
successful results would erase fresh measurement evidence; sharing mutable
executable artifacts requires a separately validated integrity and toolchain
policy. This release provides neither and makes no warm-cache performance
guarantee. Idempotency replay returns the same logical evaluation and is
intentionally distinct from result caching.

Old incompatible records stay in their original namespace. Cutover requires
draining the previous deployment, collecting its results, and moving consumer
and evaluator to compatible HTTP models and behavior. Compatibility is checked
by HTTP and consumer tests; the running application publishes its OpenAPI directly.
The former generated-contract startup gate has been retired. The new release
never guesses how to reinterpret prior records.
