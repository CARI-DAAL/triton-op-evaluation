# Official candidate input defaults

Use this contract when grading official tasks with omitted workload parameters.
The assembler preserves an adapter-owned `_ABSENT` object in raw input groups;
its reference `Model.forward` removes those positions before calling the
original official `run(**kwargs)`. Sending that object to a candidate bypassed
Python defaults and could turn a correct implementation into a runtime error.

The engine now derives candidate inputs from the pinned reference class and the
signature of its official `run`. It replaces only that reference module's own
sentinel with the declared default. Explicit `None`, `False`, zero and strings
retain their workload values. A missing required parameter fails as a reference
contract error. Candidate declarations do not supply the official defaults.
Nonofficial tasks preserve their existing invocation behavior.

Correctness, performance and profile use the same pure preparation function.
Correctness and performance carry the reference class through cache hits without
instantiating or executing the reference. Keyword-only candidate arguments still
follow the existing signature-aware invocation contract.

The raw groups and raw cache input hash retain the sentinel. Correctness private
traces separately bind the derived candidate input hash, the raw hash, the
`official-run-defaults-v1` semantics and the names of defaulted parameters.
They disclose no tensor values. Benchmark/profile use this same frozen function;
their individual candidate hashes are not retained in the correctness trace.

This correction changes engine file hashes and requires a new sealed cache
binding and experiment source pin. It does not change the official data,
assembler-generated reference bytes, catalog digests or golden reference output
semantics. Old cache deployments cannot silently pass the changed engine hash.

The [omission inventory](evidence/official-absent-defaults.json)
records nine B1 tasks and 87 omitted parameters: all defaults are immutable
literals. The [pre-fix reproducer](evidence/absent-candidate-default-reproducer.json)
shows the ordinary `scale=2` candidate failing on the raw sentinel while the
reference returns 6. Controlled tests exercise a composed official reference,
real CPU candidate calls, cache hit/miss paths, explicit values, keyword-only and
non-None defaults, retained traces, and a separate profile child. Native NPU
acceptance remains a separate gate.
