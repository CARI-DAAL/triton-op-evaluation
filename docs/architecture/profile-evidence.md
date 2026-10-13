# Profile evidence ownership

`ProfileEvaluator` owns one engine invocation for an explicit case selection or
an all-case run. It does not repair private evidence. The engine's envelope binds
the original selected case IDs and all counts. The shaper also reads historical
collections of single-case envelopes, validating every original envelope.

`shape_profile_runs(raw, params)` owns the complete evidence decision:

1. Validate every original envelope, including timeout, remaining cases,
   status counts and private fields.
2. Bind observations to the requested mode and case selection. All-case output
   must cover the reported reference inventory; selected runs must identify the
   requested case and agree on the total inventory.
3. Project validated observations into public cases and combine their counts.
4. Bind explicitly selected deep metric groups and sampling scope to the capture.
5. Apply recommendation visibility and bounded kernel presentation, recording
   truncation as partial evidence.

Previously, merging reset `timed_out` and `remaining_case_indices` and rebuilt
status counts before validation. That could turn incomplete or contradictory
evidence into success. Hiding recommendations could similarly erase invalid
private fields before the validator saw them. Missing dictionary fields could
escape as Python exceptions rather than a classified check failure.

Invalid observations now flow through the normal shaping failure path as
`INVALID_ENGINE_OUTPUT`, with no candidate attribution or fabricated success.
Legitimate partial and failed profile cases retain explicit coverage gaps.

`trace` is a separate kernel-duration capability supplied by the existing
`torch_npu.profiler` timing collector. A combined timing/trace request shares one
candidate child process. Event timing completes with profiling off and publishes
an atomic checkpoint before trace capture; profiler timing can reuse the same
capture. Each phase keeps its own requested cases and original identities.
Trace-only requests use the same runner without a correctness requirement and
single-source traces remain correctness-unverified. These durations do not stand
in for op/deep utilization counters.

The shared runner loads source and discovers input groups once, streams cases,
and recreates input/model state for separate phases. Built-in generators declare
fresh tensor ownership; unknown/custom providers require protected replay.
Baseline cache production stays reference-only. Existing op/deep collectors still
launch native capture processes for their requested metrics.

`tests/test_profile_evidence.py` exercises the production runner, shaper and
fenced orchestrator together, replacing only the engine process adapter. It
covers contradictory completion fields, malformed envelopes, identity drift,
missing all-case evidence and normal op/deep/partial/failed outputs. CPU tests
establish evidence handling, not hardware profiling accuracy or NPU speed.
