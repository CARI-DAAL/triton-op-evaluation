# Profile evidence ownership

`ProfileEvaluator` owns sequential engine invocations and collection of their
raw results. It does not repair, merge or filter private evidence. An explicit
case selection produces a collection of untouched envelopes; an all-case run
produces the engine's single envelope.

`shape_profile_runs(raw, params)` owns the complete evidence decision:

1. Validate every original envelope, including timeout, remaining cases,
   status counts and private fields.
2. Bind observations to the requested mode and case selection. All-case output
   must cover the reported reference inventory; selected runs must identify the
   requested case and agree on the total inventory.
3. Project validated observations into public cases and combine their counts.
4. Apply the caller's recommendation-visibility preference to the public result.

Previously, merging reset `timed_out` and `remaining_case_indices` and rebuilt
status counts before validation. That could turn incomplete or contradictory
evidence into success. Hiding recommendations could similarly erase invalid
private fields before the validator saw them. Missing dictionary fields could
escape as Python exceptions rather than a classified check failure.

Invalid observations now flow through the normal shaping failure path as
`INVALID_ENGINE_OUTPUT`, with no candidate attribution or fabricated success.
Legitimate partial and failed profile cases retain their existing public
representation. Engine exit-code policies and the published schema are unchanged.

`tests/test_profile_evidence.py` exercises the production runner, shaper and
fenced orchestrator together, replacing only the engine process adapter. It
covers contradictory completion fields, malformed envelopes, identity drift,
missing all-case evidence and normal op/deep/partial/failed outputs. CPU tests
establish evidence handling, not hardware profiling accuracy or NPU speed.
