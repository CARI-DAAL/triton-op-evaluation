# Freeze a correctness reference policy

Set `TRITON_EVAL_REFERENCE_DEVICE_POLICY` identically on the API and all workers:

- `cpu-first` (default) preserves the existing CPU reference and NPU fallback
  when the reference cannot execute on CPU. Numeric mismatches and attributed
  candidate failures do not trigger fallback.
- `npu-strict` selects the worker's NPU for every reference model and input.
  Reference execution failures, candidate failures and mismatches never select
  another device or trigger another reference attempt.

`Settings.reference_device_policy` is wired through `EvaluationEvaluator` and
`VerifyEvaluator` to an explicit engine `--reference-device-policy` argument,
including its watchdog subprocess. It works for programmatic Settings and
environment launches; it does not depend on inheriting variables through the
engine sandbox. A request cannot select its own policy. Input generation, model
seeding, comparison formulas, tolerances, candidate execution and performance
measurement are unchanged.

Both admission and worker compatibility use
`Settings.effective_correctness_algorithm_version`. CPU-first keeps the configured
base version unchanged. NPU-strict appends `:npu-strict-v1`, so the default becomes
`correctness-v1:npu-strict-v1`. Keep `TRITON_EVAL_CORRECTNESS_ALGORITHM_VERSION`
as the base version; do not append the policy suffix manually. A differently
configured worker cannot claim the evaluation even if its build was misdeclared.
Deploy a new evaluator build that binds code, runtime manifest and policy, and
declare a new experiment before comparing results under a changed policy.

The public v2.1 schema does not change. Consumers that pin algorithms must accept
the new explicit correctness version in their experiment configuration. Open
Evo's formal projection defaults to the old version; pass all four
`expected_algorithms`, including the suffixed correctness version, for this
experiment. Never weaken its exact comparison to make mismatched evidence pass.

## Observe the actual reference devices

The engine writes a private `verify_reference.json` sidecar. Each case attempt
records `selected_reference_device`, whether the reference forward actually
completed, and the observed device of every top-level tensor output. A CPU-first
fallback produces a second entry for that case. A failure before reference
completion has `reference_completed=false` and no observed output devices.
An import failure may have no case entries. Scalar outputs have no tensor
device; do not infer an observed device from the selected device. Internal
operations inside a reference are not traced by this telemetry.

To retain it after attempt cleanup on a host deployment, set
`TRITON_EVAL_REFERENCE_DIAGNOSTICS_DIR` to an operator-owned directory outside
all candidate mounts. Compose retains it under the existing logs volume at
`logs/reference-diagnostics`. The worker copies validated observations to
`<directory>/<evaluation_id>/<attempt_id>.json` before deleting its attempt,
binding evaluation ID, attempt ID, request digest, evaluator build, correctness
algorithm, configured policy and physical card. It rejects symlinks, nonregular
files, oversized files, invalid fields and mismatched policy. Retention is
best-effort diagnostic evidence; an absent or invalid sidecar remains explicitly
`missing` or `invalid`, never an observed NPU result. A process killed before
cleanup can leave no retained record. Operators own retention of these files.

The sidecar is engine-produced data inside the candidate boundary, not a new
security attestation or public verdict. Admission uses a frozen known kernel,
its final result and the retained device observations together. Require a
record for the exact execution attempt and verify all expected case reference
forwards completed with observed NPU outputs. Also rerun the host-file denial
canary and declared performance method on the new immutable build. These gates
are diagnostics, not experiment samples.
