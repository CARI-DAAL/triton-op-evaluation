# Triton Evaluation

Evaluation of candidate Triton kernels against a reference, with recoverable
results and a known execution environment.

## Language

**EvaluationResource**:
A recoverable evaluation with a stable identity, a deadline, and a lifecycle
independent of the caller's connection.

**Attempt**:
One worker's leased execution of an EvaluationResource. A replacement attempt
belongs to the same evaluation and must not overwrite a newer result.

**Card health**:
The assessment of whether a worker's assigned NPU card is fit to execute another
attempt. It is distinct from the correctness of a candidate kernel.

**Canary**:
A known-good kernel used to assess card health independently of the candidate.

**Drained worker**:
A worker that has suspended new attempts until its card passes a canary.
It is not available execution capacity while drained.

**Suspicious ending**:
An attempt ending without trustworthy execution evidence, such as a lost lease
or an evaluator crash. A normal candidate failure is not a suspicious ending.

**Recovery ladder**:
An ordered sequence of increasingly disruptive card recovery actions, each
followed by a canary to determine whether recovery succeeded.
