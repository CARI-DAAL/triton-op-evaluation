# Triton Evaluation

Existing evaluation ownership, attempts, check evidence and card health terms
are defined in [CONTEXT.md](CONTEXT.md). These terms describe tool callers.

## Language

**Kernel evaluation**:
A tool caller's request to verify, benchmark or profile a candidate against an
official task, with a bounded wait for recoverable evaluation evidence.

**Tool operation**:
One caller action within a trial. An explicitly identified operation can be
retried without consuming another turn or creating a new evaluation.

**Trial turn**:
A durable allowance consumed by a tool operation. Distinct legacy calls consume
distinct turns even when they recover the same underlying evaluation.

**Optional diagnostics**:
Additional profiling evidence attached to a successful tool observation. Its
absence does not invalidate the primary evidence or consume another trial turn.
