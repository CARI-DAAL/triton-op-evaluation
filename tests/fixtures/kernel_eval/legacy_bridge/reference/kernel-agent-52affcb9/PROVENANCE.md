# Vendored reference files

These two modules are vendored **verbatim** from the Triton kernel-agent
harness at revision `52affcb987de470c88a3dcc06d66f1f8a09ff75f` — the same
revision the open-evo `triton-uni-agent` baseline harness was materialized
from (see its `provenance.json`). They define the observation contract the
baseline model was trained against; the byte-parity test in
`../test_legacy_alignment.py` compares our port against them so the contract
cannot drift silently.

| File | Source path in the kernel-agent repo |
| --- | --- |
| `evaluation_observation.py` | `kernel_agent/eval/evaluation_observation.py` |
| `error_classify.py` | `kernel_agent/tools/kernel_eval/error_classify.py` |

License: Apache-2.0 (`LICENSE`, from the same revision). Do not edit these
files; if the contract ever intentionally changes, vendor the new revision
here and update `legacy_contract.py` together.
