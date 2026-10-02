# Controlled local official-catalog checks

These reports describe the assembler at service revision
`430109fccd29b8b417567f12acaefc039475694e`. They show 20 official tasks,
1046 cases, byte-identical reassembly and the transition from the original
staging adapter to this unfrozen v1 implementation. The official input
digests and case counts did not change. Reference source digests changed
with loader fixes and the single-case interface.

These are CPU assembly/catalog checks, not NPU correctness, timing or
whole-case admission. A later reference-source change requires new reports
and W3 audits before final freeze; do not bind these intermediate reference
digests as device acceptance for a different revision.

- [Registration](catalog-registration-local.json)
- [Replay](catalog-replay-local.json)
- [Digest transition](adapter-digest-transition-local.json)
