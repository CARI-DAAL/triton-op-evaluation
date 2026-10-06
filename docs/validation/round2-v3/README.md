# Controlled local round2 service checks

The catalog reports bind the assembler file SHA recorded in
[the digest transition](adapter-digest-transition-local.json). They describe
the same assembler code committed with these reports: 20 official tasks,
1046 cases, byte-identical reassembly, and changes from the original staging
adapter to the unfrozen v1 loader. Official input digests and counts did not
change. Reference digests changed with declared loader fixes, bounded-memory
single-case iteration and pure case counting.

These are CPU assembly/catalog checks. Earlier machine2 revision 430109f
passed 71 native-environment structural tests; its source differs from this
final loader and must not substitute for W3 against the final revision.

The full controlled CPU regression used:
`python -m pytest -q -m 'not npu and not release_gate'`
and passed 1197 tests, with 11 skipped and 6 deselected. This includes 23
canonical historical renderer goldens, four synthetic missing branches,
durable native-route/budget/ownership tests and real local HTTP/TLS handshake
plus fail-closed listener cleanup. It establishes no NPU A/B equality,
throughput improvement, 1046-case device acceptance, or live cutover.

- [Registration](catalog-registration-local.json)
- [Replay](catalog-replay-local.json)
- [Digest transition](adapter-digest-transition-local.json)
