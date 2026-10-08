# Historical material

The working tree contains supported implementation, behavior tests and current
operating instructions. Completed implementation plans, superseded designs,
retired compile experiments and host-specific deployment snapshots remain in Git.

| Material | Last snapshot before repository cleanup |
| --- | --- |
| Initial designs | [docs/specs](https://github.com/CARI-DAAL/triton-op-evaluation/tree/9bec91c7652a1654756990f1098513cc7ba5666e/docs/specs) |
| Implementation plans | [docs/plans](https://github.com/CARI-DAAL/triton-op-evaluation/tree/9bec91c7652a1654756990f1098513cc7ba5666e/docs/plans) |
| Optimization plans and sketches | [docs/superpowers](https://github.com/CARI-DAAL/triton-op-evaluation/tree/9bec91c7652a1654756990f1098513cc7ba5666e/docs/superpowers) |
| Cardless compilation experiments | [scripts/spike](https://github.com/CARI-DAAL/triton-op-evaluation/tree/9bec91c7652a1654756990f1098513cc7ba5666e/scripts/spike) |
| Machine-specific deployment snapshot | [deploy/nb2-build-prep](https://github.com/CARI-DAAL/triton-op-evaluation/tree/9bec91c7652a1654756990f1098513cc7ba5666e/deploy/nb2-build-prep) |
| Round-two local validation reports | [docs/validation/round2-v3](https://github.com/CARI-DAAL/triton-op-evaluation/tree/9bec91c7652a1654756990f1098513cc7ba5666e/docs/validation/round2-v3) |

Current decisions live in [adr/](adr/) and [architecture/](architecture/).
Legacy bridge bytes used by regression checks live under
[tests/fixtures/kernel_eval/legacy_bridge](../tests/fixtures/kernel_eval/legacy_bridge/)
with their hash manifest and upstream license. They are test inputs, not deployable
code. Two input-default evidence files remain beside their
[active explanation](architecture/candidate-input-defaults.md).
