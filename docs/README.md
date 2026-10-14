# Documentation

The running service exposes its HTTP schemas at `/openapi.json` and interactive
documentation at `/docs`. These pages explain workflows and behavior that a schema
alone cannot convey.

## Use the service

- [Client plugin and skill](../plugins/triton-eval/README.md)
- [Submit, poll and recover evaluations](usage/evaluations.md)
- [Operator shortcuts: verify, benchmark and profile](usage/kernel-eval.md)

## Operate an instance

- [Deploy on one or multiple hosts](operations/deployment.md)
- [Recovery, retries and upgrades](operations/recovery.md)
- [NPU 全量验证方案：功能、恢复、压力与稳定性](operations/npu-validation.md)
- [NPU 验证与反馈提速测试指南 草案](operations/feedback-performance.md)
- [官方 BatchMatmul 单题 fixture](../tests/fixtures/npu_kernelbench/batchmatmul/README.md)
- [Prepare the profiler runtime](operations/profiler-runtime.md)
- [Prepare and use reference caches](operations/reference-cache.md)

## Understand the implementation

- [Domain vocabulary](../README.md#domain-vocabulary)
- [Operator input and durable execution](architecture/kernel-evaluation.md)
- [Profile evidence ownership](architecture/profile-evidence.md)
- [Card health](architecture/card-health.md)
- [Attempt card provenance](architecture/attempt-card-provenance.md)
- [Engine sandbox](architecture/engine-sandbox.md)
- [Reference device policy](architecture/reference-device-policy.md)
- [Legacy reference input compatibility](architecture/candidate-input-defaults.md)

## Maintain it

- [Contributing and repository conventions](../README.md#contributing)
- [Test guide](../tests/README.md)
- [Supported scripts](../scripts/README.md)
- [Instance ownership decision](adr/0001-evaluation-instance-ownership.md)
- [Admission and evidence decision](adr/0002-admission-before-execution.md)
- [Historical designs and retired experiments](history.md)
