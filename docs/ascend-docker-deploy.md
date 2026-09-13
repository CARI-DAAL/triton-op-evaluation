# Ascend deployment

The supported deployment uses the image's declared Python dependencies and
mounts the host CANN toolkit and driver. Follow [the current deployment
instructions](../README.md#one-or-multiple-hosts) and `docker.env.example`.
`docker-compose.yml` handles both single-card and multi-card instances.

Each host owns its local Redis database and card identities. Configure consumer
routing with `EVALUATION_POOL_FILE`; no shared cross-host queue or API load
balancer participates in recovery.

## Coordinated v2.1 cutover

1. Stop new admissions to the previous release and collect all in-flight results
   from their original instance. Retain its Redis volume and matching image.
2. Rebuild the service and consumer against the new published contract digests.
   Regenerate release catalogs pinned to earlier digests. Rebuild sandbox images
   so their `kernel_eval` command calls the new SessionServer facade.
3. Configure each host with an immutable image, its `sha256:...` evaluator build,
   private bind address, card list, service capability registry, catalog and CANN.
   Export its env file, then run `scripts/compose.sh up -d --force-recreate`.
   `restart` does not apply changed environment or mounts.
4. Run the authenticated smoke and target-NPU release gate documented in
   `scripts/npu_release_gate.py`. Point migrated drivers at the new instances
   with their own opaque capability token files.

The new namespace is `triton-eval:v2.1`. Old durable state is retained and is not
interpreted by the new process. Rollback must restore the matching consumer,
image, catalog and original namespace. Do not remove the Redis volume.

See [the contract cutover](architecture/evaluations-v21-contract.md) and
[instance ownership decision](adr/0001-evaluation-instance-ownership.md) for
interface and recovery semantics. Host-specific deployment snapshots have been
removed from this guide; their earlier contents remain in Git history.
