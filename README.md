# triton-evaluation

Standalone FastAPI + Redis evaluator service wrapping the `triton-op-verifier` skill.

## Run

```bash
export TRITON_EVAL_REDIS_URL=redis://localhost:6379/0
export TRITON_EVAL_SKILL_SCRIPTS_DIR=/abs/path/to/cannbot-skills/ops/triton-op-verifier/scripts

# API
python run_api.py

# One worker per physical NPU card (pins ASCEND_RT_VISIBLE_DEVICES)
python run_worker.py --card 0 &
python run_worker.py --card 1 &
# ... one per card. Supervise with systemd/docker; restart on crash.
```

## Endpoints

- `POST /verify` → `{task_id}`
- `POST /benchmark` (requires `verify_task_id` or `verify_not_required`) → `{task_id}`
- `GET /tasks/{task_id}` → status + skill result JSON
- `GET /healthz`

## Test

```bash
pip install -e ".[test]"
python -m pytest -v
```
