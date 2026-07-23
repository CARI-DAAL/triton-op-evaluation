# triton-evaluation

Standalone FastAPI + Redis evaluator service wrapping the `triton-op-verifier` skill.

## Run

### Quick start (script)

```bash
cp .env.example .env        # then edit .env: set TRITON_EVAL_SKILL_SCRIPTS_DIR and TRITON_EVAL_CARDS
bash scripts/start.sh       # starts the API + one worker per card in TRITON_EVAL_CARDS
```

`scripts/start.sh` sources `.env`, optionally sources `CANN_SET_ENV_PATH`, then launches
`uvicorn run_api:app` and one `run_worker.py --card <n>` per card (each pins
`ASCEND_RT_VISIBLE_DEVICES`). Logs go to `LOG_DIR/` (`api.log`, `worker_<card>.log`).
Ctrl-C stops everything. For production, supervise with systemd/docker and restart on crash.

### Manual

```bash
export TRITON_EVAL_REDIS_URL=redis://localhost:6379/0
export TRITON_EVAL_SKILL_SCRIPTS_DIR=/abs/path/to/cannbot-skills/ops/triton-op-verifier/scripts

python3 run_api.py                 # API on 0.0.0.0:8000
python3 run_worker.py --card 0 &   # one worker per physical NPU card
python3 run_worker.py --card 1 &
```

See `.env.example` for all environment variables.

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
