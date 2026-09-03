# THROWAWAY spike: cardless triton-ascend compile

Gate for `compile_stage=cpu_lane` (spec §8). Nothing here is production code.

## Run (on the NPU host, repo checkout)

```bash
mkdir -p spike-cache spike-cold-cache spike-report
export TRITON_EVAL_IMAGE=<pinned sha256 digest>

# 1. cardless compile probe (criteria 1, 4; half of 3)
docker compose -f docker-compose.spike-cardless.yml up --abort-on-container-exit
echo "exit=$?"   # 0 = all corpus expectations met cardless

# 2. on-card cache reuse (criterion 2)
docker compose -f docker-compose.spike-oncard.yml up --abort-on-container-exit

# 3. on-card failure classification via the npu0 smoke service (other half of 3)
python3 scripts/spike/oncard_classify.py \
  --url http://127.0.0.1:11908 --token <trial-token>

# 4. fill docs/superpowers/spikes/<date>-cardless-compile-report.md from
#    spike-report/*.json; record the compile_stage gate decision.
```
