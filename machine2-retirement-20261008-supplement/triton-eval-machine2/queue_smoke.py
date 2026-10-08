"""Real 12-request concurrency probe; read-only observation of broker leases."""
import concurrent.futures
import json
from pathlib import Path
import time
import urllib.request
import uuid
import redis

DEPLOY = Path('/home/e84381970/triton-eval-machine2')
config = json.loads((DEPLOY / 'client.json').read_text())
payload = json.loads((DEPLOY / 'smoke-pending.json').read_text())['request']
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
prefix = 'triton-eval:machine2:v2.1:'
broker = redis.Redis(host='127.0.0.1', port=16379, decode_responses=True)
tag = uuid.uuid4().hex[:10]

def api(method, path, body=None, index=0):
    request = urllib.request.Request(config['url'] + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={'Authorization': 'Bearer ' + config['token'], 'Content-Type': 'application/json',
                 'X-Session-ID': f'migration-queue-{tag}-{index}', 'Idempotency-Key': f'migration-queue-{tag}-{index}'}, method=method)
    with opener.open(request, timeout=30) as response:
        return json.load(response)

with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
    jobs = list(pool.map(lambda i: api('POST', '/evaluations?wait_seconds=0', payload, i), range(12)))
ids = [j['evaluation_id'] for j in jobs]
report = {'ids': ids, 'snapshots': [], 'cards_used': [], 'max_running': 0, 'max_queued': 0, 'max_per_card': 0}
(DEPLOY / 'queue-smoke-report.json').write_text(json.dumps(report, indent=2))
print('SUBMITTED', json.dumps(ids), flush=True)
seen = set()
deadline = time.monotonic() + 1800
previous = None
while time.monotonic() < deadline:
    counts = {}; running_cards = {}; states = {}; bindings = {}
    batch = broker.pipeline(transaction=True)
    for identity in ids:
        batch.hgetall(prefix + 'resource:' + identity)
    for identity, row in zip(ids, batch.execute()):
        resource = json.loads(row['resource'])
        state = resource['state']; states[identity] = state
        counts[state] = counts.get(state, 0) + 1
        binding = json.loads(row.get('attempt_card') or '{}')
        if 'card' in binding:
            card = binding['card']; seen.add(card); bindings[identity] = card
            if state == 'running':
                running_cards[card] = running_cards.get(card, 0) + 1
    report['max_running'] = max(report['max_running'], counts.get('running', 0))
    report['max_queued'] = max(report['max_queued'], counts.get('queued', 0))
    report['max_per_card'] = max(report['max_per_card'], max(running_cards.values(), default=0))
    current = [counts, running_cards]
    if current != previous:
        snapshot = {'time': time.time(), 'counts': counts, 'running_cards': running_cards, 'bindings': bindings}
        report['snapshots'].append(snapshot); previous = current
        print('QUEUE', json.dumps(snapshot), flush=True)
    if all(s in {'completed', 'failed', 'timed_out'} for s in states.values()):
        break
    time.sleep(0.25)
report['cards_used'] = sorted(seen)
report['results'] = [api('GET', '/evaluations/' + identity) for identity in ids]
(DEPLOY / 'queue-smoke-report.json').write_text(json.dumps(report, indent=2))
assert report['max_running'] == 8, report['max_running']
assert report['max_queued'] >= 4, report['max_queued']
assert report['max_per_card'] == 1, report['max_per_card']
assert report['cards_used'] == list(range(8)), report['cards_used']
for result in report['results']:
    assert result['state'] == 'completed', result
    assert result['result']['correctness']['data']['passed'] is True, result
    assert result['result']['performance']['status'] == 'succeeded', result
print('QUEUE_SMOKE_PASSED', json.dumps({k: report[k] for k in ['cards_used','max_running','max_queued','max_per_card']}), flush=True)
