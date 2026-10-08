import asyncio
import json
import os
from pathlib import Path
import sys

REPO = Path('/home/l00515014/op_gen/self-evo/Uniagent-triton-ascend-evo')
DEPLOY = Path('/home/e84381970/triton-eval-machine2')
sys.path.insert(0, str(REPO))
os.environ.update(TRITON_EVAL_BACKEND='service', TRITON_EVAL_SERVICE_URL='http://192.169.0.227:11907',
                  TRITON_EVAL_SERVICE_CONFIG='/run/triton-eval-client.json', TRITON_TRAIN_BEST_FIRST='1')
from examples.triton_agent.synth_common import setup_workspace, upload_workspace, create_sandbox_env
from examples.triton_agent.reward import evaluate_triton_workspace
from uni_agent.tools.run_verify import RunVerifyTool

REFERENCE = '''import torch
import json
import os
class Model(torch.nn.Module):
    def forward(self, x, y):
        return x + y
def get_input_groups():
    with open(os.path.join(os.path.dirname(__file__), "cases.json")) as stream:
        cases = [json.loads(line) for line in stream if line.strip()]
    return [[torch.randn(c["n"], dtype=torch.float32), torch.randn(c["n"], dtype=torch.float32)] for c in cases]
def get_init_inputs():
    return []
'''
KERNEL = '''import torch
import triton
import triton.language as tl
@triton.jit
def add_kernel(X, Y, Z, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask, other=0)
    y = tl.load(Y + offsets, mask=mask, other=0)
    tl.store(Z + offsets, x + y, mask=mask)
class ModelNew(torch.nn.Module):
    def forward(self, x, y):
        z = torch.empty_like(x)
        add_kernel[(triton.cdiv(x.numel(), 128),)](x, y, z, x.numel(), BLOCK=128)
        return z
'''

async def main():
    op = 'EvalServiceMigrationAdd'
    work = DEPLOY / 'smoke-host'
    ws = '/opt/workspace_evalsvc_smoke/agent_workdir'
    task = dict(op_name=op, task_code=REFERENCE, support_files={'cases.json': '{"n":128}\n{"n":1024}\n'})
    setup_workspace(task, work)
    env = create_sandbox_env('evalsvc-migration-smoke', attach_port=22199, attach_auth_token=(DEPLOY / 'smoke-auth').read_text().strip(),
                             tool_install_dir='/opt/uni-agent-tools-evalsvc/smoke', op_name=op, workspace_dir=ws)
    await env.start()
    try:
        print('ENV_STARTED', flush=True)
        await upload_workspace(env, work, op_name=op, workspace_dir=ws)
        await env.install_tools([RunVerifyTool()])
        await env.write_file(ws + '/src/' + op + '_triton_ascend_impl.py', KERNEL)
        print('WORKSPACE_READY', flush=True)
        out = await env.communicate('run_verify', timeout=2100, check='raise')
        print(out, flush=True)
        reward, evaluation = await evaluate_triton_workspace(env, {'op_name': op}, workspace_dir=ws)
        metrics = evaluation['metrics']
        report = {'reward': reward, 'metrics': metrics, 'evaluation': evaluation, 'run_verify_output': out, 'workspace': ws}
        (DEPLOY / 'smoke-report.json').write_text(json.dumps(report, indent=2))
        assert metrics.get('passed_cases') == metrics.get('total_cases') == 2, metrics
        assert metrics.get('speedup_vs_torch', 0) > 0, metrics
        print('SMOKE_PASSED', json.dumps({'passed_cases': 2, 'reward': reward, 'speedup': metrics['speedup_vs_torch']}), flush=True)
    finally:
        await env.close()

if __name__ == '__main__':
    asyncio.run(main())
