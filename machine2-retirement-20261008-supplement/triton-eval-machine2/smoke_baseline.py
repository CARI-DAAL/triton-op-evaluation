"""One archived baseline kernel, through the real CPU AgentEnv, without an LLM."""
import asyncio
import json
from pathlib import Path
import sys
sys.path.insert(0, '/home/e84381970/triton-eval-machine2')
import smoke_agent as smoke

async def main():
    fixture = json.loads((smoke.DEPLOY / 'baseline-smoke-fixture.json').read_text())
    task = fixture['task']; op = task['op_name']
    work = smoke.DEPLOY / 'baseline-smoke-host'
    ws = '/opt/workspace_evalsvc_baseline_smoke/agent_workdir'
    smoke.setup_workspace(task, work)
    env = smoke.create_sandbox_env('evalsvc-baseline-smoke', attach_port=22199,
            attach_auth_token=(smoke.DEPLOY / 'smoke-auth').read_text().strip(),
            tool_install_dir='/opt/uni-agent-tools-evalsvc/baseline-smoke', op_name=op, workspace_dir=ws)
    await env.start()
    try:
        await smoke.upload_workspace(env, work, op_name=op, workspace_dir=ws)
        await env.install_tools([smoke.RunVerifyTool()])
        await env.write_file(ws + '/src/' + op + '_triton_ascend_impl.py', fixture['kernel'])
        print('BASELINE_WORKSPACE_READY', op, flush=True)
        output = await env.communicate('run_verify', timeout=2100, check='raise')
        print(output, flush=True)
        reward, evaluation = await smoke.evaluate_triton_workspace(env, {'op_name':op}, workspace_dir=ws)
        metrics = evaluation['metrics']
        report = {'reward':reward,'evaluation':evaluation,'workspace':ws,'run_verify_output':output}
        (smoke.DEPLOY / 'baseline-smoke-report.json').write_text(json.dumps(report, indent=2))
        assert metrics['passed_cases'] == metrics['total_cases'] == 51, metrics
        assert metrics['speedup_vs_torch'] > 0, metrics
        print('BASELINE_SMOKE_PASSED', json.dumps({k:metrics[k] for k in ['passed_cases','total_cases','speedup_vs_torch']}), flush=True)
    finally:
        await env.close()

asyncio.run(main())
