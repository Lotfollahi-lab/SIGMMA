import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

out = Path(__file__).resolve().parent
repo = out.parents[1]
profile = sys.argv[1]
prefix = Path('/tmp/sigmma-conda-' + profile)
py = str(prefix / 'bin/python')
work = out / profile
work.mkdir(exist_ok=True)
env = {k: v for k, v in os.environ.items() if not k.startswith(('PYTHON', 'PIP_', 'UV_')) and k != 'DGL_LIBRARY_PATH'}
env.update(PYTHONNOUSERSITE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', DGLBACKEND='pytorch')
env['LD_LIBRARY_PATH'] = str(prefix / 'lib') + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
if (work / 'result.json').exists():
    first = work / 'initial_attempt'
    first.mkdir(exist_ok=True)
    for f in list(work.glob('*.log')) + [work / 'result.json']:
        if not (first / f.name).exists():
            shutil.copy2(f, first / f.name)
state = dict(profile=profile, prefix=str(prefix), source='official dglteam Conda binary', steps={}, status='running')
state['runtime_environment'] = {k: env[k] for k in ('LD_LIBRARY_PATH', 'PYTHONNOUSERSITE', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'DGLBACKEND')}
def save():
    (work / 'result.json').write_text(json.dumps(state, indent=2))
def run(name, args, timeout=1800):
    print(profile, name, flush=True)
    with (work / (name + '.log')).open('w') as log:
        try:
            rc = subprocess.run(args, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = 'timeout'
    state['steps'][name] = dict(command=args, exit_code=rc)
    save()
    if rc != 0:
        state['status'] = 'failed_at_' + name
        save()
        sys.exit(1)

save()
for f in (prefix / 'conda-meta').glob('dgl-*.json'):
    shutil.copy2(f, work / 'dgl-conda-provenance.json')
pip = [py, '-I', '-m', 'pip', '--isolated', '--disable-pip-version-check']
index = 'cpu' if profile == 'cpu' else 'cu121'
run('torch_install', pip + ['install', 'torch==2.3.1', 'torchvision==0.18.1', '--index-url', 'https://download.pytorch.org/whl/' + index])
run('wheel_build', pip + ['wheel', '--no-deps', '--wheel-dir', str(work / 'dist'), str(repo)])
wheel = next((work / 'dist').glob('sigmma*.whl'))
state['wheel_sha256'] = hashlib.sha256(wheel.read_bytes()).hexdigest()
run('sigmma_install', pip + ['install', '--index-url', 'https://pypi.org/simple', str(wheel)])
run('pip_check', pip + ['check'])
run('inventory', pip + ['list', '--format=json'])
shutil.copy2(repo / 'scripts/check_install.py', work / 'check_install.py')
run('runtime', [py, '-I', str(work / 'check_install.py'), '--device', 'cpu' if profile == 'cpu' else 'cuda'])
for command in ('sigmma-train', 'sigmma-retrieval', 'sigmma-extract-embeddings', 'sigmma-precompute-hier-cache'):
    run(command, [str(prefix / 'bin' / command), '--help'], 120)
state['status'] = 'passed'
save()
print(json.dumps(state, indent=2), flush=True)
