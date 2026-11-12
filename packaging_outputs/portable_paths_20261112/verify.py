"""Build, scan, and test the portable release artifacts."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import zipfile

out = Path(__file__).resolve().parent
repo = out.parents[1]
dist = repo / 'dist' / 'portable'
state = {'status': 'running', 'steps': {}, 'artifacts': {}}
def save():
    (out / 'result.json').write_text(json.dumps(state, indent=2))
def run(name, args, profile='cpu'):
    prefix = '/tmp/sigmma-conda-' + profile
    env = {k: v for k, v in os.environ.items() if not k.startswith(('PYTHON', 'PIP_', 'SIGMMA_')) and k != 'DGL_LIBRARY_PATH'}
    env.update(LD_LIBRARY_PATH=prefix+'/lib', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', DGLBACKEND='pytorch')
    with (out / (name+'.log')).open('w') as log:
        result = subprocess.run(args, env=env, cwd=out, stdout=log, stderr=subprocess.STDOUT, timeout=600)
    state['steps'][name] = {'command': args, 'exit_code': result.returncode}
    save()
    if result.returncode:
        state['status'] = 'failed_at_' + name
        save()
        raise SystemExit(result.returncode)
    print(name, 'passed', flush=True)

py = '/tmp/sigmma-conda-cpu/bin/python'
run('build_tools', [py, '-I', '-m', 'pip', '--isolated', 'install', 'build'])
run('build', [py, '-I', '-m', 'build', '--outdir', str(dist), str(repo)])
needles = [b'/nfs/', b'/lustre/', b'/home/', b'/scratch/', b'/software/', b'/dj16/', b'/team361/']
for artifact in sorted(dist.iterdir()):
    if artifact.suffix == '.whl':
        with zipfile.ZipFile(artifact) as z:
            entries = [(n, z.read(n)) for n in z.namelist()]
    elif artifact.name.endswith('.tar.gz'):
        with tarfile.open(artifact) as t:
            entries = [(m.name, t.extractfile(m).read()) for m in t.getmembers() if m.isfile()]
    else:
        continue
    hits = [name for name, content in entries if any(n in content or n in name.encode() for n in needles)]
    state['artifacts'][artifact.name] = {'sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(), 'files_scanned': len(entries), 'server_path_hits': hits}
    save()
    if hits:
        state['status'] = 'failed_artifact_scan'
        save()
        raise SystemExit(hits)
wheel = next(dist.glob('*.whl'))
for profile in ('cpu', 'cuda'):
    py = '/tmp/sigmma-conda-' + profile + '/bin/python'
    run(profile+'_install', [py, '-I', '-m', 'pip', '--isolated', 'install', '--no-deps', '--force-reinstall', str(wheel)], profile)
    run(profile+'_dependencies', [py, '-I', '-m', 'pip', 'check'], profile)
    run(profile+'_path_tests', [py, '-I', str(repo / 'tests/test_portable_paths.py')], profile)
    run(profile+'_runtime', [py, '-I', str(repo / 'scripts/check_install.py'), '--device', profile], profile)
    for cli in ('sigmma-train', 'sigmma-retrieval', 'sigmma-extract-embeddings', 'sigmma-precompute-hier-cache'):
        run(profile+'_'+cli, ['/tmp/sigmma-conda-'+profile+'/bin/'+cli, '--help'], profile)
state['status'] = 'passed'
save()
