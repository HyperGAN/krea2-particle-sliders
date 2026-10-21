import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

root = Path('/ml2/codex-storage/future-workspaces/2026-10-21/task-6/krea2')
out = root / 'evidence/core070'
python = '/ml2/codex-storage/future-workspaces/2026-10-21/task-2/.venv-krea311/bin/python'
limit = 4 * 1024**3
floor = 16 * 1024**3

def available():
    return int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))) * 1024

def constrain():
    resource.setrlimit(resource.RLIMIT_DATA, (3 * 1024**3, 3 * 1024**3))
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    os.setsid()

env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
    TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1', OMP_NUM_THREADS='1',
    MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
    VECLIB_MAXIMUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
    PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTHONDONTWRITEBYTECODE='1', PIP_NO_INDEX='1')
bootstrap = '''import sys, importlib.metadata as md, json
from pathlib import Path
sys.path.insert(0, %r)
import torch
import krea2.lifecycle as product
import particle_sliders
from krea2.shared import validate_shared_runtime
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
print(json.dumps(dict(product_origin=product.__file__, core_origin=particle_sliders.__file__,
    versions={n:md.version(n) for n in ('torch','diffusers','peft','transformers','safetensors','pytest','particlegan','particle-sliders-core')},
    shared=validate_shared_runtime(), cpu_threads=torch.get_num_threads(), cuda_initialized=torch.cuda.is_initialized()), indent=2), flush=True)
import pytest
result = pytest.main(%r)
print('CUDA initialized after tests:', torch.cuda.is_initialized(), flush=True)
if torch.cuda.is_initialized():
    raise RuntimeError('unexpected CUDA initialization')
sys.exit(result)
''' % (str(root), ['-o', 'pythonpath=', '-p', 'no:cacheprovider', '-q',
    str(root/'tests/test_lifecycle.py'), str(root/'tests/test_pg09_product.py'),
    '--junitxml='+str(out/'bounded-tests.xml'), '--basetemp=/tmp/krea2-core070-bounded-tests'])
command = [python, '-I', '-c', bootstrap]
receipt = dict(source_head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'], text=True).strip(),
    mode='current product source overlay plus unchanged installed dependencies; not a fresh wheel qualification',
    command=command, memory_limit_bytes=limit, limit_type='hard 3 GiB RLIMIT_DATA plus conservative 3.5 GiB RSS watchdog (below 4 GiB cap)',
    host_available_floor_bytes=floor, cpu_affinity_count=1, min_available_bytes=available(),
    peak_rss_bytes=0, env={k:env[k] for k in ('CUDA_VISIBLE_DEVICES','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTEST_DISABLE_PLUGIN_AUTOLOAD')})
if available() < floor:
    receipt.update(outcome='not_started', reason='host available memory below 16 GiB')
else:
    start=time.monotonic()
    with (out/'bounded-tests.log').open('w') as log:
        child=subprocess.Popen(command, cwd='/tmp', env=env, stdout=log, stderr=subprocess.STDOUT, preexec_fn=constrain)
        while child.poll() is None:
            free=available()
            receipt['min_available_bytes']=min(receipt['min_available_bytes'],free)
            try:
                status=Path(f'/proc/{child.pid}/status').read_text().splitlines()
                rss=int(next(line.split()[1] for line in status if line.startswith('VmRSS:'))) * 1024
                receipt['peak_rss_bytes']=max(receipt['peak_rss_bytes'],rss)
            except (FileNotFoundError,StopIteration):
                rss=0
            if free < floor or rss > 3.5 * 1024**3 or time.monotonic()-start > 120:
                receipt['stop_reason']='host available memory below 16 GiB' if free < floor else 'RSS above conservative 3.5 GiB guard' if rss > 3.5 * 1024**3 else '120 second bounded runtime exceeded'
                os.killpg(child.pid, signal.SIGKILL)
                break
            time.sleep(.05)
        receipt.update(returncode=child.wait(), elapsed_seconds=time.monotonic()-start)
        receipt['outcome']='passed' if receipt['returncode']==0 else 'stopped' if 'stop_reason' in receipt else 'failed'
(out/'bounded-tests-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='command'},indent=2))
print((out/'bounded-tests.log').read_text()[-16000:] if (out/'bounded-tests.log').exists() else '')
