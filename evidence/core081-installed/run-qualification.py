"""Serial installed-wheel qualification with conservative RAM/host guards."""
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import tarfile
import time

ROOT=Path('/ml2/codex-storage/future-workspaces/2026-10-21/task-6/krea2')
BASE=ROOT.parent
OUT=ROOT/'evidence/core081-installed'
OWNER=Path('/ml2/codex-storage/future-workspaces/2026-10-21/task-2')
ENV=BASE/'.venv-krea-core081'
PY=ENV/'bin/python'
RAM=2*1024**3
FLOOR=16*1024**3
report={'source_head':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        'core_commit':'32f6830f6402e85aa77b6631db64fcbbb2131f6b','core_version':'0.8.1','particlegan':'0.9.1',
        'rss_cap_bytes':RAM,'rss_stop_bytes':int(1.75*1024**3),'data_limit_bytes':int(1.5*1024**3),
        'host_available_floor_bytes':FLOOR,'one_cpu_affinity':True,'phases':[]}
def available():
    return int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
def limits():
    resource.setrlimit(resource.RLIMIT_DATA,(int(1.5*1024**3),int(1.5*1024**3)))
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    os.setsid()
def group_rss(group):
    total=peak=0
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit(): continue
        try:
            if int((proc/'stat').read_text().rsplit(')',1)[1].split()[2])!=group: continue
            rss=int(next(x.split()[1] for x in (proc/'status').read_text().splitlines() if x.startswith('VmRSS:')))*1024
            total+=rss; peak=max(peak,rss)
        except (OSError,ValueError,StopIteration): pass
    return total,peak
common=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',
            OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
            TOKENIZERS_PARALLELISM='false',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTHONDONTWRITEBYTECODE='1',
            PIP_NO_INDEX='1',PIP_DISABLE_PIP_VERSION_CHECK='1',PYTHONPATH='')
report['environment']={k:common[k] for k in ('CUDA_VISIBLE_DEVICES','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONPATH')}
def save(): (OUT/'qualification-receipt.json').write_text(json.dumps(report,indent=2)+'\n')
def run(name,command,cwd='/tmp',timeout=180):
    state={'name':name,'command':[str(x) for x in command],'cwd':str(cwd),'min_host_available_bytes':available(),
           'peak_group_rss_bytes':0,'peak_process_rss_bytes':0}
    report['phases'].append(state); save()
    if available()<FLOOR:
        state.update(outcome='not_started',reason='host available below16GiB'); save(); raise RuntimeError(state)
    start=time.monotonic()
    with (OUT/(name+'.log')).open('w') as log:
        child=subprocess.Popen(state['command'],cwd=cwd,env=common,stdout=log,stderr=subprocess.STDOUT,preexec_fn=limits)
        while child.poll() is None:
            free=available(); total,peak=group_rss(child.pid)
            state['min_host_available_bytes']=min(state['min_host_available_bytes'],free)
            state['peak_group_rss_bytes']=max(state['peak_group_rss_bytes'],total)
            state['peak_process_rss_bytes']=max(state['peak_process_rss_bytes'],peak)
            if free<FLOOR or total>1.75*1024**3 or time.monotonic()-start>timeout:
                state['stop_reason']='host reserve' if free<FLOOR else 'conservative RAM guard' if total>1.75*1024**3 else 'timeout'
                os.killpg(child.pid,signal.SIGKILL); break
            time.sleep(.05)
        state.update(returncode=child.wait(),elapsed_seconds=time.monotonic()-start)
    state['outcome']='passed' if state['returncode']==0 else 'failed'
    save(); print(name+': '+state['outcome'],flush=True)
    if state['returncode']:
        print((OUT/(name+'.log')).read_text()[-16000:],flush=True)
        raise RuntimeError(name+' failed')

if __name__=='__main__':
    import sys
    if len(sys.argv)>1:
        report=json.loads((OUT/'qualification-receipt.json').read_text())
        run(sys.argv[1],sys.argv[2:]); raise SystemExit()
    assert report['source_head']=='c5bf060bd15925ff2e9ea31415ceabf232717699'
    original=json.loads((ROOT/'evidence/core081/source-adoption.json').read_text())
    for name,digest in original['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    canonical=OWNER/'evidence/core081-build/receipt.json'
    lock=json.loads((ROOT/'krea2/shared.lock.json').read_text())
    assert hashlib.sha256(canonical.read_bytes()).hexdigest()==lock['canonical_receipt_sha256']
    assert json.loads(canonical.read_text())['source_sha256']==lock['sources']
    core=OWNER/'evidence/core081-build/particle_sliders_core-0.8.1-py3-none-any.whl'
    assert hashlib.sha256(core.read_bytes()).hexdigest()==lock['wheel_sha256']
    report['canonical_source_files']=len(lock['sources']); report['core_wheel_sha256']=lock['wheel_sha256']; save()
    run('venv',['/home/mikkel/anaconda3/envs/minimax-music3/bin/python3.11','-I','-m','venv','--system-site-packages',ENV])
    source=OUT/'source'; source.mkdir(exist_ok=True)
    run('archive',['git','-C',ROOT,'archive','--format=tar','--output',OUT/'source.tar',report['source_head']])
    with tarfile.open(OUT/'source.tar') as archive: archive.extractall(source)
    artifacts=OUT/'artifacts'; artifacts.mkdir(exist_ok=True)
    run('sdist',[PY,'-I','-c',f'import setuptools.build_meta as build; build.build_sdist({str(artifacts)!r})'],source)
    sdist=next(artifacts.glob('krea2*.tar.gz'))
    run('wheel',[PY,'-I','-m','pip','wheel','--no-index','--no-deps','--no-cache-dir','--no-build-isolation','--wheel-dir',artifacts,sdist])
    wheel=next(artifacts.glob('krea2*.whl'))
    run('install',[PY,'-I','-m','pip','install','--no-index','--no-deps',core,OWNER/'dependencies/particlegan-0.9.1-py3-none-any.whl',wheel])
    report['artifacts']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts.iterdir()};save()
    tests=[str(source/'tests'/name) for name in ('test_lifecycle.py','test_native_inference.py','test_pg09_product.py','test_native_checkpoint.py')]
    code='''import json,sys,importlib.metadata as md
from pathlib import Path
import torch,krea2,particle_sliders,particlegan
from krea2.shared import validate_shared_runtime
torch.set_num_threads(1); torch.set_num_interop_threads(1)
assert '/task-6/.venv-krea-core081/' in krea2.__file__
assert '/task-6/.venv-krea-core081/' in particle_sliders.__file__
print(json.dumps(dict(origins={n:sys.modules[n].__file__ for n in ('krea2','particle_sliders','particlegan')},versions={n:md.version(n) for n in ('torch','diffusers','peft','transformers','safetensors','particle-sliders-core','particlegan')},shared=validate_shared_runtime(),cpu_threads=torch.get_num_threads(),cuda_initialized=torch.cuda.is_initialized()),indent=2),flush=True)
import pytest
result=pytest.main(sys.argv[1:])
assert not torch.cuda.is_initialized()
print('CUDA initialized after tests: False',flush=True)
sys.exit(result)
'''
    run('installed-tests',[PY,'-I','-c',code,'-o','pythonpath=','-p','no:cacheprovider','-q',*tests,'--junitxml='+str(OUT/'installed-tests.xml'),'--basetemp='+str(OUT/'tmp')],timeout=180)
    print('BASE QUALIFICATION COMPLETE',flush=True)
