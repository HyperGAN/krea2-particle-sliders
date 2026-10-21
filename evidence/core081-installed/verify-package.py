import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import tarfile
import zipfile

out=Path('/ml2/codex-storage/future-workspaces/2026-10-21/task-6/krea2/evidence/core081-installed')
source=out/'source'
wheel=next((out/'artifacts').glob('*.whl'))
installed=Path(metadata.distribution('krea2-particle-sliders').locate_file(''))
matched={}
with zipfile.ZipFile(wheel) as archive:
    for name in archive.namelist():
        if name.startswith(('krea2/','configs/')) and name.endswith(('.py','.json','.yaml')):
            data=archive.read(name)
            assert data==(source/name).read_bytes()==(installed/name).read_bytes(),name
            matched[name]=hashlib.sha256(data).hexdigest()
    names=set(archive.namelist())
    assert {'krea2/native_inference.py','krea2/native_runtime.py','krea2/shared.lock.json'}<=names
    entry=next(name for name in names if name.endswith('.dist-info/entry_points.txt'))
    assert 'particle_sliders.inference_backends' in archive.read(entry).decode()
with tarfile.open(next((out/'artifacts').glob('*.tar.gz'))) as archive:
    names=archive.getnames()
    assert any(name.endswith('/LIFECYCLE.md') for name in names)
    assert any(name.endswith('/tests/test_native_inference.py') for name in names)
lock=json.loads((source/'krea2/shared.lock.json').read_text())
assert lock['particlegan']=='0.9.1'
from krea2.shared import validate_shared_runtime
from particle_sliders.pg09 import _released_sources
runtime=validate_shared_runtime()
pg_digest=_released_sources()
import torch
assert not torch.cuda.is_initialized()
(out/'package-byte-check.json').write_text(json.dumps(dict(source_head='c5bf060bd15925ff2e9ea31415ceabf232717699',
    matched_files=matched,matched_count=len(matched),canonical_core_files=len(lock['sources']),
    shared=runtime,particlegan_sources_sha256=pg_digest,cuda_initialized=False),indent=2)+'\n')
print('Wheel/source/installed bytes match:',len(matched),'files; canonical core map:',len(lock['sources']))
