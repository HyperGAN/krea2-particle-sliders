import json
from pathlib import Path
import shutil
import torch
from particle_sliders.lifecycle import SessionConfig, open_session, UnsupportedOperation
from particle_sliders.session_cli import main

root=Path('/ml2/codex-storage/future-workspaces/2026-10-21/task-6/krea2/evidence/core081-installed/cli-runtime')
root.mkdir(exist_ok=False)
torch.set_num_threads(1); torch.set_num_interop_threads(1)
config=SessionConfig('krea2','installed-core081-cli',dummy=True,steps=4,ema_decay=.9)
config_file=root/'config.json'
config_file.write_text(json.dumps(config.as_dict()))
main(['capabilities','--config',str(config_file)])
main(['train','--config',str(config_file),'--stop-after','2','--selection','ema','--out',str(root/'initial')])
original={p.name:p.read_bytes() for p in (root/'initial').iterdir()}
main(['train','--config',str(config_file),'--resume',str(root/'initial/training.pt'),'--selection','ema','--out',str(root/'continued')])
assert original=={p.name:p.read_bytes() for p in (root/'initial').iterdir()}
for run,step in (('initial',2),('continued',4)):
    manifest=json.loads((root/run/'manifest.json').read_text())
    assert manifest['completed_steps']==step
    assert len((root/run/'metrics.jsonl').read_text().splitlines())==2
main(['evaluate','--config',str(config_file),'--artifact',str(root/'continued/adapter.pt'),
      '--strength','-.5','--out',str(root/'v1-response.pt')])
assert torch.isfinite(torch.load(root/'v1-response.pt',weights_only=True)['response']).all()
try:
    main(['render','--config',str(config_file),'--out',str(root/'unsupported.png')])
except UnsupportedOperation:
    pass
else:
    raise AssertionError('tiny render was accepted')
assert not (root/'unsupported.png').exists()
expected={}
with open_session(config,resume=root/'continued/training.pt') as training:
    before=training.checkpoint()['sha256']
    prepared=training.game.rows[0]
    request=dict(prompt='neutral',latent=prepared.row.latent.cpu().tolist(),timestep=prepared.row.timestep.item())
    (root/'request.json').write_text(json.dumps(request))
    raw=training.export_inference(root/'raw.pt')
    ema=training.export_inference(root/'ema.pt',selection='ema')
    assert any(not torch.equal(raw['network'][name],ema['network'][name]) for name in raw['network'])
    expected['raw']=training.evaluate(strength=-.5)
    state={key:value.clone() for key,value in training.game.network.state_dict().items()}
    try:
        training.game.network.load_state_dict(training.ema.values)
        expected['ema']=training.evaluate(strength=-.5)
    finally:
        training.game.network.load_state_dict(state)
    assert training.checkpoint()['sha256']==before
    for artifact in (raw,ema):
        def no_data(value):
            if isinstance(value,dict):
                assert not ({'rows','teacher_cache','frozen_cache','optimizer','critic','training_data','cache_file'}&value.keys())
                for child in value.values(): no_data(child)
            elif isinstance(value,(tuple,list)):
                for child in value:no_data(child)
        no_data(artifact['specification'])
torch.save(expected,root/'expected.pt')
# Keep only explicit native requests/artifacts/expected values, not training input/state.
for name in ('initial','continued'): shutil.rmtree(root/name)
config_file.unlink(); (root/'v1-response.pt').unlink()
assert not torch.cuda.is_initialized()
(root/'producer-checks.json').write_text(json.dumps(dict(checks=['installed CLI capabilities','split CLI train/resume preserves original receipts',
    'CLI fixed four-update budget and manifests','v1 CLI evaluation','tiny render fails before output',
    'raw/EMA v2 exports differ','v2 export leaves full state unchanged','native specification has no training payload',
    'training config/checkpoints deleted before fresh process'],cuda_initialized=False),indent=2)+'\n')
print('NATIVE ARTIFACT PRODUCER PASSED',flush=True)
