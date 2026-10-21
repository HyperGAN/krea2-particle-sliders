from copy import deepcopy
import json
from pathlib import Path
import torch
from particle_sliders.inference import open_inference
from particle_sliders.lifecycle import CheckpointError, UnsupportedOperation
from particle_sliders.pg09 import ConditionalPG09,_hash
from particle_sliders.field_critic import FieldCritic
from particle_sliders.session_cli import main
import krea2.lifecycle as training
import krea2.pg09 as product
import krea2.native_inference as native
import krea2.synthetic as synthetic

root=Path('/ml2/codex-storage/future-workspaces/2026-10-21/task-6/krea2/evidence/core081-installed/cli-runtime')
torch.set_num_threads(1);torch.set_num_interop_threads(1)
assert not (root/'config.json').exists() and not (root/'continued').exists()
def forbidden(*args,**kwargs):raise AssertionError('training construction during native-only inference')
product.build_game=forbidden;synthetic.staged_fixture=forbidden
ConditionalPG09.__init__=forbidden;FieldCritic.__init__=forbidden;torch.optim.Optimizer.__init__=forbidden
expected=torch.load(root/'expected.pt',weights_only=True)
request=json.loads((root/'request.json').read_text())
prior_rng=torch.get_rng_state().clone()
checks=[]
for selection in ('raw','ema'):
    with open_inference(root/(selection+'.pt'),operation='response') as inference:
        assert inference.selection==selection
        assert not any(hasattr(inference,name) for name in ('game','rows','critic','optimizer'))
        assert torch.equal(inference.response(strength=-.5,**request),expected[selection])
        for strength in (-1.,0.,.5,1.): assert torch.isfinite(inference.response(strength=strength,**request)).all()
        for method in (inference.train,inference.step,inference.checkpoint,inference.restore):
            try:method()
            except UnsupportedOperation:pass
            else:raise AssertionError('training operation accepted')
    main(['infer','--artifact',str(root/(selection+'.pt')),'--mode','response','--native',str(root/'request.json'),
          '--strength','-.5','--out',str(root/(selection+'-response.pt'))])
    packet=torch.load(root/(selection+'-response.pt'),weights_only=True)
    assert packet['selection']==selection and torch.equal(packet['value'],expected[selection])
    receipt=json.loads((root/(selection+'-response.pt.json')).read_text())
    assert receipt['selection']==selection
checks+=['fresh-process raw/EMA native response parity','signed/zero/fractional native strengths',
         'no training/teacher/critic/optimizer construction','native session refuses train/step/checkpoint/restore',
         'installed native CLI raw/EMA labels and numeric parity']
assert torch.equal(prior_rng,torch.get_rng_state())
checks.append('global RNG unchanged across direct and CLI inference')
# Allocation-free rejection and owned cleanup after staged validation failure.
allocations=[];closed=[]
original_stage=synthetic.staged_native
original_own=native.own_native
def staged():
    allocations.append(True);return original_stage()
def own(runtime):
    close=original_own(runtime)
    before={name:p.detach().clone() for name,p in runtime.model.named_parameters() if p.requires_grad}
    def checked_close():
        assert all(torch.equal(before[name],p) for name,p in runtime.model.named_parameters() if name in before)
        closed.append(True);close()
    return checked_close
synthetic.staged_native=staged;native.own_native=own
valid=torch.load(root/'raw.pt',weights_only=True)
for kind in ('render','training-options','checksum'):
    packet=deepcopy(valid);options={};operation='response'
    if kind=='render':operation='render'
    if kind=='training-options':options={'cache_file':'forbidden'}
    if kind=='checksum':packet['sha256']='wrong'
    try:open_inference(packet,options=options,operation=operation)
    except (ValueError,CheckpointError,UnsupportedOperation):pass
    else:raise AssertionError(kind+' accepted')
assert not allocations
checks.append('render/training-options/checksum rejection before native allocation')
for kind in ('model','tensor'):
    packet=deepcopy(valid)
    if kind=='model':packet['specification']['identity']['model']['frozen_model_sha256']='different'
    else:
        name=next(iter(packet['network']));packet['network'][name]=torch.zeros(999)
    packet['sha256']=_hash({key:value for key,value in packet.items() if key!='sha256'})
    try:open_inference(packet,operation='response')
    except (ValueError,CheckpointError):pass
    else:raise AssertionError(kind+' accepted')
assert len(allocations)==len(closed)==2
checks.append('model/tensor rejection cleans owned adapter before any weight mutation')
try:main(['infer','--config','forbidden.json','--artifact',str(root/'raw.pt'),'--out',str(root/'bad.pt')])
except SystemExit as exc:assert exc.code==2
else:raise AssertionError('infer accepted training config')
checks.append('native CLI rejects training configuration')
assert not torch.cuda.is_initialized()
(root/'consumer-checks.json').write_text(json.dumps(dict(checks=checks,checks_passed=len(checks),
    rejected_owned_allocations=len(allocations),cleanups=len(closed),cuda_initialized=False),indent=2)+'\n')
print('FRESH NATIVE CONSUMER PASSED',len(checks),'checks',flush=True)
