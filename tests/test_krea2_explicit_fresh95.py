"""FOUR NEW authored/unrun software cases, no pretrained files or old proofs.

Random tiny original Krea/PEFT constructor; targets below are explicit synthetic
software data, never teacher/native-quality evidence. Root alone may execute.
"""
from copy import deepcopy
import math
from types import SimpleNamespace

import pytest
import torch

from krea2.synthetic import staged_native
from krea2.fresh95 import open_training, open_inference
from krea2.fresh95 import api, policy
from krea2.fresh95.ownership import Components, NativeOwner


OPTIONS = dict(dummy=True, device='cpu', base_precision='float32',
               steps=4, seed=7, batch=2, preset='bispf_optimized', lr=.001, ema_decay=None)


def factory_fixture(record=None, *, cache=True, release=None):
    def factory(request):
        native = staged_native()  # No pretrained/teacher call.
        if record is not None:
            record.append(native)
        rows = None
        if cache:
            rows = dict(edit=[], preserve=[], key={'fixture': 'NEW synthetic fresh95 software fields'})
            for holding, name in ((False, 'edit'), (True, 'preserve')):
                for index in range(2):
                    z = torch.randn(1, 4, 16, generator=torch.Generator().manual_seed(41 + index))
                    baseline = torch.full_like(z, .02 * index)
                    rows[name].append(dict(z=z, t=torch.tensor(400. + 100 * index),
                        prompt='hold' if holding else 'neutral',
                        positive='hold' if holding else 'positive',
                        baseline=baseline, teacher=baseline.clone() if holding else baseline + .1))
        return Components(native, rows, 'NEW-synthetic-fields-not-native-teachers-v1' if cache else None, release)
    return factory


def arguments(**extra):
    return dict(options=deepcopy(OPTIONS), mode='fresh', cohort_id='NEW-Krea2-software-only',
                owner_factory=factory_fixture(), owner_admission=lambda request: True, **extra)


def frozen_values(native):
    return {name: value.detach().clone() for name, value in native.model.named_parameters()
            if not value.requires_grad}


def test_new_fresh_before_optimizers_PEFT_QR_FieldCritic_KEEP_and_Parameter_identity(monkeypatch):
    observed = {}
    original_init, original_game = policy.initialize, api.ConditionalPG09
    def initialize(product, adapter, critic, **kwargs):
        rows = policy.policies(product, adapter, critic, model=kwargs['model'])
        observed['ids'] = [id(row['parameter']) for row in rows]
        observed['frozen'] = frozen_values(SimpleNamespace(model=kwargs['model']))
        observed['before'] = {row['name']: policy.tensor_sha(row['parameter']) for row in rows}
        receipt = original_init(product, adapter, critic, **kwargs)
        assert observed['ids'] == [id(row['parameter']) for row in rows]
        for row, detail in zip(rows, receipt['parameter_policies']):
            if detail['role'] == 'active-PEFT-host-a-sqrt5-down-QR':
                weight = row['parameter'].detach().double()
                bound = 1 / math.sqrt(weight.shape[1])
                assert detail['distribution'] == repr(policy.pg_init.Uniform(-bound, bound))
                assert detail['after_sha256'] != detail['before_sha256']
                # Rectangular QR rows have equal host-variance-preserving norm.
                gram = weight @ weight.T
                expected = torch.eye(len(weight), dtype=torch.float64) * (1 / 3)
                assert torch.allclose(gram, expected, rtol=2e-5, atol=2e-6)
            else:
                assert detail['after_sha256'] == observed['before'][row['name']]
            if detail['role'] == 'protected-zero-UP':
                assert torch.count_nonzero(row['parameter']) == 0
        observed['receipt'] = receipt
        return receipt
    def game(backend, rows, critic, **kwargs):
        assert backend.receipt == observed['receipt']  # Receipt precedes constructor/optimizers.
        assert kwargs['particles'] is None
        assert all(row['before_sha256'] == row['after_sha256'] for row in
                   backend.receipt['parameter_policies'] if row['name'].startswith('critic.'))
        result = original_game(backend, rows, critic, **kwargs)
        assert result.prior is None and result.spread is None
        observed['optimizer_seen'] = True
        return result
    monkeypatch.setattr(policy, 'initialize', initialize)
    monkeypatch.setattr(api, 'ConditionalPG09', game)
    natives = []
    args = arguments()
    args['owner_factory'] = factory_fixture(natives)
    with open_training(**args) as owner:
        assert observed['optimizer_seen']
        assert owner.game._signature['backend']['fresh_initialization']['receipt'] == owner.initialization
        assert owner.session.ema.values is None and owner.session.ema.decay is None
        assert all(torch.equal(value, dict(natives[0].model.named_parameters())[name])
                   for name, value in observed['frozen'].items())
        assert set(map(id, owner.game.network.parameters())) == {
            id(p) for p in natives[0].model.parameters() if p.requires_grad}


def test_new_deterministic_named_initialization_and_no_ambient_RNG_or_frozen_change():
    receipts, weights = [], []
    for unused_seed in (123, 986):
        torch.manual_seed(unused_seed)
        before = torch.get_rng_state().clone()
        with open_training(**arguments()) as owner:
            receipts.append(deepcopy(owner.initialization))
            weights.append({k: v.clone() for k, v in owner.game.network.state_dict().items()})
        assert torch.equal(before, torch.get_rng_state())
    assert receipts[0] == receipts[1]
    assert api.equal_state(weights[0], weights[1])
    with open_training(**dict(arguments(), options=dict(OPTIONS, seed=8))) as other:
        assert [r['seed'] for r in receipts[0]['parameter_policies']] != [
            r['seed'] for r in other.initialization['parameter_policies']]


def test_new_strict_whole_FULL_resume_and_RAW_native_inference_skip_initializer(tmp_path, monkeypatch):
    args = arguments()
    with open_training(**args) as owner:
        owner.step()
        full = owner.checkpoint(tmp_path / 'FULL1.pt')
        original = owner.session.checkpoint()
    def forbidden(*args, **kwargs):
        raise AssertionError('initializer called during resume or RAW inference')
    monkeypatch.setattr(policy, 'initialize', forbidden)
    with open_training(**dict(args, mode='resume', resume=full, owner_factory=factory_fixture())) as resumed:
        assert api.equal_state(resumed.session.checkpoint(), original)
        # One software update TOTAL; restore the full optimizer/RNG history,
        # then export/infer without repeating an update for this fixture.
        raw = resumed.export(tmp_path / 'RAW1.pt')
        row = resumed.game.rows[0].row
        response = resumed.game.evaluate(0)
        latent, timestep = row.latent.clone(), row.timestep.clone()
    def no_training(*args, **kwargs):
        raise AssertionError('RAW route constructed game, critic or cache rows')
    monkeypatch.setattr(api, 'ConditionalPG09', no_training)
    monkeypatch.setattr(api, 'FieldCritic', no_training)
    monkeypatch.setattr(api, 'Krea2Rows', no_training)
    with open_inference(raw, dict(device='cpu', base_precision='float32'), mode='raw-inference',
                        owner_factory=factory_fixture(cache=False), owner_admission=lambda request: True) as inference:
        actual = inference.response(prompt='neutral', latent=latent, timestep=timestep, strength=1., seed=0)
        assert torch.equal(actual, response)
        assert inference.session.selection == 'raw'
        assert not hasattr(inference.session, 'game')
        with pytest.raises(Exception):
            inference.session.step()


def test_new_preconstruction_refusals_retryable_cleanup_and_foreign_ownership():
    calls = []
    def forbidden_factory(request):
        calls.append(request)
        raise AssertionError('invalid entry allocated components')
    for changes in (dict(mode='ambiguous'), dict(mode='resume'), dict(options=dict(OPTIONS, ema_decay=.99))):
        args = arguments()
        args.update(owner_factory=forbidden_factory)
        args.update(changes)
        with pytest.raises(ValueError):
            open_training(**args)
    assert not calls
    model = torch.nn.Linear(2, 2)
    original, installed, foreign = object(), object(), object()
    model._gradient_checkpointing_func = original
    release_calls = []
    def release():
        release_calls.append('release')
        if len(release_calls) == 1:
            raise RuntimeError('NEW injected owned-release failure')
    native_owner = NativeOwner(Components(SimpleNamespace(model=model), release=release))
    model._gradient_checkpointing_func = installed
    native_owner.seal()
    model._gradient_checkpointing_func = foreign
    with pytest.raises(BaseExceptionGroup):
        native_owner.close()
    assert model._gradient_checkpointing_func is foreign and release_calls == []
    assert native_owner.closing and native_owner.cleanup_pending and not native_owner.closed
    with pytest.raises(RuntimeError):
        api.OwnedTraining(None, native_owner, {}, {}).step()
    with pytest.raises(RuntimeError):
        api.OwnedInference(None, native_owner).response()
    # Explicit synthetic return of ownership, not production foreign overwrite.
    model._gradient_checkpointing_func = installed
    with pytest.raises(BaseExceptionGroup):
        native_owner.close()
    assert model._gradient_checkpointing_func is original
    assert native_owner.cleanup_pending and not native_owner.closed
    assert len(release_calls) == 1
    native_owner.close()
    native_owner.close()
    assert len(release_calls) == 2
    assert native_owner.closed and not native_owner.closing and not native_owner.cleanup_pending
