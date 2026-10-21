"""Core 0.8.1 adoption cases, deferred until the next bounded CPU-test slot."""
import pytest
import torch

from particle_sliders.inference import open_inference
from particle_sliders.lifecycle import SessionConfig, UnsupportedOperation
from krea2.lifecycle import BACKEND as TRAINING, _validate_device, preflight
from krea2.native_inference import BACKEND as INFERENCE


def test_v2_load_without_training_construction(monkeypatch):
    torch.set_num_threads(1)
    config = SessionConfig('krea2', 'new-krea-native-v2', dummy=True, steps=4, ema_decay=.9)
    with TRAINING.open(config) as training:
        training.train(stop_after=2)
        raw = training.export_inference()
        ema = training.export_inference(selection='ema')
        prepared = training.game.rows[0]
        request = dict(prompt='neutral', latent=prepared.row.latent.clone(),
                       timestep=prepared.row.timestep.clone())
        expected = training.evaluate(index=0, strength=-.5)
        state = training.checkpoint()['sha256']
        training.export_inference()
        assert training.checkpoint()['sha256'] == state
    assert raw['schema'] == 'particle-sliders-native-inference-v2'
    assert raw['selection'] == 'raw' and ema['selection'] == 'ema'
    assert set(raw) == {'schema', 'specification', 'origin', 'selection', 'network', 'sha256'}
    import krea2.pg09 as product
    import krea2.synthetic as synthetic
    from particle_sliders.field_critic import FieldCritic
    def forbidden(*args, **kwargs):
        raise AssertionError('training construction during native-only loading')
    monkeypatch.setattr(product, 'build_game', forbidden)
    monkeypatch.setattr(synthetic, 'staged_fixture', forbidden)
    monkeypatch.setattr(FieldCritic, '__init__', forbidden)
    monkeypatch.setattr(torch.optim.Optimizer, '__init__', forbidden)
    before = torch.get_rng_state().clone()
    for packet in (raw, ema):
        with open_inference(packet, backend=INFERENCE, operation='response') as inference:
            assert not any(hasattr(inference, name) for name in ('game', 'rows', 'critic', 'optimizer'))
            assert inference.selection == packet['selection']
            actual = inference.response(strength=-.5, **request)
            if packet is raw:
                assert torch.equal(actual, expected)
            for strength in (-1., 0., .5, 1.):
                assert torch.isfinite(inference.response(strength=strength, **request)).all()
            with pytest.raises(UnsupportedOperation):
                inference.train()
    assert torch.equal(before, torch.get_rng_state())
    assert not torch.cuda.is_initialized()


def test_explicit_cuda_device_preflight_needs_no_cuda(monkeypatch):
    from types import SimpleNamespace
    _validate_device({'device': 'cuda:2'}, SimpleNamespace(device=2))
    with pytest.raises(ValueError, match='declare device'):
        _validate_device({}, SimpleNamespace(device=2))
    with pytest.raises(ValueError):
        _validate_device({'device': 'cuda:0'}, SimpleNamespace(device=2))
    with pytest.raises(UnsupportedOperation, match='cached prompt'):
        preflight({'render_prompts': ['encoded']}, dict(operation='render',
            references=False, output_type='pil', native={'prompt': 'unseen'}), {})


def test_native_admission_refusal_preserves_session():
    config = SessionConfig('krea2', 'new-krea-admission-v2', dummy=True, steps=4, ema_decay=.9)
    def reject(request):
        assert request['completed_steps'] == 0
        return False
    with TRAINING.open(config, before_update=reject) as training:
        before = training.checkpoint()['sha256']
        with pytest.raises(UnsupportedOperation, match='admission'):
            training.step()
        assert not training.closed
        assert training.checkpoint()['sha256'] == before


def test_native_observation_restores_external_buffers_and_cache_on_failure():
    from types import SimpleNamespace
    from krea2.native_runtime import observation
    module = torch.nn.Linear(1, 1)
    module.register_buffer('native_only', torch.tensor([3.]), persistent=False)
    original = module.native_only
    position_ids = {'existing': torch.tensor([2.])}
    native = SimpleNamespace(model=module, position_ids=position_ids,
                             pipe=SimpleNamespace(scheduler={'cursor': 0}))
    scheduler = native.pipe.scheduler
    with pytest.raises(RuntimeError, match='probe'):
        with observation(native):
            module.native_only = torch.tensor([9.])
            module.register_buffer('added', torch.ones(1))
            native.position_ids['new'] = torch.tensor([1.])
            native.pipe.scheduler['cursor'] = 9
            native.pipe._interrupt = True
            raise RuntimeError('probe')
    assert module.native_only is original and torch.equal(original, torch.tensor([3.]))
    assert 'native_only' in module._non_persistent_buffers_set and 'added' not in module._buffers
    assert native.pipe.scheduler is scheduler and scheduler == {'cursor': 0}
    assert not hasattr(native.pipe, '_interrupt')
    assert native.position_ids is position_ids and set(position_ids) == {'existing'}
