"""CPU native-factory integration; no pretrained/model downloads."""
import torch
import pytest
from particle_sliders.lifecycle import SessionConfig, open_session, backend, UnsupportedOperation
from particle_sliders.pg09 import _hash


def test_common_factory_roundtrip(tmp_path):
    torch.set_num_threads(1)
    config = SessionConfig('krea2', 'lifecycle-cpu-new-v1', **dict(dummy=True, steps=4, ema_decay=.9))
    before = torch.get_rng_state().clone()
    with open_session(config) as full:
        assert torch.equal(before, torch.get_rng_state())
        full.train()
        expected = full.checkpoint()
    with open_session(config) as partial:
        partial.train(stop_after=2)
        state = partial.checkpoint(tmp_path / 'training.pt')
        for strength in (-1., 0., .5, 1.):
            response = partial.evaluate(strength=strength)
            assert torch.isfinite(response).all()
            assert partial.checkpoint()['sha256'] == state['sha256']
    with open_session(config, resume=tmp_path / 'training.pt') as resumed:
        resumed.train()
        assert resumed.checkpoint()['sha256'] == expected['sha256']
        resumed.export(tmp_path / 'adapter.pt', selection='ema')
        ema = resumed.ema.values
    with open_session(config, artifact=tmp_path / 'adapter.pt', operation='evaluate') as inference:
        assert _hash(inference.game.network.state_dict()) == _hash(ema)
        assert torch.isfinite(inference.evaluate()).all()
    assert torch.equal(before, torch.get_rng_state())
    assert not torch.cuda.is_initialized()


def test_preflight_unsupported_before_factory(monkeypatch):
    contract = backend('krea2')
    with pytest.raises(UnsupportedOperation):
        contract.open(SessionConfig('krea2', 'new', dummy=True), operation='render')


def _staged_binding(monkeypatch, *, frozen='frozen', generate=None):
    """Exercise ownership/callback wiring without allocating a native model."""
    from types import SimpleNamespace
    import krea2.lifecycle as product
    import krea2.pg09 as binding

    args = SimpleNamespace(model_id='bbox', transformer_subfolder='transformer',
        revision='r', skeleton_model='raw', skeleton_revision='s', rank=2,
        resolution=32, sample_steps=8, sample_guidance=0., mu=1.15)
    key = dict(model_id='bbox', transformer_subfolder='transformer', revision='r',
        skeleton='raw', skeleton_revision='s', rank=2, resolution=32, sample_steps=8,
        guidance=0., mu=1.15, particles=None, objective='g_adv;d_adv+public_penalty',
        frozen_model_sha256='frozen')
    key['conditioning_identity'] = {'fixture': 'encoding'}
    record = dict(prompt='cached', positive='cached')
    cache = dict(key=key, edit=[record], preserve=[record])
    unloads = []
    model = SimpleNamespace(training=True, gradient_checkpointing=True)
    model.modules = lambda: [model]
    model.parameters = lambda: []
    def unload():
        unloads.append(True)
        return 'unwrapped'
    model.unload = unload
    native = SimpleNamespace(model=model, pipe=SimpleNamespace(scheduler={'cursor': 0}),
                             embeddings={'cached': object()}, frozen_identity=lambda: frozen,
                             position_ids={}, conditioning_identity=key['conditioning_identity'])
    native.generate = generate or (lambda *args: 'pil-result')
    monkeypatch.setattr(product, '_native_arguments', lambda options: args)
    monkeypatch.setattr(product, '_product_identity', lambda value: value)
    monkeypatch.setattr(binding, 'file_digest', lambda path: 'cache-sha')
    import krea2.native_inference as native_backend
    monkeypatch.setattr(native_backend, 'specification', lambda *args, **kwargs: None)
    def build(*args, **kwargs):
        model.gradient_checkpointing = False
        model.training = False
        return SimpleNamespace(backend=None)
    monkeypatch.setattr(binding, 'build_game', build)
    return product, native, cache, unloads


def test_native_cache_identity_failure_releases_adapter(monkeypatch):
    product, native, cache, unloads = _staged_binding(monkeypatch, frozen='other-model')
    with pytest.raises(ValueError, match='frozen model'):
        product.prepare({'cache_file': 'unused'}, native=native, cache=cache)
    assert unloads == [True]
    assert native.pipe.transformer == 'unwrapped'


def test_native_callback_restores_scheduler_on_failure_and_close_is_idempotent(monkeypatch):
    product, native, cache, unloads = _staged_binding(monkeypatch)
    scheduler = native.pipe.scheduler
    def fail(*args):
        native.pipe.scheduler['cursor'] = 9
        raise RuntimeError('sampling failed')
    native.generate = fail
    owned = product.prepare({'cache_file': 'unused'}, native=native, cache=cache)
    try:
        with pytest.raises(RuntimeError, match='sampling failed'):
            owned.render(prompt='cached', strength=-.5, seed=2, images=None)
        assert native.pipe.scheduler is scheduler
        assert scheduler == {'cursor': 0}
    finally:
        owned.close()
        owned.close()
    assert unloads == [True]
    assert native.model.gradient_checkpointing is True
    assert native.model.training is True


def test_native_render_outputs_and_prompt_scope_fail_explicitly(monkeypatch):
    product, native, cache, unloads = _staged_binding(monkeypatch)
    with pytest.raises(UnsupportedOperation, match='render output'):
        product.capabilities({}).check('render', output_type='latent')
    owned = product.prepare({'cache_file': 'unused'}, native=native, cache=cache)
    try:
        with pytest.raises(UnsupportedOperation, match='cached prompt'):
            owned.render(prompt='unseen', strength=.5, seed=2, images=None)
        assert owned.render(prompt='cached', strength=-.5, seed=2, images=None,
                            output_type='pil') == 'pil-result'
    finally:
        owned.close()
    assert unloads == [True]


def test_native_cache_rejects_sampler_mismatch(monkeypatch):
    product, native, cache, unloads = _staged_binding(monkeypatch)
    cache['key']['mu'] = 1.0
    with pytest.raises(ValueError, match='mu'):
        product.prepare({'cache_file': 'unused'}, native=native, cache=cache)
    assert unloads == []  # Rejected before accepting the supplied native runtime.
