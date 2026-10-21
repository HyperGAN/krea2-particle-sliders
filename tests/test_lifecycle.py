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
