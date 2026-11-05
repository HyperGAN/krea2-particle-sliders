"""Installed modern profile only: real tiny Krea2/PEFT, shared PG and HNDL."""
from copy import deepcopy
from importlib.metadata import version, entry_points
import os
from pathlib import Path

import pytest
import torch

if version('particle-sliders-core') != '0.13.0' or not entry_points(group='particle_sliders.image_backends', name='krea2-hndl'):
    pytest.skip('separate exact core0.12 HNDL candidate environment required', allow_module_level=True)

from particle_sliders.inference import open_inference
from particle_sliders.lifecycle import CheckpointError, SessionConfig, open_session
from krea2_hndl.lifecycle import Krea2HNDLFieldCritic


@pytest.mark.parametrize('preset', ['bispf', 'bispf_optimized'])
def test_real_hndl_native_lifecycle_resume_and_inference(preset, tmp_path, monkeypatch):
    torch.set_num_threads(1)
    config = SessionConfig('krea2-hndl', 'krea2-hndl-cpu-v1', dummy=True,
                           steps=4, preset=preset, ema_decay=.9)
    rng = torch.get_rng_state().clone()
    with open_session(config) as full:
        assert isinstance(full.game.critic, Krea2HNDLFieldCritic)
        assert full.game._signature['pg_version'] == '0.9.5'
        assert full.game.rows.response_shape == (1, 4, 1, 4, 4)
        assert full.game._signature['generator_arm'] == 'adapter_gan'
        native = full.game.backend.transformer
        frozen = {n: p.detach().clone() for n, p in native.named_parameters() if not p.requires_grad}
        owned = list(full.game.network.parameters())
        assert len(owned) == len({id(p) for p in owned})
        assert {id(p) for p in owned} == {id(p) for p in native.parameters() if p.requires_grad}
        prepared = full.game.rows[0]
        generated = full.game.backend.generated(prepared)
        gradients = torch.autograd.grad(generated.square().sum(), owned, allow_unused=True)
        assert any(g is not None and g.abs().sum() > 0 for g in gradients)
        assert all(p.grad is None for p in full.game.critic.parameters())
        base = full.evaluate(strength=0)
        assert torch.equal(base, prepared.neutral)
        initial = [p.detach().clone() for p in owned]
        full.train()
        assert any(not torch.equal(a, b) for a, b in zip(initial, owned))
        assert all(torch.equal(p, frozen[n]) and p.grad is None for n, p in native.named_parameters() if n in frozen)
        expected = full.checkpoint()['sha256']
    with open_session(config) as split:
        split.train(stop_after=2)
        packet = split.checkpoint(tmp_path / 'training.pt')
        before = packet['sha256']
        for strength in (-1., 0., .5, 1.):
            assert torch.isfinite(split.evaluate(strength=strength)).all()
            assert split.checkpoint()['sha256'] == before
        bad = deepcopy(packet)
        bad['manifest']['config']['cohort'] = 'another-cohort'
        with pytest.raises(CheckpointError):
            split.restore(bad)
        assert split.checkpoint()['sha256'] == before
    with open_session(config, resume=tmp_path / 'training.pt') as resumed:
        resumed.train()
        assert resumed.checkpoint()['sha256'] == expected
        raw = resumed.export_inference()
        ema = resumed.export_inference(selection='ema')
        row = resumed.game.rows[0].row
        request = dict(prompt='neutral', latent=row.latent.clone(), timestep=row.timestep.clone())
        response = resumed.evaluate(strength=-.5)
        raw_weights = deepcopy(resumed.game.network.state_dict())
        ema_weights = deepcopy(resumed.ema.values)
        assert any(not torch.equal(value, ema_weights[name]) for name, value in raw_weights.items())
        before = resumed.checkpoint()['sha256']
        try:
            resumed.game.network.load_state_dict(ema_weights)
            ema_response = resumed.evaluate(strength=-.5)
        finally:
            resumed.game.network.load_state_dict(raw_weights)
        assert resumed.checkpoint()['sha256'] == before
    import krea2_hndl.lifecycle as product
    import krea2.synthetic as synthetic
    def forbidden(*args, **kwargs):
        raise AssertionError('native-only inference constructed training machinery')
    monkeypatch.setattr(product, 'build_game', forbidden)
    monkeypatch.setattr(synthetic, 'staged_fixture', forbidden)
    monkeypatch.setattr(Krea2HNDLFieldCritic, '__init__', forbidden)
    monkeypatch.setattr(torch.optim.Optimizer, '__init__', forbidden)
    for artifact in (raw, ema):
        with open_inference(artifact, operation='response') as inference:
            assert not any(hasattr(inference, name) for name in ('game', 'rows', 'critic', 'optimizer'))
            for strength in (-1., 0., .5, 1.):
                assert torch.isfinite(inference.response(strength=strength, **request)).all()
            if artifact is raw:
                assert torch.equal(inference.response(strength=-.5, **request), response)
            else:
                assert torch.equal(inference.response(strength=-.5, **request), ema_response)
                assert all(torch.equal(value, ema_weights[name]) for name, value in artifact['network'].items())
    assert torch.equal(rng, torch.get_rng_state())
    assert not torch.cuda.is_initialized()


def test_singleton_axis_binding_matches_shared_field_critic_and_derivatives():
    from particle_sliders.field_critic import FieldCritic
    torch.set_num_threads(1)
    old, new = FieldCritic(4, 3), Krea2HNDLFieldCritic(4, 3, (3, 4))
    with torch.no_grad():
        for name, layer in [('field0', old.field[0]), ('field2', old.field[2]),
                            ('condition', old.condition), ('score0', old.score[0]), ('score2', old.score[2])]:
            new.graph[name].weight.copy_(layer.weight)
            new.graph[name].bias.copy_(layer.bias)
    value = torch.randn(2, 4, 1, 3, 4, requires_grad=True)
    other = value.detach().clone().requires_grad_()
    condition = torch.randn(2, 3)
    expected, actual = old(value, condition), new(other, condition)
    torch.testing.assert_close(actual, expected)
    g0 = torch.autograd.grad(expected.sum(), value, create_graph=True)[0]
    g1 = torch.autograd.grad(actual.sum(), other, create_graph=True)[0]
    torch.testing.assert_close(g1, g0)
    torch.testing.assert_close(torch.autograd.grad(g1.square().sum(), new.graph['field0'].weight)[0],
                               torch.autograd.grad(g0.square().sum(), old.field[0].weight)[0])
    with pytest.raises(ValueError, match='native'):
        new(torch.zeros(2, 4, 3, 4), condition)


def test_actual_legacy_packet_refused_without_mutation():
    filename = os.environ.get('KREA2_LEGACY_PACKET')
    if not filename:
        pytest.skip('optional external core081 packet; full qualification must supply KREA2_LEGACY_PACKET')
    legacy = Path(filename)
    assert legacy.is_file(), 'KREA2_LEGACY_PACKET was set but does not name an existing packet'
    packet = torch.load(legacy, map_location='cpu', weights_only=True)
    with open_session(SessionConfig('krea2-hndl', 'krea2-hndl-cpu-v1', dummy=True, steps=4)) as session:
        before = session.checkpoint()['sha256']
        with pytest.raises(CheckpointError):
            session.restore(packet)
        assert session.checkpoint()['sha256'] == before


def test_source_admission_rejects_tamper_before_native_loading(monkeypatch):
    from krea2_hndl import runtime
    import particle_sliders.fibo_hndl as shared
    original = shared._sha
    monkeypatch.setattr(shared, '_sha', lambda p: '0' * 64 if str(p).endswith('hndl_networks.py') else original(p))
    with pytest.raises(ValueError, match='HyperGAN builder closure'):
        runtime.validate_runtime()


def test_valid_other_cohort_packet_refused_without_mutation():
    # Required even when the optional cross-environment legacy packet is absent.
    with open_session(SessionConfig('krea2-hndl', 'other-valid-cohort', dummy=True, steps=4)) as other:
        packet = other.checkpoint()
    with open_session(SessionConfig('krea2-hndl', 'krea2-hndl-cpu-v1', dummy=True, steps=4)) as session:
        before = session.checkpoint()['sha256']
        with pytest.raises(CheckpointError, match='cohort'):
            session.restore(packet)
        assert session.checkpoint()['sha256'] == before


def test_pg_source_bytes_checked_by_shared_admission(tmp_path, monkeypatch):
    import shutil
    import particlegan
    from krea2_hndl.runtime import validate_runtime
    source = Path(particlegan.__file__).parent
    copied = tmp_path / 'particlegan'
    shutil.copytree(source, copied, ignore=shutil.ignore_patterns('__pycache__'))
    (copied / 'policy.py').write_text((copied / 'policy.py').read_text() + '\n# source drift\n')
    monkeypatch.setattr(particlegan, '__file__', str(copied / '__init__.py'))
    with pytest.raises(RuntimeError, match='ParticleGAN sources differ'):
        validate_runtime()


@pytest.mark.parametrize('height,width', [(3, 4), (4, 3)])
def test_training_critic_has_fixed_spatial_contract(height, width):
    critic = Krea2HNDLFieldCritic(4, 3, (height, width))
    condition = torch.zeros(2, 3)
    assert critic(torch.zeros(2, 4, 1, height, width), condition).shape == (2,)
    before = deepcopy(critic.state_dict())
    # Even an equal-area H/W swap needs another resolved training graph/cohort.
    with pytest.raises(ValueError, match='fixed training H/W'):
        critic(torch.zeros(2, 4, 1, width, height), condition)
    assert all(torch.equal(value, before[name]) for name, value in critic.state_dict().items())


def test_rejected_geometry_preserves_complete_training_state():
    with open_session(SessionConfig('krea2-hndl', 'geometry-rejection', dummy=True,
                                    steps=4, ema_decay=.9)) as session:
        session.train(stop_after=2)  # Include initialized optimizers, EMA and sampler state.
        before = session.checkpoint()['sha256']
        _, channels, temporal, height, width = session.game.rows.response_shape
        with pytest.raises(ValueError, match='fixed training H/W'):
            session.game.critic(torch.zeros(2, channels, temporal, height, width + 1),
                                session.game.rows.conditions([0, 1]))
        assert session.checkpoint()['sha256'] == before


@pytest.mark.parametrize('change', ['license', 'extra_nested_data'])
def test_recursive_core_data_tamper_rejected_before_preparation(tmp_path, monkeypatch, change):
    import shutil
    import particle_sliders
    import krea2.synthetic as synthetic
    root = Path(particle_sliders.__file__).resolve().parent.parent
    for package in ('particle_sliders', 'concept_slider_core'):
        shutil.copytree(root / package, tmp_path / package, ignore=shutil.ignore_patterns('__pycache__'))
    if change == 'license':
        (tmp_path / 'particle_sliders/LICENSE').write_text('changed data bytes')
    else:
        path = tmp_path / 'particle_sliders/nested/unexpected.dat'
        path.parent.mkdir()
        path.write_bytes(b'unexpected recursive data')
    monkeypatch.setattr(particle_sliders, '__file__', str(tmp_path / 'particle_sliders/__init__.py'))
    def forbidden():
        raise AssertionError('source/data rejection must precede model preparation')
    monkeypatch.setattr(synthetic, 'staged_fixture', forbidden)
    before = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match='shared source differs'):
        open_session(SessionConfig('krea2-hndl', 'tampered-core', dummy=True, steps=4))
    assert torch.equal(before, torch.get_rng_state())


def test_old_modern_preview_packets_refused_without_rebinding(monkeypatch):
    filename = os.environ.get('KREA2_OLD_PREVIEW_DIR')
    if not filename:
        pytest.skip('qualification supplies actual old 4abc training, raw and EMA previews')
    directory = Path(filename)
    packet = torch.load(directory / 'training.pt', map_location='cpu', weights_only=True)
    with open_session(SessionConfig('krea2-hndl', 'krea2-hndl-cpu-v1', dummy=True,
                                    steps=4, ema_decay=.9)) as session:
        before = session.checkpoint()['sha256']
        with pytest.raises(CheckpointError):
            session.restore(packet)
        assert session.checkpoint()['sha256'] == before
    # Capture the fresh inference network without changing the registered factory sources.
    import krea2.native_inference as product
    original = product.prepare
    observed = []
    def capture(*args, **kwargs):
        binding = original(*args, **kwargs)
        observed.append((binding.network, deepcopy(binding.network.state_dict())))
        return binding
    monkeypatch.setattr(product, 'prepare', capture)
    for name in ('raw.pt', 'ema.pt'):
        before = torch.get_rng_state().clone()
        with pytest.raises(CheckpointError, match='source|protocol|model'):
            open_inference(directory / name, operation='response')
        assert torch.equal(before, torch.get_rng_state())
    assert len(observed) == 2
    for network, values in observed:
        assert all(torch.equal(value, values[name]) for name, value in network.state_dict().items())


def test_observation_failure_restores_state_and_allows_retry():
    from contextlib import contextmanager
    with open_session(SessionConfig('krea2-hndl', 'observation-retry', dummy=True,
                                    steps=4, ema_decay=.9)) as session:
        session.train(stop_after=2)
        before = session.checkpoint()['sha256']
        rng = torch.get_rng_state().clone()
        original = session.binding.observation
        attempts = 0
        @contextmanager
        def fail_once():
            nonlocal attempts
            with original():
                yield
                attempts += 1
                if attempts == 1:
                    torch.rand(3)
                    raise RuntimeError('observation cleanup fixture')
        session.binding.observation = fail_once
        with pytest.raises(RuntimeError, match='observation cleanup fixture'):
            session.evaluate()
        assert session.checkpoint()['sha256'] == before
        assert torch.equal(rng, torch.get_rng_state())
        assert torch.isfinite(session.evaluate()).all()
        assert session.checkpoint()['sha256'] == before


def test_close_failure_before_product_cleanup_allows_retry():
    session = open_session(SessionConfig('krea2-hndl', 'close-retry', dummy=True, steps=4))
    original = session.binding.close
    attempts = 0
    def fail_once():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError('before product cleanup fixture')
        original()
    session.binding.close = fail_once
    try:
        before = session.checkpoint()['sha256']
        with pytest.raises(RuntimeError, match='before product cleanup fixture'):
            session.close()
        assert not session.closed and session.checkpoint()['sha256'] == before
        session.close()
        session.close()
        assert session.closed and attempts == 2
    finally:
        session.binding.close = original
        session.close()


def test_lock_file_digest_tamper_rejected(monkeypatch):
    import json
    from krea2_hndl.runtime import validate_runtime
    original = Path.read_text
    def changed(path, *args, **kwargs):
        text = original(path, *args, **kwargs)
        if path.name == 'core.lock.json' and path.parent.name == 'krea2_hndl':
            lock = json.loads(text)
            lock['sources']['particle_sliders/LICENSE'] = '0' * 64
            return json.dumps(lock)
        return text
    monkeypatch.setattr(Path, 'read_text', changed)
    with pytest.raises(RuntimeError, match='shared source differs'):
        validate_runtime()
