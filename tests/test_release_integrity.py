"""Release loading and recovery gates; fake pipelines never stage a GPU model."""

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from release_tools import distill
from scripts import finish_final_boss_samples as recovery


def release_runtime(tmp_path):
    path = tmp_path / 'adapter.safetensors'
    path.write_bytes(b'reviewed adapter fixture')
    runtime = distill.Runtime.__new__(distill.Runtime)
    runtime.args = SimpleNamespace()
    runtime.model = Mock()
    runtime.pipe = Mock()
    runtime.active = 'previous'
    runtime._release_frozen_identity = 'reviewed frozen base'
    runtime.frozen_identity = Mock(return_value=runtime._release_frozen_identity)
    return runtime, path, hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('changed', [False, True])
def test_wrong_or_altered_adapter_rejected_before_pipeline_mutation(tmp_path, changed):
    runtime, path, expected = release_runtime(tmp_path)
    if changed:
        path.write_bytes(b'changed after catalog creation')
    else:
        expected = '0' * 64
    with pytest.raises(ValueError, match='checksum'):
        runtime.load('candidate', path, expected_sha256=expected)
    runtime.pipe.load_lora_weights.assert_not_called()
    runtime.pipe.set_adapters.assert_not_called()
    assert runtime.active == 'previous'


def test_adapter_load_requires_expected_artifact_hash(tmp_path):
    runtime, path, _ = release_runtime(tmp_path)
    with pytest.raises(TypeError, match='expected_sha256'):
        runtime.load('candidate', path)
    runtime.pipe.load_lora_weights.assert_not_called()


def test_new_gan_sidecar_revalidated_before_loading(tmp_path, monkeypatch):
    runtime, path, expected = release_runtime(tmp_path)
    (tmp_path / 'adapter_identity.json').write_text('{}')
    contract = {'model_id': 'frozen bbox fixture', 'rank': 16}
    monkeypatch.setattr(distill, 'run_contract', Mock(return_value=contract))
    validated = Mock()
    monkeypatch.setattr(distill, 'validate_adapter_identity', validated)
    events = []
    validated.side_effect = lambda *args: events.append('validated')
    runtime.pipe.load_lora_weights.side_effect = lambda *args, **kwargs: events.append('loaded')

    runtime.load('candidate', path, expected_sha256=expected)

    validated.assert_called_once_with(tmp_path, dict(
        contract, frozen_model_sha256=runtime._release_frozen_identity))
    assert events == ['validated', 'loaded']
    runtime.pipe.load_lora_weights.assert_called_once_with(str(path), adapter_name='candidate')
    runtime.pipe.set_adapters.assert_called_once_with('candidate', adapter_weights=1.)
    assert runtime.active == 'candidate'


def test_invalid_new_gan_sidecar_does_not_mutate_pipeline(tmp_path, monkeypatch):
    runtime, path, expected = release_runtime(tmp_path)
    (tmp_path / 'adapter_identity.json').write_text('{}')
    monkeypatch.setattr(distill, 'run_contract', Mock(return_value={}))
    monkeypatch.setattr(distill, 'validate_adapter_identity', Mock(side_effect=ValueError('sidecar mismatch')))
    with pytest.raises(ValueError, match='sidecar mismatch'):
        runtime.load('candidate', path, expected_sha256=expected)
    runtime.pipe.load_lora_weights.assert_not_called()
    runtime.pipe.set_adapters.assert_not_called()
    assert runtime.active == 'previous'


@pytest.mark.parametrize('changed_before', [True, False])
def test_adapter_swaps_keep_verified_frozen_base(tmp_path, changed_before):
    runtime, path, expected = release_runtime(tmp_path)
    runtime.frozen_identity.side_effect = (
        ['changed base'] if changed_before else ['reviewed frozen base', 'changed base'])
    with pytest.raises(ValueError, match='Frozen base changed'):
        runtime.load('candidate', path, expected_sha256=expected)
    if changed_before:
        runtime.pipe.load_lora_weights.assert_not_called()
    else:
        runtime.pipe.load_lora_weights.assert_called_once()
    runtime.pipe.set_adapters.assert_not_called()
    assert runtime.active == 'previous'


@pytest.mark.parametrize('render_size', [8, 16])
def test_recovery_records_roundtrip_mismatch_then_stops_before_showcase(tmp_path, monkeypatch, render_size):
    out = tmp_path / 'outputs/final-boss-krea2-bbox'
    samples = out / 'samples'
    samples.mkdir(parents=True)
    (tmp_path / 'krea2').mkdir()
    for name in ('attention.py', 'cuda_train.py'):
        (tmp_path / 'krea2' / name).write_text('# source fixture\n')
    args = dict(device=0, sample_seed=9, sample_resolution=8, final_resolution=16)
    (out / 'run.json').write_text(json.dumps(args))
    (out / 'status.json').write_text(json.dumps(dict(phase='failed', error='old failure')))
    (out / 'prompts.json').write_text(json.dumps([{'neutral': f'neutral {i}'} for i in range(5)]))
    for label in ('knight', 'cave-warrior', 'heldout-bridge', 'fruit-control'):
        for scale in (0., .5, 1.):
            Image.new('RGB', (8, 8)).save(samples / f'{label}-scale-{scale:g}-seed-9.png')
    original_metadata = b'{"preserved": true}\n'
    (samples / 'metadata.json').write_bytes(original_metadata)
    generated = Mock(return_value=Image.new('RGB', (render_size, render_size), color=(1, 1, 1)))
    monkeypatch.setattr(recovery, 'ROOT', tmp_path)
    monkeypatch.setattr(recovery, 'CudaUNI', Mock(return_value=SimpleNamespace(generate=generated)))
    monkeypatch.setattr(recovery.torch, 'set_num_threads', lambda count: None)
    monkeypatch.setattr(recovery.torch.cuda, 'set_device', lambda device: None)

    with pytest.raises(ValueError, match='roundtrip differs'):
        recovery.main()

    report = json.loads((out / 'showcase_recovery/roundtrip.json').read_text())
    assert report['pixel_identical'] is False
    assert report['reference_shape'] == [8, 8, 3]
    assert report['reloaded_shape'] == [render_size, render_size, 3]
    assert report['pixel_mae'] == (1. if render_size == 8 else None)
    generated.assert_called_once_with(recovery.HELDOUT, 1., 9, 8)
    assert json.loads((out / 'status.json').read_text())['phase'] == 'failed'
    assert (samples / 'metadata.json').read_bytes() == original_metadata
    assert not list(samples.glob('showcase-*.png'))


def test_real_tiny_adapter_swap_keeps_frozen_bytes_and_nonunit_alpha(tmp_path):
    import torch
    from diffusers import Krea2Pipeline, Krea2Transformer2DModel
    from peft import LoraConfig, get_peft_model
    from release_tools.lora import export_lora, digest

    torch.manual_seed(11)
    base = Krea2Transformer2DModel(in_channels=4, num_layers=1, num_attention_heads=2,
        attention_head_dim=16, num_key_value_heads=1, intermediate_size=64,
        text_hidden_dim=16, text_intermediate_size=32, text_num_attention_heads=1,
        text_num_key_value_heads=1, num_text_layers=2, num_layerwise_text_blocks=1,
        num_refiner_text_blocks=1, axes_dims_rope=(4, 6, 6)).requires_grad_(False)
    wrapped = get_peft_model(base, LoraConfig(r=2, lora_alpha=3, target_modules=['to_q']))
    runtime = distill.Runtime.__new__(distill.Runtime)
    runtime.model = wrapped
    runtime.pipe = SimpleNamespace(vae=torch.nn.Linear(2, 2).requires_grad_(False))
    initial = runtime.frozen_identity()
    runtime.model = wrapped.unload()
    assert runtime.frozen_identity() == initial
    runtime._release_frozen_identity = initial
    runtime.active = None
    def load(path, adapter_name):
        state, metadata = Krea2Pipeline.lora_state_dict(path, return_lora_metadata=True)
        Krea2Pipeline.load_lora_into_transformer(state, transformer=runtime.model,
            adapter_name=adapter_name, metadata=metadata)
    runtime.pipe.load_lora_weights = load
    runtime.pipe.set_adapters = lambda name, adapter_weights: runtime.model.set_adapters(name, weights=adapter_weights)
    name = 'transformer_blocks.0.attn.to_q'
    down, up = torch.randn(2, 32), torch.randn(32, 2)
    exported = export_lora(tmp_path, 'weights', 'swap-fixture', {name: (down, up)}, 3., 'parent', 'CPU fixture')
    path = tmp_path / exported['files']['native']
    runtime.load('fixture', path, expected_sha256=exported['sha256']['native'])
    assert runtime.frozen_identity() == initial
    layer = runtime.model.transformer_blocks[0].attn.to_q
    assert torch.allclose(layer.get_delta_weight('fixture'), 1.5 * (up @ down))
    with runtime.scale(-.5):
        assert layer.scaling['fixture'] == pytest.approx(-.75)
        with runtime.scale(0):
            assert torch.count_nonzero(layer.get_delta_weight('fixture')) == 0
        assert layer.scaling['fixture'] == pytest.approx(-.75)
    assert layer.scaling['fixture'] == pytest.approx(1.5)
    assert runtime.frozen_identity() == initial
    assert not torch.cuda.is_initialized()
