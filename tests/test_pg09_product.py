"""Actual tiny Krea2 + PEFT through the product's installed shared GAN binding.

No Hub downloads, CUDA initialization, mock GAN losses or vendored core modules.
"""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from diffusers import Krea2Pipeline, Krea2Transformer2DModel, FlowMatchEulerDiscreteScheduler
from peft import LoraConfig, get_peft_model

from krea2.cuda_train import CudaUNI, run_contract, validate_args
from krea2.pg09 import build_game
from krea2.train import parse_args


def fixture(preset='bispf_optimized'):
    torch.set_num_threads(1)
    torch.manual_seed(23)
    model = Krea2Transformer2DModel(in_channels=16, num_layers=1, attention_head_dim=16,
        num_attention_heads=2, num_key_value_heads=1, intermediate_size=64,
        timestep_embed_dim=32, text_hidden_dim=16, num_text_layers=2,
        text_num_attention_heads=2, text_num_key_value_heads=1, text_intermediate_size=32,
        num_layerwise_text_blocks=1, num_refiner_text_blocks=1, axes_dims_rope=(4, 6, 6))
    model.requires_grad_(False)
    model = get_peft_model(model, LoraConfig(r=2, lora_alpha=3,
        target_modules=['to_q', 'to_k', 'to_v', 'to_out.0']))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.get_base_model().enable_gradient_checkpointing()
    pipe = object.__new__(Krea2Pipeline)
    pipe.patch_size, pipe.vae_scale_factor, pipe.transformer = 2, 8, model
    pipe.scheduler = FlowMatchEulerDiscreteScheduler()
    native = CudaUNI.__new__(CudaUNI)
    native.pipe, native.model, native.device, native.dtype = pipe, model, torch.device('cpu'), torch.float32
    native.args = SimpleNamespace(resolution=32)
    embeds = torch.randn(1, 3, 2, 16)
    mask = torch.tensor([[True, True, False]])
    native.embeddings = {'neutral': (embeds, mask), 'positive': (embeds + .2, mask),
                         'hold': (embeds - .2, mask)}
    native.position_ids = {}
    native.params = [p for p in model.parameters() if p.requires_grad]
    cache = dict(edit=[], preserve=[], key={'fixture': 'real-tiny-Krea2-cpu'})
    for holding, pool in ((False, 'edit'), (True, 'preserve')):
        for i in range(2):
            neutral, positive = ('hold', 'hold') if holding else ('neutral', 'positive')
            z = torch.randn(1, 4, 16)
            t = torch.tensor(700. - 300. * i)
            with torch.no_grad(), native.scale(0):
                baseline = native.forward(neutral, z, t).cpu()
                teacher = native.forward(positive, z, t).cpu()
            cache[pool].append(dict(prompt=neutral, positive=positive, z=z, t=t,
                                    baseline=baseline, teacher=teacher))
    game = build_game(native, cache, cache_sha256='fixture-generated-from-frozen-native-model',
        model_identity={'qualification': 'tiny-native-CPU-only'}, total_steps=5,
        batch_size=2, seed=37, preset=preset)
    return native, cache, game


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize('preset', ['bispf', 'bispf_optimized'])
def test_native_shared_gan_updates_and_exact_resume(preset, tmp_path):
    native, cache, full = fixture(preset)
    base = {n: p.detach().clone() for n, p in native.model.named_parameters() if not p.requires_grad}
    adapter = [p.detach().clone() for p in full.network.parameters()]
    row = full.rows[0]
    with torch.no_grad():
        generated = full.backend.generated(row)
    assert generated.shape == (1, 4, 1, 4, 4)
    assert torch.equal(generated, row.neutral)
    assert full.rows.edit_slots == 4 * full.rows.preserve_slots
    assert full._signature['generator_arm'] == 'adapter_gan'
    assert full._signature['pg_version'] == '0.9.1'
    assert full._signature['objective'] == 'g_adv+public_prior;d_adv+public_penalty'
    for _ in range(5):
        stats = full.step()
    assert all(torch.isfinite(torch.tensor(stats[k])) for k in ('g_adv', 'd_adv', 'd_pen'))
    assert any(not torch.equal(a, b) for a, b in zip(adapter, full.network.parameters()))
    for n, p in native.model.named_parameters():
        if n in base:
            assert torch.equal(p, base[n])
    _, _, split = fixture(preset)
    split.step(); split.step()
    saved = tmp_path / 'state.pt'
    torch.save(split.state_dict(), saved)
    _, _, resumed = fixture(preset)
    resumed.load_state_dict(torch.load(saved, weights_only=True))
    for _ in range(3):
        resumed.step()
    equal(full.state_dict(), resumed.state_dict())
    _, _, loaded = fixture(preset)
    loaded.load_export_state_dict(full.export_state_dict())
    assert torch.equal(full.evaluate(0), loaded.evaluate(0))
    with pytest.raises(RuntimeError, match='inference export'):
        loaded.step()
    assert not torch.cuda.is_initialized()


def test_cache_conditioning_and_checkpoint_drift_rejected():
    _, cache, game = fixture()
    changed = deepcopy(game.state_dict())
    changed['config']['backend']['model_identity']['qualification'] = 'different base'
    with pytest.raises(ValueError, match='mismatch'):
        game.load_state_dict(changed)
    with pytest.raises(ValueError, match='mismatch'):
        game.load_state_dict({'step': 2, 'optimizer': {}, 'rng': ()})
    a, b = game.rows[0], game.rows[1]
    assert not torch.equal(game.rows.conditions([0]), game.rows.conditions([1]))
    for field in ('embeds', 'mask', 'position_ids'):
        assert field in a.row.neutral
    assert a.row.guidance == 1 and a.row.scale == 1


def test_contract_binds_model_skeleton_schedule_and_sources(tmp_path):
    args = parse_args(['--live'])
    first = run_contract(args, [])
    for field, value in [('skeleton_revision', 'different'), ('revision', 'different'),
                         ('mu', 1.2), ('sample_steps', 9)]:
        changed = deepcopy(args)
        setattr(changed, field, value)
        assert run_contract(changed, []) != first
    model = tmp_path / 'transformer'
    model.mkdir()
    (model / 'config.json').write_text('{}')
    args.transformer = str(model)
    before = run_contract(args, [])
    (model / 'config.json').write_text('{"changed":true}')
    assert run_contract(args, []) != before
    args.transformer = None
    args.model_id = str(model)
    before = run_contract(args, [])
    (model / 'config.json').write_text('{"changed_again":true}')
    assert run_contract(args, []) != before


@pytest.mark.parametrize('overrides', [dict(resume='old', load_te_lora='sample'),
    dict(resolution=0), dict(revision='main'), dict(sample_resolution=765), dict(gan_batch_size=1), dict(lr=float('nan')), dict(hold_weight=.5)])
def test_bad_live_arguments_rejected_before_model_load(overrides):
    args = parse_args(['--live'])
    for key, value in overrides.items():
        setattr(args, key, value)
    with pytest.raises(ValueError):
        validate_args(args)


def test_full_live_dispatch_crash_and_new_directory_resume_on_cpu(tmp_path, monkeypatch):
    """Replace only resource loading/rendering; execute real native fields and GAN updates."""
    import json
    from PIL import Image
    from diffusers import FlowMatchEulerDiscreteScheduler
    import krea2.cuda_train as live
    from krea2.train import train

    from krea2.pg09 import build_game as actual_builder
    import krea2.pg09 as bridge
    built = []
    def cpu_native(args, texts, out):
        native, _, _ = fixture()
        native.args, native.out = args, out
        object.__setattr__(native.pipe, 'scheduler', FlowMatchEulerDiscreteScheduler())
        native.pipe.vae = torch.nn.Linear(2, 2).requires_grad_(False)
        embedding = torch.randn(1, 3, 2, 16)
        mask = torch.tensor([[True, True, False]])
        native.embeddings = {text: (embedding + (.2 if text == 'positive' else 0.), mask)
                             for text in texts}
        native.generate = lambda *a: Image.new('RGB', (32, 32), 'white')
        return native
    def crashing_builder(*args, **kwargs):
        game = actual_builder(*args, **kwargs)
        built.append(game)
        step = game.step
        if len(built) == 1:
            def interrupted():
                if game.completed_steps == 3:
                    raise RuntimeError('simulated interruption after saved checkpoint')
                return step()
            game.step = interrupted
        return game
    monkeypatch.setattr(live, 'CudaUNI', cpu_native)
    monkeypatch.setattr(bridge, 'build_game', crashing_builder)
    import sys
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: sys._getframe(1).f_code.co_name == 'train_cuda')
    monkeypatch.setattr(torch.cuda, 'set_device', lambda *a: None)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda *a: 'CPU fixture; no GPU initialized')
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda *a: 0)
    prompts = tmp_path / 'prompts.yaml'
    prompts.write_text('rows:\n  - target: neutral\n    neutral: neutral\n    positive: positive\n    action: enhance\n')
    original, continued = tmp_path / 'original', tmp_path / 'continued'
    command = ['--live', '--prompts_file', str(prompts), '--save_dir', str(original),
               '--steps', '5', '--rank', '2', '--resolution', '32', '--sample_resolution', '32',
               '--final_resolution', '0', '--save_every', '2', '--cache_seeds', '1']
    with pytest.raises(RuntimeError, match='simulated interruption'):
        train(parse_args(command))
    checkpoint = original / 'checkpoint-0002'
    log = (original / 'train.jsonl').read_bytes()
    manifest = (original / 'run.json').read_bytes()
    assert [json.loads(line)['step'] for line in log.splitlines()] == [1, 2, 3]
    args = parse_args(command + ['--resume', str(checkpoint)])
    with pytest.raises(ValueError, match='new --save_dir'):
        train(args)
    assert (original / 'train.jsonl').read_bytes() == log
    assert (original / 'run.json').read_bytes() == manifest
    args.save_dir = str(continued)
    status = train(args)
    assert json.loads(status.read_text())['step'] == 5
    assert [json.loads(line)['step'] for line in (continued / 'train.jsonl').read_text().splitlines()] == [1, 2, 3, 4, 5]
    assert (original / 'train.jsonl').read_bytes() == log
    assert json.loads((continued / 'run.json').read_text())['continued_from'] == str(checkpoint)
    assert json.loads((continued / 'checkpoint-0005/adapter_identity.json').read_text())['completed_steps'] == 5
    # Fresh same-contract native run gives exactly the resumed adapter/optimizer/sampler bytes.
    monkeypatch.setattr(bridge, 'build_game', actual_builder)
    args.resume, args.save_dir = None, str(tmp_path / 'full')
    train(args)
    full = torch.load(tmp_path / 'full/checkpoint-0005/training_state.pt', weights_only=True)
    resumed = torch.load(continued / 'checkpoint-0005/training_state.pt', weights_only=True)
    equal(full, resumed)
    assert not torch.cuda.is_initialized()


def test_shared_features_and_frozen_model_content_identity():
    from particle_sliders.pg09_images import image_condition_features
    native, _, game = fixture()
    actual = image_condition_features([game.rows[i] for i in range(len(game.rows))])
    assert torch.equal(actual, game.rows.conditions(range(len(game.rows))))
    native.pipe.vae = torch.nn.Linear(2, 2).requires_grad_(False)
    before = native.frozen_identity()
    with torch.no_grad():
        next(p for p in native.model.parameters() if not p.requires_grad).add_(.01)
    assert native.frozen_identity() != before


def test_new_adapter_identity_rejects_wrong_base_and_modified_files(tmp_path):
    import json
    from krea2.cuda_train import validate_adapter_identity
    from krea2.pg09 import file_digest
    adapter = tmp_path / 'adapter'
    adapter.mkdir()
    weights = adapter / 'adapter_model.safetensors'
    weights.write_bytes(b'fixture only; not a native adapter')
    current = {'model_id': 'pinned-model', 'revision': 'a' * 40, 'frozen_model_sha256': 'verified-frozen-bytes'}
    packet = dict(schema='krea2-gan-adapter-v1', model_identity=current, completed_steps=3,
                  files={weights.name: file_digest(weights)})
    (adapter / 'adapter_identity.json').write_text(json.dumps(packet))
    assert validate_adapter_identity(adapter, current) == 3
    with pytest.raises(ValueError, match='frozen model differs'):
        validate_adapter_identity(adapter, dict(current, revision='b' * 40))
    weights.write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum differs'):
        validate_adapter_identity(adapter, current)
    assert validate_adapter_identity(tmp_path / 'legacy-folder', current) is None
