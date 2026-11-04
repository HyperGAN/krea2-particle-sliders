"""LR admission and exact shared recipe wiring; no pretrained/GPU work."""
import pytest
from krea2.learning_rates import recipe_overrides

@pytest.mark.parametrize('key', ['lr', 'd_lr_mult', 'prior_lr_mult'])
@pytest.mark.parametrize('value', [True, False, 0, -1, float('nan'), float('inf'), '0.001'])
def test_invalid_rates(key, value):
    with pytest.raises(ValueError):
        recipe_overrides(**{key: value})


def test_optional_rates_keep_preset_defaults():
    assert recipe_overrides() == {}
    assert recipe_overrides(lr=2e-5, d_lr_mult=3) == {'lr': 2e-5, 'd_lr_mult': 3.0}


def test_adapter_only_rejects_prior_rate():
    with pytest.raises(ValueError, match='adapter-only'):
        recipe_overrides(prior_lr_mult=2)


def test_cli_default_matches_game_and_common_api():
    from krea2.train import build_parser
    from inspect import signature
    from krea2.pg09 import build_game
    assert build_parser().get_default('lr') == signature(build_game).parameters['lr'].default == 5e-5


def test_shared_game_rates_and_exact_resume(tmp_path):
    from particle_sliders.lifecycle import SessionConfig
    from krea2.lifecycle import BACKEND
    config = SessionConfig('krea2', 'lr-standardization-v1', dummy=True, steps=4,
                           lr=2e-5, d_lr_mult=3)
    with BACKEND.open(config) as full:
        assert full.game.recipe.lr == 2e-5 and full.game.recipe.d_lr_mult == 3
        assert full.game.prior is None
        assert full.game.base_lrs[0] == [2e-5]
        assert all(rate == pytest.approx(6e-5) for rate in full.game.base_lrs[1])
        full.train()
        expected = full.checkpoint()['sha256']
    with BACKEND.open(config) as split:
        split.train(stop_after=2)
        packet = split.checkpoint(tmp_path / 'training.pt')
    with BACKEND.open(config) as resumed:
        resumed.restore(packet)
        resumed.train()
        assert resumed.checkpoint()['sha256'] == expected
    from particle_sliders.lifecycle import CheckpointError
    with pytest.raises(CheckpointError):
        with BACKEND.open(SessionConfig('krea2', 'lr-standardization-v1', dummy=True, steps=4,
                                       lr=3e-5, d_lr_mult=3)) as changed:
            changed.restore(packet)


def test_closed_hndl_borrowed_runtime_bytes_are_preserved(tmp_path):
    import hashlib
    import importlib.util
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('lr_archive_builder', root / 'scripts/build_hndl_candidate.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    manifest = json.loads((root/'cohorts/pre-lr-hndl/manifest.json').read_text())
    for relative in manifest['files']:
        (tmp_path/relative).parent.mkdir(parents=True, exist_ok=True)
    builder.restore_pre_lr_runtime(root,tmp_path)
    for relative, expected in manifest['files'].items():
        assert hashlib.sha256((tmp_path/relative).read_bytes()).hexdigest() == expected
    assert not list(tmp_path.rglob('learning_rates.py'))
