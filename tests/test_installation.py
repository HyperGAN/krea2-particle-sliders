"""Installer/source-cohort compatibility guards; no model allocation."""
import json
from pathlib import Path
import subprocess
import sys
import tomllib
from urllib.parse import unquote, urlparse
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_original_dependency_only_installer_preserved():
    original = subprocess.check_output(['git', 'show',
        'afc226f4710db5e4fe9379f22606cd5ccb8ee917:requirements.txt'], cwd=ROOT)
    assert (ROOT / 'requirements-legacy.txt').read_bytes() == original


def test_canonical_lock_and_immutable_legacy_profile():
    lock = json.loads((ROOT / 'krea2/shared.lock.json').read_text())
    default = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    candidate = tomllib.loads((ROOT / 'profiles/hypergan-hndl/pyproject.toml').read_text())
    assert default['project']['version'] == '0.2.1'
    assert lock['schema'] == 'krea2-shared-api-v3'
    assert lock['version'] == '0.13.0' and lock['particlegan'] == '0.9.5'
    assert lock['commit'] == '3dd672a7f9b8c169daef90fd63e4bb42b9ddc423'
    assert lock['commit'] in default['project']['dependencies'][0]
    assert 'particle_sliders/LICENSE' in lock['sources']
    assert any(name.endswith('.yaml') for name in lock['sources'])
    assert '91ae74b12c123a7241142f134cc4501c5288583a' in (ROOT / 'profiles/core081/requirements.txt').read_text()
    assert '3dd672a7f9b8c169daef90fd63e4bb42b9ddc423' in candidate['project']['dependencies'][0]
    assert 'hypergan-hndl' in candidate['project']['dependencies'][0]


def test_inspection_checks_installed_distribution(tmp_path):
    result = subprocess.run([sys.executable, '-I', str(ROOT / 'scripts/check_install.py')],
        cwd=tmp_path, text=True, capture_output=True, check=True)
    report = json.loads(result.stdout)
    assert report['status'] == 'passed'
    assert report['shared']['version'] == '0.13.0'
    assert report['shared']['particlegan'] == '0.9.5'
    assert report['cuda_initialized'] is False
    origin = Path(report['origins']['krea2']).resolve()
    if report['installation']['editable']:
        source = Path(unquote(urlparse(report['installation']['direct_url']).path)).resolve()
        assert origin.is_relative_to(source / 'krea2')
    else:
        assert not origin.is_relative_to(ROOT / 'krea2')


def test_unknown_native_profile_rejected_before_loading():
    from krea2.cuda_train import CudaUNI
    with pytest.raises(ValueError, match='unknown Krea2 runtime profile'):
        CudaUNI(None, [], None, runtime_profile='unqualified')


def test_canonical_admission_checks_data_and_official_pg(monkeypatch):
    import particle_sliders
    import particle_sliders.pg09 as pg
    from krea2.shared import validate_shared_runtime
    data = next(Path(particle_sliders.__file__).parent.rglob('*.yaml')).resolve()
    original = Path.read_bytes
    with monkeypatch.context() as changed:
        changed.setattr(Path, 'read_bytes', lambda path: b'drift' if path.resolve() == data else original(path))
        with pytest.raises(RuntimeError, match='shared API source differs'):
            validate_shared_runtime()
    def reject(*args, **kwargs):
        raise RuntimeError('official PG source drift')
    monkeypatch.setattr(pg, 'validate_pg09_domain', reject)
    with pytest.raises(RuntimeError, match='official PG source drift'):
        validate_shared_runtime()


def test_supported_discovery_keeps_subpackages_and_excludes_candidate(tmp_path):
    from setuptools import find_namespace_packages
    config = tomllib.loads((ROOT / 'pyproject.toml').read_text())['tool']['setuptools']['packages']['find']
    for name in ('krea2/subpackage', 'krea2_hndl'):
        path = tmp_path / name
        path.mkdir(parents=True)
        (path / '__init__.py').write_text('')
    packages = find_namespace_packages(str(tmp_path), include=config['include'])
    assert set(packages) == {'krea2', 'krea2.subpackage'}
    actual = find_namespace_packages(str(ROOT), include=config['include'])
    assert 'krea2' in actual and 'krea2_hndl' not in actual


def test_candidate_installer_refuses_existing_product(tmp_path):
    result = subprocess.run([sys.executable, '-I', str(ROOT / 'scripts/install_hndl_candidate.py'),
                             '--check-only'], cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 2
    assert 'already installed' in result.stderr


def test_candidate_installer_refuses_non_venv_with_intended_error(tmp_path):
    # Execute the actual base interpreter rather than failing in the current venv.
    base_python = getattr(sys, '_base_executable', sys.executable)
    result = subprocess.run([base_python, '-I', str(ROOT / 'scripts/install_hndl_candidate.py'),
                             '--check-only'], cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 2
    assert 'error: create a fresh virtual environment for the candidate' in result.stderr
    assert 'already installed' not in result.stderr


def test_candidate_builder_rejects_wrong_core_artifact(tmp_path):
    wheel = tmp_path / 'wrong.whl'
    wheel.write_bytes(b'not the recorded core wheel')
    output = tmp_path / 'output'
    result = subprocess.run([sys.executable, '-I', str(ROOT / 'scripts/build_hndl_candidate.py'),
        '--output', str(output), '--core-wheel', str(wheel)], text=True, capture_output=True)
    assert result.returncode == 2
    assert 'reference wheel SHA256' in result.stderr
    assert not output.exists()
