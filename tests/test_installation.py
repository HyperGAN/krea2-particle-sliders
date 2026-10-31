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


def test_supported_lock_and_legacy_profile_remain_exact():
    original = subprocess.check_output(['git', 'show',
        'afc226f4710db5e4fe9379f22606cd5ccb8ee917:krea2/shared.lock.json'], cwd=ROOT)
    assert (ROOT / 'krea2/shared.lock.json').read_bytes() == original
    default = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    candidate = tomllib.loads((ROOT / 'profiles/hypergan-hndl/pyproject.toml').read_text())
    assert default['project']['version'] == '0.2.0'
    assert '32f6830f6402e85aa77b6631db64fcbbb2131f6b' in default['project']['dependencies'][0]
    assert '4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231' in candidate['project']['dependencies'][0]
    assert 'hypergan-hndl' in candidate['project']['dependencies'][0]


def test_inspection_checks_installed_distribution(tmp_path):
    result = subprocess.run([sys.executable, '-I', str(ROOT / 'scripts/check_install.py')],
        cwd=tmp_path, text=True, capture_output=True, check=True)
    report = json.loads(result.stdout)
    assert report['status'] == 'passed'
    assert report['shared']['version'] == '0.8.1'
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
