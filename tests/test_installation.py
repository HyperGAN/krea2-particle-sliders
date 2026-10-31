"""Installer/source-cohort compatibility guards; no model allocation."""
import json
from pathlib import Path
import subprocess
import sys
import tomllib
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
    assert str(ROOT / 'krea2') not in report['origins']['krea2']


def test_unknown_native_profile_rejected_before_loading():
    from krea2.cuda_train import CudaUNI
    with pytest.raises(ValueError, match='unknown Krea2 runtime profile'):
        CudaUNI(None, [], None, runtime_profile='unqualified')
