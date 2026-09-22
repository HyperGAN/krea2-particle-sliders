"""Wrappers fail closed and never call the stock Raw trainer."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "scripts" / "train_krea2.py"
INFER = ROOT / "scripts" / "infer_krea2.py"


def _run(script: Path, args: list[str], root: str | None, extra_env: dict | None = None):
    env = os.environ.copy()
    env.pop("HF_HUB_OFFLINE", None)
    if root is None:
        env.pop("PARTICLE_SLIDERS_ROOT", None)
    else:
        env["PARTICLE_SLIDERS_ROOT"] = root
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_train_and_infer_fail_without_backend(tmp_path: Path):
    for script in (TRAIN, INFER):
        proc = _run(script, ["--dummy"], str(tmp_path))
        assert proc.returncode == 2, proc.stderr
        err = proc.stderr
        assert "jimmycarter/krea2-turbo-bbox" in err
        assert "epoch-14-step-73184/transformer" in err
        assert "train_lora_krea.py" in err
        assert "Nothing was downloaded." in err
        assert "krea/Krea-2-Raw" in err


def test_stock_trainer_is_not_a_fallback(tmp_path: Path):
    backend = tmp_path / "conceptmod" / "textsliders"
    backend.mkdir(parents=True)
    stock = backend / "train_lora_krea.py"
    stock.write_text(
        "import sys\nprint('STOCK_TRAINER_RAN')\nsys.exit(0)\n",
        encoding="utf-8",
    )
    proc = _run(TRAIN, ["--dummy"], str(tmp_path))
    assert proc.returncode == 2
    assert "STOCK_TRAINER_RAN" not in proc.stdout
    assert "STOCK_TRAINER_RAN" not in proc.stderr


def test_train_forwards_locked_defaults(tmp_path: Path):
    backend = tmp_path / "conceptmod" / "textsliders"
    backend.mkdir(parents=True)
    script = backend / "train_lora_krea2.py"
    script.write_text(
        "import os, sys\n"
        "print('OFFLINE=' + os.environ.get('HF_HUB_OFFLINE', ''))\n"
        "print('\\n'.join(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    proc = _run(
        TRAIN,
        ["--dummy", "--prompts_file", "configs/krea2/prompts-lighting.yaml"],
        str(tmp_path),
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "OFFLINE=1" in out
    assert "--model_id\njimmycarter/krea2-turbo-bbox" in out
    assert "--transformer_subfolder\nepoch-14-step-73184/transformer" in out
    assert "--pipeline_id\nkrea/Krea-2-Raw" in out
    assert "--sample_steps\n8" in out
    assert "--sample_guidance\n0.0" in out
    assert "--mu\n1.15" in out
    assert "--hold_weight\n0.1" in out
    assert "--dummy" in out
    assert "prompts-lighting.yaml" in out
    assert out.count("--prompts_file") == 1
    assert "config-expression.yaml" not in out


def test_allow_hub_is_not_forced_offline(tmp_path: Path):
    backend = tmp_path / "conceptmod" / "textsliders"
    backend.mkdir(parents=True)
    script = backend / "train_lora_krea2.py"
    script.write_text(
        "import os\nprint('OFFLINE=' + os.environ.get('HF_HUB_OFFLINE', 'unset'))\n",
        encoding="utf-8",
    )
    proc = _run(TRAIN, ["--allow_hub", "--dummy"], str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    assert "OFFLINE=unset" in proc.stdout


def test_infer_forwards_when_present(tmp_path: Path):
    backend = tmp_path / "conceptmod" / "textsliders"
    backend.mkdir(parents=True)
    script = backend / "infer_lora_krea2.py"
    script.write_text(
        "import sys\nprint('\\n'.join(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    proc = _run(INFER, ["--load_lora", "models/missing"], str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    assert "--model_id\njimmycarter/krea2-turbo-bbox" in proc.stdout
    assert "--mu\n1.15" in proc.stdout
    assert "--load_lora\nmodels/missing" in proc.stdout
    assert "--hold_weight" not in proc.stdout
