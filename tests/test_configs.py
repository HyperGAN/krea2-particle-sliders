"""Parse starter cards. No Hub access."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

from krea2.defaults import (
    CONTROL_PROMPT,
    COMFY_CFG,
    COMFY_FILENAME,
    COMFY_ORG_REPO,
    COMFY_TEXT_ENCODER,
    COMFY_VAE,
    MODEL_ID,
    PIPELINE_ID,
    RESOLUTION,
    TRANSFORMER_SUBFOLDER,
    TURBO_GUIDANCE,
    TURBO_MU,
    TURBO_STEPS,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "krea2"
CARDS = ("expression", "lighting", "panel-clarity", "smile")
UNUSED = ("male", "female")


def _load(path: Path):
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_model_lock_matches_defaults():
    lock = json.loads((CONFIGS / "model.lock.json").read_text(encoding="utf-8"))
    assert lock["model_id"] == MODEL_ID
    assert lock["transformer_subfolder"] == TRANSFORMER_SUBFOLDER
    assert lock["pipeline_id"] == PIPELINE_ID
    assert lock["comfy_filename"] == COMFY_FILENAME
    assert lock["comfy_text_encoder"] == COMFY_TEXT_ENCODER
    assert lock["comfy_vae"] == COMFY_VAE
    assert lock["comfy_org_repo"] == COMFY_ORG_REPO
    assert lock["turbo_steps"] == TURBO_STEPS
    assert lock["diffusers_guidance_scale"] == TURBO_GUIDANCE
    assert lock["mu"] == TURBO_MU
    assert lock["comfy_cfg"] == COMFY_CFG
    assert lock["weights_in_repo"] is False
    assert lock["variant_label"].startswith("epoch-14-step-73184")


def test_every_card_pins_turbo_bbox():
    for name in CARDS:
        config = _load(CONFIGS / f"config-{name}.yaml")
        prompts = _load(CONFIGS / f"prompts-{name}.yaml")
        model = config["pretrained_model"]
        assert model["name_or_path"] == MODEL_ID
        assert model["subfolder"] == TRANSFORMER_SUBFOLDER
        assert model["pipeline_id"] == PIPELINE_ID
        assert config["prompts_file"] == f"configs/krea2/prompts-{name}.yaml"
        assert (ROOT / config["prompts_file"]).is_file()
        assert config["network"]["rank"] == 16
        assert config["network"]["training_method"] == "lora"
        assert config["train"]["resolution"] == RESOLUTION
        sample = config["sample"]
        assert sample["mode"] == "turbo"
        assert sample["turbo_steps"] == TURBO_STEPS
        assert sample["turbo_guidance"] == TURBO_GUIDANCE
        assert sample["mu"] == TURBO_MU
        assert sample["control_prompt"] == CONTROL_PROMPT
        assert sample["scales"] == [0.0, 0.25, 0.5, 1.0]
        assert prompts["bare_captions"] is True
        assert prompts["control_prompt"] == CONTROL_PROMPT
        assert prompts["recommended_range"] == [0.0, 2.0]
        assert prompts["concept_words"]
        rows = prompts["rows"]
        assert len(rows) >= 2
        for row in rows:
            assert row["action"] == "enhance"
            assert row["guidance_scale"] == TURBO_GUIDANCE
            assert row["resolution"] == RESOLUTION
            assert row["target"] == row["neutral"]
            assert row["negative"]
            assert row["positive"] != row["neutral"]
            assert row["attributes"] == list(UNUSED)
            for field in ("target", "positive", "neutral"):
                for token in UNUSED:
                    assert token not in row[field]


def test_repo_does_not_vendor_weights():
    banned = {".safetensors", ".pt", ".pth", ".ckpt"}
    # A local CUDA run writes ignored checkpoints; only versioned weights are vendored.
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT, text=True).split('\0')
    found = [path for path in tracked if Path(path).suffix in banned]
    assert found == []
