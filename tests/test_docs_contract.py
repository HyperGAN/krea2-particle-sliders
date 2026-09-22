"""README and sibling docs name the locked base and stay a scaffold."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _text(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_readme_pins_base_and_backend():
    readme = _text("README.md")
    for needle in (
        "jimmycarter/krea2-turbo-bbox",
        "epoch-14-step-73184/transformer",
        "krea/Krea-2-Raw",
        "guidance_scale=0.0",
        "mu=1.15",
        "is_distilled",
        "8 steps",
        "train_lora_krea2",
        "train_lora_krea.py",
        "krea_live.py",
        "docs/krea-slider.md",
        "particle-sliders",
        "PROMPTING.md",
        "krea2-bbox-turbo-comfy-latest.safetensors",
        "Comfy-Org/Krea-2",
        "CFG 1.0",
        "scaffold",
    ):
        assert needle in readme, needle
    assert "does not ship slider weights" in readme


def test_comfy_doc():
    text = _text("COMFYUI.md")
    for needle in (
        "krea2-bbox-turbo-comfy-latest.safetensors",
        "qwen3vl_4b_fp8_scaled.safetensors",
        "qwen_image_vae.safetensors",
        "Comfy-Org/Krea-2",
        "8 steps",
        "CFG 1.0",
        "epoch-14-step-73184/transformer",
        "Load LoRA",
        "guidance_scale=0.0",
        "mu=1.15",
    ):
        assert needle in text, needle
    assert "does not vendor" in text


def test_reproduce_is_honest():
    text = _text("REPRODUCE.md")
    for needle in (
        "jimmycarter/krea2-turbo-bbox",
        "epoch-14-step-73184/transformer",
        "--dummy",
        "HF_HUB_OFFLINE",
        "train_lora_krea2",
        "not on",
    ):
        assert needle in text, needle
    assert "finished Krea-2 slider" in text


def test_formulation_boundaries():
    text = _text("FORMULATION.md")
    assert "HyperGAN/conceptmod" in text
    assert "ParticleGAN" in text
    assert "jimmycarter/krea2-turbo-bbox" in text


def test_prompting_pointer_is_short():
    text = _text("PROMPTING.md")
    assert "https://huggingface.co/jimmycarter/krea2-turbo-bbox/blob/main/PROMPTING.md" in text
    assert "[x0, y0, x1, y1]" in text
    assert "0–1000" in text
    for tag in ("`p`", "`o`", "`t`", "`pe`", "`ac`", "`fc`"):
        assert tag in text
    assert len(text) < 8000
