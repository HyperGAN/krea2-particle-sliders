"""Docs name this repo as the Krea2 product, not a particle-sliders wrapper."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _text(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_readme_is_the_product():
    readme = _text("README.md")
    for needle in (
        "krea2-particle-sliders",
        "anima-particle-sliders",
        "jimmycarter/krea2-turbo-bbox",
        "epoch-14-step-73184/transformer",
        "krea/Krea-2-Raw",
        "guidance_scale=0.0",
        "mu=1.15",
        "is_distilled",
        "8 steps",
        "scripts/train_krea2.py",
        "scripts/infer_krea2.py",
        "krea2-bbox-turbo-comfy-latest.safetensors",
        "Comfy-Org/Krea-2",
        "CFG 1.0",
        "PROMPTING.md",
        "rename",
    ):
        assert needle in readme, needle
    assert "PARTICLE_SLIDERS_ROOT" not in readme
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
        "NTC/Krea2",
        "guidance_scale=0.0",
        "mu=1.15",
        "comfy_krea2.py",
    ):
        assert needle in text, needle
    assert "does not vendor" in text


def test_reproduce_is_in_repo():
    text = _text("REPRODUCE.md")
    for needle in (
        "jimmycarter/krea2-turbo-bbox",
        "epoch-14-step-73184/transformer",
        "--dummy",
        "scripts/train_krea2.py",
        "scripts/infer_krea2.py",
        "pip install -r requirements.txt",
        "mu=1.15",
    ):
        assert needle in text, needle
    assert "PARTICLE_SLIDERS_ROOT" not in text
    assert "finished Krea-2 slider" in text


def test_formulation_boundaries():
    text = _text("FORMULATION.md")
    assert "HyperGAN/conceptmod" in text
    assert "ParticleGAN" in text
    assert "jimmycarter/krea2-turbo-bbox" in text
    assert "krea2-particle-sliders" in text


def test_prompting_pointer_is_short():
    text = _text("PROMPTING.md")
    assert "https://huggingface.co/jimmycarter/krea2-turbo-bbox/blob/main/PROMPTING.md" in text
    assert "[x0, y0, x1, y1]" in text
    assert "0–1000" in text
    for tag in ("`p`", "`o`", "`t`", "`pe`", "`ac`", "`fc`"):
        assert tag in text
    assert len(text) < 8000
