"""Docs name this repo as the Krea2 product and pin particle-sliders-core."""

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
        "winning_formulation",
        "particle-gmix-1600-v2",
    ):
        assert needle in readme, needle
    assert "does not ship slider weights" in readme
    assert "checkout is not required" not in readme.lower()
    assert "do not depend on particle-sliders" not in readme.lower()


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
        "4340e28bed388d50800c469525b460a108091da0",
        "winning_formulation",
        "particle-gmix-1600-v2",
        "subdirectory=packages/particle-sliders-core",
    ):
        assert needle in text, needle
    assert "finished Krea-2 slider" in text
    assert "checkout is not required" not in text.lower()
    assert "do not depend on particle-sliders" not in text.lower()


def test_formulation_boundaries():
    text = _text("FORMULATION.md")
    assert "HyperGAN/conceptmod" in text
    assert "ParticleGAN" in text
    assert "does not vendor ParticleGAN" in text
    assert "jimmycarter/krea2-turbo-bbox" in text
    assert "krea2-particle-sliders" in text
    assert "winning_formulation()" in text
    assert "stamp.require" in text
    assert "particle-gmix-1600-v2" in text
    assert "gmix" in text
    assert "4340e28bed388d50800c469525b460a108091da0" in text
    assert "subdirectory=packages/particle-sliders-core" in text
    assert "comfy_krea2.py" in text
    lowered = text.lower()
    assert "comes back" not in lowered
    assert "local config" not in lowered
    assert "checkout is not required" not in lowered
    assert "do not depend on particle-sliders" not in lowered
    assert "not a wrapper around particle-sliders" not in lowered


def test_requirements_pin_shared_core():
    text = _text("requirements.txt")
    pin = (
        "particle-sliders-core @ git+https://github.com/HyperGAN/particle-sliders.git"
        "@4340e28bed388d50800c469525b460a108091da0"
        "#subdirectory=packages/particle-sliders-core"
    )
    assert pin in text


def test_prompting_pointer_is_short():
    text = _text("PROMPTING.md")
    assert "https://huggingface.co/jimmycarter/krea2-turbo-bbox/blob/main/PROMPTING.md" in text
    assert "[x0, y0, x1, y1]" in text
    assert "0–1000" in text
    for tag in ("`p`", "`o`", "`t`", "`pe`", "`ac`", "`fc`"):
        assert tag in text
    assert len(text) < 8000
