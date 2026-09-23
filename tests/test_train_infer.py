"""In-repo train/infer wiring. No Hub download.

The shared game is particle-sliders-core ``winning_formulation()``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from krea2.card import (
    assert_krea2_only,
    assert_krea2_skeleton,
    live_train_command,
    resolve_source,
)
from krea2.defaults import COMFY_FILENAME, MODEL_ID, TRANSFORMER_SUBFOLDER
from krea2.prompts import load_prompts, unused_words_for
from krea2.train import parse_args, train
from krea2.uni import plus_neu_teachers, scheduler_mu

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "configs" / "krea2" / "prompts-smile.yaml"
LIVE = ROOT / "krea2" / "live.py"


def _run(script: str, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_help_names_product_and_shared_game():
    train_proc = _run("train_krea2.py", ["--help"])
    assert train_proc.returncode == 0, train_proc.stderr
    text = train_proc.stdout
    assert MODEL_ID in text
    assert TRANSFORMER_SUBFOLDER in text
    assert "krea/Krea-2-Raw" in text
    assert "--dummy" in text
    assert "winning_formulation()" in text
    assert "gmix-1600-v2" in text
    assert "gmix" in text
    infer_proc = _run("infer_krea2.py", ["--help"])
    assert infer_proc.returncode == 0, infer_proc.stderr
    infer_text = infer_proc.stdout
    assert MODEL_ID in infer_text
    assert TRANSFORMER_SUBFOLDER in infer_text
    assert "krea/Krea-2-Raw" in infer_text
    assert "--dummy" in infer_text
    train_src = (ROOT / "krea2" / "train.py").read_text(encoding="utf-8")
    script_src = (ROOT / "scripts" / "train_krea2.py").read_text(encoding="utf-8")
    infer_src = (ROOT / "scripts" / "infer_krea2.py").read_text(encoding="utf-8")
    assert "winning_formulation" in train_src
    assert "lock_winning_formulation" in train_src
    assert "winning_formulation" in script_src
    assert "conceptmod.textsliders" not in train_src
    assert "conceptmod.textsliders" not in infer_src


def test_print_card_is_the_inrepo_cli():
    card = train(parse_args(["--print_card"]))
    assert card["backend"] == "krea2_turbo_bbox"
    assert card["model_id"] == MODEL_ID
    assert card["transformer_subfolder"] == TRANSFORMER_SUBFOLDER
    assert card["sample_steps"] == 8
    assert card["sample_guidance"] == 0.0
    assert card["mu"] == 1.15
    assert card["not_raw_card"]["raw_cfg"] == 4.5
    formulation = card["formulation"]
    assert formulation["architecture_id"] == "gmix"
    assert formulation["formulation_id"] == "particle-gmix-1600-v2"
    assert formulation["formulation_provisional"] is True
    assert formulation["critic"] == "gmix"
    assert formulation["parts"] == 128
    assert formulation["particle_dim"] == 4
    assert formulation["noise_decay_steps"] == 1600
    assert "particle-sliders checkout" not in card["non_goals"]
    assert "vendored ParticleGAN" in card["non_goals"]
    command = live_train_command()
    assert "python scripts/train_krea2.py" in command
    assert "conceptmod/textsliders" not in command
    assert "--transformer_subfolder epoch-14-step-73184/transformer" in command
    assert "--skeleton_model krea/Krea-2-Raw" in command
    assert "--sample_guidance 0" in command
    assert "--mu 1.15" in command


def test_cfg0_teacher_is_not_raw_4_5():
    plus, zero = plus_neu_teachers([2.0, 0.0], [0.0, 1.0], [0.5, 0.5], guidance=0.0)
    assert plus == [2.0, 0.0]
    assert zero == [0.0, 1.0]
    raw_plus, _raw_zero = plus_neu_teachers(
        [2.0, 0.0], [0.0, 1.0], [0.5, 0.5], guidance=4.5
    )
    assert raw_plus != plus
    assert scheduler_mu(is_distilled=True, mu=None) == pytest.approx(1.15)
    assert scheduler_mu(is_distilled=False, mu=None) is None
    assert scheduler_mu(is_distilled=False, mu=1.15) == pytest.approx(1.15)


def test_smile_yaml_keeps_bare_captions_and_dsl():
    rows, meta = load_prompts(PROMPTS)
    text = PROMPTS.read_text(encoding="utf-8")
    assert meta.bare_captions is True
    assert meta.control_prompt == "a bowl of fruit on a table"
    assert len(rows) == 3
    assert rows[0].guidance_scale == 0.0
    assert "male" not in rows[0].positive
    grounded = rows[2]
    assert "pe[200,80,800,980]" in grounded.positive
    unused = unused_words_for(grounded)
    assert "male" in unused and "female" in unused
    assert "smile" not in unused
    assert "PROMPTING.md" in text


def test_refuses_stock_raw_and_foreign_ids():
    with pytest.raises(ValueError, match="Krea2-turbo-bbox-only"):
        assert_krea2_only("krea/Krea-2-Raw")
    with pytest.raises(ValueError, match="anima"):
        assert_krea2_only("circlestone-labs/Anima-Base-v1.0-Diffusers")
    assert_krea2_only(MODEL_ID)
    assert_krea2_skeleton("krea/Krea-2-Raw")
    with pytest.raises(ValueError, match="skeleton"):
        assert_krea2_skeleton("krea/Krea-2-Turbo")
    with pytest.raises(ValueError, match="Krea2-turbo-bbox-only"):
        train(parse_args(["--dummy", "--model_id", "krea/Krea-2-Raw", "--steps", "1"]))


def test_load_plan_without_download(tmp_path: Path):
    hub = resolve_source(MODEL_ID)
    assert hub["source"] == "hub_subfolder"
    assert hub["subfolder"] == TRANSFORMER_SUBFOLDER
    assert hub["transformer_path"] is None
    comfy = tmp_path / COMFY_FILENAME
    comfy.write_bytes(b"not-a-real-weight")
    plan = resolve_source(MODEL_ID, transformer=str(comfy))
    assert plan["source"] == "comfy_safetensors"
    root = tmp_path / "transformer"
    root.mkdir()
    (root / "config.json").write_text("{}", encoding="utf-8")
    local = resolve_source(str(root))
    assert local["source"] == "local_transformer"
    assert local["subfolder"] == ""


def test_dummy_train_ignores_row_guidance_4_5(tmp_path: Path):
    prompts = tmp_path / "prompts.yaml"
    prompts.write_text(
        """
plus_label: Happy
minus_label: Sad
concept_words: "smiling, smile, teeth"
control_prompt: "a bowl of fruit on a table"
bare_captions: true
rows:
  - target: "a person, neutral expression, closed mouth"
    positive: "a person, big smile showing teeth, happy joyful expression"
    neutral: "a person, neutral expression, closed mouth"
    negative: "a sad person"
    attributes: ["male", "female"]
    guidance_scale: 4.5
""",
        encoding="utf-8",
    )
    sidecar_path = train(
        parse_args(
            [
                "--dummy",
                "--name",
                "smile-krea2-dummy",
                "--prompts_file",
                str(prompts),
                "--save_dir",
                str(tmp_path / "out"),
                "--steps",
                "8",
                "--hold_weight",
                "0.1",
                "--seed",
                "7",
            ]
        )
    )
    payload = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
    assert payload["kind"] == "krea2_turbo_bbox"
    assert payload["sample_steps"] == 8
    assert payload["sample_guidance"] == 0.0
    assert payload["train_guidance"] == 0.0
    assert payload["mu"] == pytest.approx(1.15)
    assert payload["weights"]["source"] == "hub_subfolder"
    assert payload["dummy"] is True
    assert payload["allow_hub"] is False
    formulation = payload["formulation"]
    assert formulation["architecture_id"] == "gmix"
    assert formulation["formulation_id"] == "particle-gmix-1600-v2"
    assert formulation["formulation_provisional"] is True
    smoke = payload["gmix_smoke"]
    assert smoke["gmix_regularizer_arm"] == "b_cap"
    assert smoke["gmix_regularizer_lazy_k"] == 4
    assert smoke["gmix_penalty_applied"] is True
    assert smoke["gmix_noise_std"] > 0
    assert smoke["gmix_particle_grad_norm"] > 0
    lines = [
        json.loads(line)
        for line in (tmp_path / "out" / "smile-krea2-dummy_train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(lines) == 2
    assert lines[-1]["minus_teacher"] == 0.0
    meta = json.loads((tmp_path / "out" / "samples" / "final_meta.json").read_text(encoding="utf-8"))
    assert meta["gate"] == "smile-first"
    assert "a bowl of fruit on a table" in meta["prompts"]
    assert all(shot["cfg"] == 0.0 for shot in meta["samples"])
    assert all(shot["sample_steps"] == 2 for shot in meta["samples"])


def test_dummy_default_yaml_does_not_import_live(tmp_path: Path):
    sys.modules.pop("krea2.live", None)
    sidecar = json.loads(
        Path(
            train(
                parse_args(
                    [
                        "--dummy",
                        "--prompts_file",
                        str(PROMPTS),
                        "--save_dir",
                        str(tmp_path),
                        "--steps",
                        "4",
                        "--seed",
                        "3",
                    ]
                )
            )
        ).read_text(encoding="utf-8")
    )
    assert sidecar["transformer_subfolder"] == TRANSFORMER_SUBFOLDER
    pngs = list((tmp_path / "samples").glob("*.png"))
    assert len(pngs) == 16
    assert "krea2.live" not in sys.modules


def test_explicit_guidance_override(tmp_path: Path):
    payload = json.loads(
        Path(
            train(
                parse_args(
                    [
                        "--dummy",
                        "--sample_guidance",
                        "1.5",
                        "--sample_steps",
                        "6",
                        "--mu",
                        "0.8",
                        "--save_dir",
                        str(tmp_path),
                        "--steps",
                        "1",
                    ]
                )
            )
        ).read_text(encoding="utf-8")
    )
    assert payload["sample_guidance"] == pytest.approx(1.5)
    assert payload["sample_steps"] == 6
    assert payload["mu"] == pytest.approx(0.8)
    assert payload["train_guidance"] == pytest.approx(1.5)


def test_embed_dummy_stays_on_distilled_card(tmp_path: Path):
    payload = json.loads(
        Path(
            train(
                parse_args(
                    [
                        "--dummy",
                        "--lm_target",
                        "embed",
                        "--lora_targets",
                        "te",
                        "--save_dir",
                        str(tmp_path),
                        "--steps",
                        "1",
                    ]
                )
            )
        ).read_text(encoding="utf-8")
    )
    assert payload["lm_target"] == "embed"
    assert payload["lora_targets"] == "te"
    assert payload["sample_guidance"] == 0.0
    assert payload["dit_velocity_supervised"] is False


def test_nondummy_refuses_before_hub(tmp_path: Path):
    with pytest.raises(RuntimeError, match="does not download"):
        train(parse_args(["--save_dir", str(tmp_path), "--steps", "1"]))
    assert "krea2.live" not in sys.modules


def test_infer_script_dummy_grid(tmp_path: Path):
    proc = _run(
        "infer_krea2.py",
        [
            "--dummy",
            "--load_te_lora",
            "models/missing-adapter",
            "--save_dir",
            str(tmp_path),
            "--steps",
            "4",
        ],
    )
    assert proc.returncode == 0, proc.stderr
    assert list((tmp_path / "samples").glob("*.png"))
    proc_missing = _run("infer_krea2.py", ["--dummy"])
    assert proc_missing.returncode != 0
    assert "load_te_lora" in proc_missing.stderr


def test_live_module_pins_distilled_load():
    src = LIVE.read_text(encoding="utf-8")
    assert "Krea2Transformer2DModel" in src
    assert "Krea2Pipeline" in src
    assert "epoch-14-step-73184/transformer" in src
    assert "krea/Krea-2-Raw" in src
    assert "is_distilled" in src
    assert "load_comfy_krea_transformer" in src
    assert "conceptmod.textsliders" not in src


def test_train_locks_winning_formulation_and_does_not_copy_the_game():
    from krea2.formulation import lock_winning_formulation

    stamp = lock_winning_formulation()
    assert stamp.architecture_id == "gmix"
    assert stamp.family == "particle-gmix"
    assert stamp.formulation_id == "particle-gmix-1600-v2"
    assert stamp.formulation_provisional is True
    forked = stamp.as_dict()
    forked["critic"] = "mlp"
    with pytest.raises(ValueError, match="drift"):
        stamp.require(forked)
    product = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "krea2").glob("*.py"))
    product += "\n" + (ROOT / "comfy_krea2.py").read_text(encoding="utf-8")
    for needle in (
        "class RoutedMLP",
        "class GradRegularizer",
        "class GlobalMixErrorCritic",
        "def locked_shared",
    ):
        assert needle not in product, needle


def test_comfy_node_is_registered():
    import comfy_krea2

    node = comfy_krea2.NODE_CLASS_MAPPINGS["NTCKrea2BboxLora"]
    assert node.CATEGORY == "NTC/Krea2"
    assert comfy_krea2.validate_strength(0) == 0
    with pytest.raises(ValueError):
        comfy_krea2.validate_strength(9)
    source = Path(comfy_krea2.__file__).read_text(encoding="utf-8")
    assert "krea2-bbox-turbo-comfy-latest.safetensors" in source
    assert "NTC/Krea2" in source
