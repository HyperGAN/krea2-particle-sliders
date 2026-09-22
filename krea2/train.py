"""In-repo train entry for Krea2 turbo-bbox sliders.

``python scripts/train_krea2.py`` calls ``main`` here. Nothing in this
module imports ``krea2.live`` unless the run is live (not ``--dummy``
and not ``--print_card``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from krea2.card import (
    DEFAULT_PROMPTS,
    assert_krea2_only,
    assert_krea2_skeleton,
    live_train_card,
    live_train_command,
    resolve_card,
    resolve_source,
)
from krea2.defaults import (
    CONTROL_PROMPT,
    HOLD_WEIGHT,
    LM_TARGET_DEFAULT,
    LORA_TARGETS_DEFAULT,
    MODEL_ID,
    RANK,
    RECIPE_DEFAULT,
    RESOLUTION,
    SAMPLE_SCALES,
    SKELETON_MODEL,
    TRANSFORMER_SUBFOLDER,
    TURBO_GUIDANCE,
    TURBO_MU,
    TURBO_STEPS,
)
from krea2.dummy import DummyBackend, step_loss
from krea2.prompts import load_prompts
from krea2.samples import emit_grid

REPO_ROOT = Path(__file__).resolve().parents[1]
LM_CHOICES = ("v", "embed", "velocity", "embed_uni", "te", "text_encoder")
RECIPE_CHOICES = ("uni", "embed_uni")
LORA_CHOICES = ("dit", "te", "dit+te", "text_encoder")


def resolve_lm_target(lm_target: str | None, recipe: str | None) -> str:
    raw_lm = None if lm_target is None else str(lm_target).strip().lower()
    raw_recipe = None if recipe is None else str(recipe).strip().lower()
    from_recipe = "embed" if raw_recipe == "embed_uni" else None
    aliases = {
        "velocity": "v",
        "dit": "v",
        "uni": "v",
        "embed_uni": "embed",
        "te": "embed",
        "te_embed": "embed",
        "text_encoder": "embed",
    }
    from_lm = None
    if raw_lm:
        from_lm = aliases.get(raw_lm, raw_lm)
        if from_lm not in ("v", "embed"):
            raise ValueError(f"lm_target must be v or embed, got {lm_target!r}")
    if from_lm and from_recipe and from_lm != from_recipe:
        if from_recipe == "embed" and from_lm == "v":
            return "embed"
        raise ValueError(f"lm_target={lm_target!r} conflicts with recipe={recipe!r}")
    return from_lm or from_recipe or LM_TARGET_DEFAULT


def force_lora_targets(lora_targets: str, lm_target: str) -> str:
    if lm_target == "embed":
        return "te"
    if lora_targets in ("text_encoder", "encoder"):
        return "te"
    return lora_targets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Train a concept slider on jimmycarter/krea2-turbo-bbox "
            f"({TRANSFORMER_SUBFOLDER}). Distilled card: {TURBO_STEPS} steps, "
            f"guidance {TURBO_GUIDANCE:g}, mu={TURBO_MU:g}. "
            "VAE and text encoder come from krea/Krea-2-Raw. "
            "--dummy never downloads Hub weights."
        )
    )
    parser.add_argument("--name", type=str, default="smile-krea2-bbox")
    parser.add_argument("--rank", type=int, default=RANK)
    parser.add_argument("--resolution", type=int, default=RESOLUTION)
    parser.add_argument(
        "--model_id",
        type=str,
        default=MODEL_ID,
        help="Hub repo for the turbo-bbox transformer",
    )
    parser.add_argument(
        "--transformer_subfolder",
        type=str,
        default=TRANSFORMER_SUBFOLDER,
        help="diffusers subfolder inside --model_id",
    )
    parser.add_argument(
        "--transformer",
        type=str,
        default=None,
        help="local Comfy .safetensors or a diffusers transformer directory",
    )
    parser.add_argument(
        "--skeleton_model",
        type=str,
        default=SKELETON_MODEL,
        help="VAE, text encoder, tokenizer, and scheduler (not the bbox repo)",
    )
    parser.add_argument("--prompts_file", type=str, default=DEFAULT_PROMPTS)
    parser.add_argument("--config_file", type=str, default="configs/krea2/config-smile.yaml")
    parser.add_argument("--save_dir", type=str, default="models/krea2-bbox-slider")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=TURBO_STEPS,
        help="distilled sample steps",
    )
    parser.add_argument(
        "--sample_guidance",
        type=float,
        default=TURBO_GUIDANCE,
        help="distilled guidance_scale (0 is CFG off)",
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=TURBO_MU,
        help="distilled timestep shift, not a CFG scale",
    )
    parser.add_argument("--hold_weight", type=float, default=HOLD_WEIGHT)
    parser.add_argument("--lora_targets", type=str, default=LORA_TARGETS_DEFAULT, choices=LORA_CHOICES)
    parser.add_argument("--lm_target", type=str, default=LM_TARGET_DEFAULT, choices=LM_CHOICES)
    parser.add_argument("--recipe", choices=list(RECIPE_CHOICES), default=RECIPE_DEFAULT)
    parser.add_argument("--dummy", action="store_true", help="CPU backend, 2 steps, no Hub weights")
    parser.add_argument("--live", action="store_true", help="run real CUDA DiT UNI training")
    parser.add_argument("--revision", default="ec7aa643a4da7e56a08c1778e03e837a2e57a94b")
    parser.add_argument("--skeleton_revision", default="6b0ece7fffb640c5e3bcbe0a7f10f66b8e60a603")
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--sample_resolution", type=int, default=768)
    parser.add_argument("--final_resolution", type=int, default=1536)
    parser.add_argument("--cache_seeds", type=int, default=2)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--allow_hub", action="store_true", help="permit a Hub download on a live run")
    parser.add_argument("--control_prompt", type=str, default=None)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument(
        "--load_te_lora",
        type=str,
        default=None,
        help="skip the train loop and write the sample grid for this adapter",
    )
    parser.add_argument("--print_card", action="store_true", help="print the distilled card and exit")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _prompts_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    candidate = REPO_ROOT / path
    return candidate if candidate.exists() else path


def train(args: argparse.Namespace) -> dict | Path:
    assert_krea2_only(args.model_id, getattr(args, "transformer", None))
    assert_krea2_skeleton(str(args.skeleton_model))
    if args.print_card:
        card = live_train_card(
            name=args.name,
            prompts_file=args.prompts_file,
            model_id=args.model_id,
            subfolder=args.transformer_subfolder,
            skeleton=args.skeleton_model,
            rank=int(args.rank),
            resolution=int(args.resolution),
            sample_steps=int(args.sample_steps),
            sample_guidance=float(args.sample_guidance),
            mu=float(args.mu),
            hold_weight=float(args.hold_weight),
            lora_targets=str(args.lora_targets),
        )
        print(json.dumps(card, indent=2))
        print()
        print(
            live_train_command(
                name=args.name,
                prompts_file=args.prompts_file,
                model_id=args.model_id,
                subfolder=args.transformer_subfolder,
                skeleton=args.skeleton_model,
                rank=int(args.rank),
                resolution=int(args.resolution),
                sample_steps=int(args.sample_steps),
                sample_guidance=float(args.sample_guidance),
                mu=float(args.mu),
                hold_weight=float(args.hold_weight),
                lora_targets=str(args.lora_targets),
                save_dir=args.save_dir,
            )
        )
        return card

    prompts, meta = load_prompts(_prompts_path(Path(args.prompts_file)))
    lm_target = resolve_lm_target(args.lm_target, args.recipe)
    lora_targets = force_lora_targets(str(args.lora_targets), lm_target)
    card = resolve_card(args.sample_steps, args.sample_guidance, args.mu)
    weights = resolve_source(
        args.model_id,
        subfolder=args.transformer_subfolder,
        transformer=getattr(args, "transformer", None),
        skeleton=args.skeleton_model,
    )
    steps = int(args.steps)
    if args.live and args.dummy:
        raise ValueError("Choose either --live or --dummy")
    if args.live:
        if lm_target != "v" or lora_targets != "dit":
            raise ValueError("Live training currently supports --lm_target v --lora_targets dit")
        from krea2.cuda_train import train_cuda
        return train_cuda(args, prompts, meta)
    if not args.dummy:
        raise RuntimeError(
            "This entry runs the in-repo CPU UNI loop with --dummy. "
            "The live loader is krea2.live.load_krea2_bbox_pipeline "
            f"({args.model_id} subfolder {args.transformer_subfolder} into "
            f"{args.skeleton_model}, {int(card['sample_steps'])} steps, "
            f"guidance {float(card['sample_guidance']):g}, mu={float(card['mu']):g}). "
            "A non-dummy run is refused so this process does not download Hub weights."
        )
    steps = min(steps, 2)
    backend = DummyBackend(seed=int(args.seed), lora_targets=lora_targets)

    skip_train = bool(getattr(args, "load_te_lora", None))
    if skip_train:
        if hasattr(backend, "load_te_adapter"):
            backend.load_te_adapter(args.load_te_lora)
        steps = 0

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / f"{args.name}_train.jsonl"
    guidance = float(card["sample_guidance"])
    control_prompt = str(args.control_prompt or meta.control_prompt or CONTROL_PROMPT)
    last_stats: dict[str, float] = {}
    params = backend.trainable_parameters()
    for step in range(steps):
        prompt = prompts[step % len(prompts)]
        backend.begin_step()
        z = backend.sample_latents()
        loss, stats = step_loss(
            backend,
            prompt,
            z,
            guidance=guidance,
            hold_weight=float(args.hold_weight),
            lm_target=lm_target,
        )
        loss.backward()
        lr = float(args.lr)
        for param in params:
            param.data -= lr * float(param.grad)
            param.grad = 0.0
        last_stats = stats
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": step, **stats}) + "\n")

    samples = emit_grid(
        backend,
        prompts,
        save_dir,
        dummy=bool(args.dummy),
        control_prompt=control_prompt,
        card=card,
        sample_seed=int(args.sample_seed),
    )
    sidecar = {
        "kind": "krea2_turbo_bbox",
        "recipe": "embed_uni" if lm_target == "embed" else "uni",
        "lm_target": lm_target,
        "dit_velocity_supervised": lm_target != "embed",
        "name": args.name,
        "model_id": args.model_id,
        "transformer_subfolder": args.transformer_subfolder,
        "transformer": getattr(args, "transformer", None),
        "rank": int(args.rank),
        "resolution": int(args.resolution),
        "official": (
            "train and sample on jimmycarter/krea2-turbo-bbox "
            f"({TURBO_STEPS} steps, CFG {TURBO_GUIDANCE:g}, mu={TURBO_MU:g})"
        ),
        "not_raw_card": {"raw_cfg": 4.5, "raw_steps": 28},
        "minus_teacher": False,
        "minus_canary": True,
        "token_hold": "unused_to_neu",
        "lyric_hold": False,
        "dummy": bool(args.dummy),
        "allow_hub": bool(args.allow_hub),
        "lora_targets": lora_targets,
        "hold_weight": float(args.hold_weight),
        "plus_label": meta.plus_label,
        "minus_label": meta.minus_label,
        "concept_words": meta.concept_words,
        "control_prompt": control_prompt,
        "bare_captions": bool(meta.bare_captions),
        "weights": weights,
        "sample_grid": {
            "scales": list(SAMPLE_SCALES),
            "gate": "smile-first",
            "crop_purity": False,
            "count": len(samples),
            "dir": "samples",
        },
        "train_guidance": guidance,
        "load_te_lora": getattr(args, "load_te_lora", None),
        "skipped_train": bool(skip_train),
        "last": last_stats,
        "music3_default_untouched": {"lm_target": "v9", "pole_mode": "hidden"},
        **card,
        "skeleton": str(args.skeleton_model),
        "mu": float(card["mu"]),
        "sample_steps": int(card["sample_steps"]),
        "sample_guidance": float(card["sample_guidance"]),
        "is_distilled": True,
    }
    sidecar_path = save_dir / f"{args.name}_last.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(f"wrote {sidecar_path}")
    return sidecar_path


def main(argv: list[str] | None = None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()
