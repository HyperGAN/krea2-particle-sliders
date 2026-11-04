"""Separate strict adapter-GAN lifecycle; no GAN implementation is duplicated."""
from pathlib import Path

from particle_sliders.lifecycle import Backend, Binding, Capabilities, SessionConfig
from krea2.lifecycle import _native_arguments, _validate_prompts, _validate_device
from krea2.native_runtime import check_request, render_callback

from . import BACKEND_NAME, COHORT
from .assets import require_args
from .execution import require_policy, resolution
from .provenance import execution_identity

ALLOWED = {"cache_file", "native_args", "steps", "seed", "batch", "preset", "lr", "d_lr_mult",
           "base_precision", "ema_decay", "device", "render_prompts", "expected_gpu_uuid"}


def config(*, cohort=COHORT, **options):
    defaults = dict(steps=4, seed=7, batch=2, lr=5e-5, preset="bispf_optimized",
                    device="cuda:0", base_precision="bfloat16", ema_decay=None)
    defaults.update(options)
    return SessionConfig(BACKEND_NAME, cohort, **defaults)


def arguments(options):
    args = _native_arguments(options)
    if args.device != 0:
        raise ValueError("strict native route owns only visible cuda:0")
    _validate_device(options, args)
    for name in ("resolution", "sample_resolution", "final_resolution"):
        resolution(getattr(args, name), allow_zero=name == "final_resolution")
    require_args(args)
    return args


def validate(options):
    if set(options) - ALLOWED or options.get("base_precision", "bfloat16") != "bfloat16":
        raise ValueError("unsupported strict native options/precision; adapter-only prior rate is unavailable")
    args = arguments(options)
    _validate_prompts(options.get("render_prompts"))
    if options.get("preset", "bispf_optimized") not in ("bispf", "bispf_optimized"):
        raise ValueError("unsupported native PG preset")
    from .execution import canonical_gpu_uuid
    canonical_gpu_uuid(options.get("expected_gpu_uuid"))
    execution = execution_identity()
    path = options.get("cache_file")
    if not isinstance(path, str) or not Path(path).is_file():
        raise ValueError("strict native requires an existing fresh cache_file")
    import torch
    from .cache import validate_cache
    assets = require_args(args)
    # The receipt records mandatory full load-time verification, which is
    # repeated by StrictCudaUNI before allocation, not asserted by the caller.
    assets["full_weight_hashes_verified"] = True
    validate_cache(torch.load(path, map_location="cpu", weights_only=True), args, execution, assets)


def preflight(options, request, staged):
    check_request(request, prompts=options.get("render_prompts", []))
    if request["operation"] == "render":
        resolution(request["native"].get("resolution", 768))
    if staged.get("before_update") is not None and not callable(staged["before_update"]):
        raise ValueError("before_update must be an admission callback")


def prepare(options, *, native=None, cache=None, before_update=None):
    import torch
    from .assets import digest
    from .cache import validate_cache, verify_native_parity
    from .game import build_game
    from .native import StrictCudaUNI, check_cache_geometry, observation, require_live
    from .inference import specification
    args = arguments(options)
    execution = execution_identity()
    assets = require_args(args)
    assets["full_weight_hashes_verified"] = True
    cache = cache if cache is not None else torch.load(options["cache_file"], map_location="cpu", weights_only=True)
    identity = validate_cache(cache, args, execution, assets)
    texts = sorted({row[k] for pool in ("edit", "preserve") for row in cache[pool] for k in ("prompt", "positive")}
                   | set(options.get("render_prompts", [])))
    if native is None:
        native = StrictCudaUNI(args, texts, Path(args.save_dir), expected_gpu_uuid=options["expected_gpu_uuid"])
    game = None
    try:
        if (native.strict_execution != execution or native.strict_assets != assets
                or identity["frozen_model_sha256"] != native.frozen_identity()
                or identity["conditioning_identity"] != native.conditioning_identity):
            raise ValueError("strict cache differs from actual staged base/conditioning/execution")
        check_cache_geometry(native, cache)
        # Self-asserted receipt is insufficient: all teacher tensors must equal
        # fresh current native/shared OFF responses before the game is created.
        parity = verify_native_parity(native, cache)
        game = build_game(native, cache, cache_sha256=digest(options["cache_file"]),
            model_identity=dict(identity, native_parity=parity), total_steps=options.get("steps", 4),
            seed=options.get("seed", 7), batch_size=options.get("batch", 2),
            preset=options.get("preset", "bispf_optimized"), lr=options.get("lr", 5e-5),
            d_lr_mult=options.get("d_lr_mult"))
        inference = specification(native, game.backend)
    except BaseException:
        native.strict_close(game=game)
        raise
    def admit(state):
        require_live(native)
        require_policy(execution["numerical"])
        return before_update(state) if before_update is not None else True
    return Binding(game, lambda: native.strict_close(game=game), render_callback(native), inference=inference,
                   before_update=admit, observation=lambda: observation(native))


CAPABILITIES = Capabilities("adapter_gan", render=True, strength_min=None, strength_max=None,
    base_precisions=("bfloat16",), render_outputs=("pil",),
    qualification="Separate strict native source candidate; pretrained GPU update/resume/render unqualified")
BACKEND = Backend(BACKEND_NAME, CAPABILITIES, prepare, validate, preflight=preflight)
