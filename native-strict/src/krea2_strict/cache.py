"""Fresh strict teacher capture and exact native parity; old caches are refused."""
from copy import deepcopy

from krea2.lifecycle import _cache_identity
from krea2.pg09 import tensor_identity

from .execution import require_policy
from .native import check_cache_geometry, observation


def coordinate_identity(args):
    return dict(model_id=args.model_id, transformer_subfolder=args.transformer_subfolder,
        revision=args.revision, skeleton=args.skeleton_model, skeleton_revision=args.skeleton_revision,
        rank=args.rank, resolution=args.resolution, sample_steps=args.sample_steps,
        guidance=args.sample_guidance, mu=args.mu, particles=None,
        objective="g_adv;d_adv+public_penalty")


def validate_cache(cache, args, execution, assets):
    identity = _cache_identity(cache, args)
    marker = identity.get("strict_native")
    expected = dict(schema="krea2-strict-native-teacher-v1", execution=execution, assets=assets,
                    teacher="fresh exact native conditional velocity, OFF adapter, both trajectories")
    if marker != expected:
        raise ValueError("strict native cache source/numerical/assets differ; fresh capture required, no legacy rebinding")
    return identity


def capture(native, edit_pairs, preserve_prompts, seeds):
    """Capture current frozen responses, never relabel historical teacher bytes."""
    if (not isinstance(edit_pairs, list) or not edit_pairs or not isinstance(preserve_prompts, list)
            or not preserve_prompts or not isinstance(seeds, list) or not seeds
            or any(type(s) is not int or s < 0 for s in seeds)
            or any(not isinstance(pair, (list, tuple)) or len(pair) != 2 or pair[0] == pair[1] for pair in edit_pairs)
            or any(not isinstance(s, str) or not s.strip() for pair in edit_pairs for s in pair)
            or any(not isinstance(s, str) or not s.strip() for s in preserve_prompts)):
        raise ValueError("explicit nonempty edited/preservation prompt pools and seeds required")
    if not native.strict_assets.get("full_weight_hashes_verified"):
        raise ValueError("full actual frozen asset hashes required for teacher capture")
    cache = dict(edit=[], preserve=[])
    with observation(native):
        for seed in seeds:
            for neutral, positive in edit_pairs:
                for positive_rollout in (False, True):
                    cache["edit"].extend(native.trajectory(neutral, positive, seed, positive_rollout))
            for prompt in preserve_prompts:
                cache["preserve"].extend(native.trajectory(prompt, prompt, seed))
    cache["key"] = dict(coordinate_identity(native.args),
        frozen_model_sha256=native.frozen_identity(), conditioning_identity=deepcopy(native.conditioning_identity),
        strict_native=dict(schema="krea2-strict-native-teacher-v1", execution=deepcopy(native.strict_execution),
            assets=deepcopy(native.strict_assets),
            teacher="fresh exact native conditional velocity, OFF adapter, both trajectories"))
    cache["parity"] = verify_native_parity(native, cache)
    return cache


def verify_native_parity(native, cache):
    """Measure all current rows under the actual shared adapter, before updates."""
    import torch
    from particle_sliders.pg09_images import ImageResponseAdapter, image_lora_parameters
    from krea2.pg09 import Krea2Rows
    require_policy(native.strict_execution["numerical"])
    check_cache_geometry(native, cache)
    rows = Krea2Rows(native, cache, cache_sha256="fresh-in-memory-parity")
    adapter = ImageResponseAdapter("krea2", native.model, image_lora_parameters(native.model),
        pipeline=native.pipe, scheduler=native.pipe.scheduler, model_identity={"strict_native_parity": True})
    identities = []
    with torch.no_grad(), observation(native):
        # Unique stored records, rather than the sampler's repeated4:1 slots.
        for pool in ("edit", "preserve"):
            for row in cache[pool]:
                prepared = rows._row(row, rows._unpack(row["z"]).to(native.device), device=native.device)
                for context, key in ((prepared.neutral, "baseline"), (prepared.positive, "teacher")):
                    actual = adapter.velocity(context, prepared.latent, prepared.timestep, scale=0).float().cpu()
                    target = rows._unpack(row[key]).float().cpu()
                    if not torch.equal(actual, target):
                        raise ValueError("strict native teacher/adapter OFF parity differs")
                identities.append({k: tensor_identity(row[k]) for k in ("z", "t", "baseline", "teacher")})
    return dict(schema="krea2-strict-native-parity-v1", all_rows_exact=True,
                unique_stored_rows=len(identities), tensors=identities)
