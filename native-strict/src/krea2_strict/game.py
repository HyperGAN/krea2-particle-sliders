"""Only native binding and scoped critic initialization; core owns all GAN math."""
def build_game(native, cache, *, cache_sha256, model_identity, total_steps, seed=7,
               batch_size=2, preset="bispf_optimized", lr=5e-5, d_lr_mult=None):
    from krea2.learning_rates import recipe_overrides
    from krea2.pg09 import Krea2Rows
    from particle_sliders.pg09_images import ImageResponseAdapter, image_lora_parameters
    from particle_sliders.field_critic import FieldCritic
    from particle_sliders.pg09 import ConditionalPG09
    from .execution import initialization, require_policy
    rates = recipe_overrides(lr=lr, d_lr_mult=d_lr_mult)
    if "lr" not in rates:
        raise ValueError("native G rate must be finite and positive")
    require_policy(native.strict_execution["numerical"])
    rows = Krea2Rows(native, cache, cache_sha256=cache_sha256)
    adapter = ImageResponseAdapter("krea2", native.model, image_lora_parameters(native.model),
        pipeline=native.pipe, scheduler=native.pipe.scheduler,
        model_identity=dict(model_identity, frozen_cache=rows.identity()))
    native.params = list(adapter.network.parameters())
    with initialization(seed, device=str(native.device)):
        critic = FieldCritic(rows.response_shape[1], rows.condition_width).to(native.device)
    return ConditionalPG09(adapter, rows, critic, particles=None,
        response_domain="continuous_field", response_scale=rows.response_scale,
        preset=preset, batch_size=batch_size, seed=seed, total_steps=total_steps,
        recipe_overrides=rates, sampling="balanced")
