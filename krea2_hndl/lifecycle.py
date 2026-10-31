"""Krea2 PEFT adapter GAN using the real shared HyperGAN/HNDL field critic."""
import torch

from particle_sliders.fibo_hndl import HNDLFieldCritic
from particle_sliders.lifecycle import Backend
from particle_sliders.pg09 import ConditionalPG09
from particle_sliders.pg09_images import ImageResponseAdapter, image_lora_parameters
from krea2 import lifecycle as product
from krea2.pg09 import Krea2Rows
from .runtime import native_factory, validate_runtime


class Krea2HNDLFieldCritic(HNDLFieldCritic):
    """Remove only Krea2's native singleton time axis at the critic boundary.

    The game and public penalty retain genuine five-dimensional fields. Every
    critic network operation executes the unchanged shared HNDL graph.
    """
    def forward(self, response, condition):
        if response.ndim != 5 or response.shape[2] != 1:
            raise ValueError('Krea2 HNDL requires native [B,C,1,H,W] responses')
        return super().forward(response.squeeze(2), condition)


def build_game(native, cache, *, cache_sha256, model_identity, total_steps, seed=7,
               batch_size=2, preset='bispf_optimized', lr=5e-5):
    runtime = validate_runtime()
    rows = Krea2Rows(native, cache, cache_sha256=cache_sha256)
    adapter = ImageResponseAdapter('krea2', native.model, image_lora_parameters(native.model),
        pipeline=native.pipe, scheduler=native.pipe.scheduler,
        model_identity=dict(model_identity, frozen_cache=rows.identity(), shared_runtime=runtime))
    native.params = list(adapter.network.parameters())
    with torch.random.fork_rng(devices=[native.device.index or 0] if native.device.type == 'cuda' else []):
        torch.manual_seed(seed)
        critic = Krea2HNDLFieldCritic(rows.response_shape[1], rows.condition_width,
                                    rows.response_shape[3:]).to(native.device)
    return ConditionalPG09(adapter, rows, critic, particles=None,
        response_domain='continuous_field', response_scale=rows.response_scale,
        preset=preset, batch_size=batch_size, seed=seed, total_steps=total_steps,
        recipe_overrides={'lr': lr}, sampling='balanced')


def validate(options):
    product.validate(options, runtime_validator=validate_runtime)


def prepare(options, **staged):
    from .native_inference import specification
    return product.prepare(options, game_builder=build_game,
        specification_factory=specification, native_factory=native_factory, **staged)


BACKEND = Backend('krea2-hndl', product.capabilities({}), prepare, validate,
                  product.capabilities, product.preflight)
