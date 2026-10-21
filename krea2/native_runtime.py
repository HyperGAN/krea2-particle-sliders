"""Native ownership and observation seams; no GAN update or model construction."""
from contextlib import contextmanager
from copy import deepcopy

from particle_sliders.lifecycle import UnsupportedOperation, native_cleanup

CARD_FIELDS = ('rank', 'resolution', 'sample_steps', 'sample_guidance', 'mu',
               'lora_targets', 'lm_target', 'recipe', 'revision', 'skeleton_revision')
PIPE_STATE = ('_guidance_scale', '_attention_kwargs', '_current_timestep',
              '_interrupt', '_num_timesteps')
VAE_STATE = ('_conv_num', '_conv_idx', '_feat_map', '_enc_conv_num',
             '_enc_conv_idx', '_enc_feat_map')


def configuration(native, *, dummy):
    return dict(dummy=dummy, card={} if dummy else
                {key: getattr(native.args, key) for key in CARD_FIELDS})


def own_native(native):
    restore = native_cleanup(native.model)
    closed = False
    def close():
        nonlocal closed
        if not closed:
            closed = True
            try:
                restore()
            finally:
                if hasattr(native.model, 'unload'):
                    native.pipe.transformer = native.model.unload()
    return close


@contextmanager
def observation(native):
    """Preserve native state outside the common adapter parameter container.

    Frozen base parameters are never cloned. Registered native buffers, sampler
    fields and the small position cache are restored; module surgery is unsupported.
    """
    restore = native_cleanup(native.model)
    scheduler = native.pipe.scheduler
    fields = {key: deepcopy(getattr(native.pipe, key))
              for key in PIPE_STATE if hasattr(native.pipe, key)}
    positions = native.position_ids
    saved_positions = deepcopy(positions)
    vae = getattr(native.pipe, 'vae', None)
    # Qwen-image VAE clear_cache replaces these lists before decode; retaining
    # the originals avoids cloning potentially large prior decoder activations.
    vae_state = {key: getattr(vae, key) for key in VAE_STATE if hasattr(vae, key)}
    roots = [native.model] + ([vae] if vae is not None else [])
    restore_vae = native_cleanup(vae) if vae is not None else lambda: None
    modules = {id(module): module for root in roots for module in root.modules()}
    buffers = [(module, dict(module._buffers), set(module._non_persistent_buffers_set))
               for module in modules.values() if hasattr(module, '_buffers')]
    values = {id(value): (value, value.detach().clone())
              for _, registered, _ in buffers for value in registered.values() if value is not None}
    native.pipe.scheduler = deepcopy(scheduler)
    try:
        yield
    finally:
        import torch
        with torch.no_grad():
            for value, saved in values.values():
                if (value.shape, value.dtype, value.device) == (saved.shape, saved.dtype, saved.device):
                    value.copy_(saved)
                else:
                    value.data = saved.clone()
        for module, registered, nonpersistent in buffers:
            module._buffers.clear()
            module._buffers.update(registered)
            module._non_persistent_buffers_set.clear()
            module._non_persistent_buffers_set.update(nonpersistent)
        native.pipe.scheduler = scheduler
        for key in PIPE_STATE:
            if key in fields:
                setattr(native.pipe, key, fields[key])
            elif hasattr(native.pipe, key):
                delattr(native.pipe, key)
        positions.clear()
        positions.update(saved_positions)
        native.position_ids = positions
        if vae is not None:
            for key in VAE_STATE:
                if key in vae_state:
                    setattr(vae, key, vae_state[key])
                elif hasattr(vae, key):
                    delattr(vae, key)
        restore_vae()
        restore()


def check_request(request, *, prompts=None):
    if request['operation'] != 'render':
        return
    native = request['native']
    if set(native) - {'prompt', 'resolution', 'images', 'output_type'}:
        raise ValueError('unsupported Krea2 native render arguments')
    if (request['references'] or native.get('images') or request['output_type'] != 'pil'
            or native.get('output_type', 'pil') != 'pil'):
        raise UnsupportedOperation('Krea2 native rendering accepts PIL output without references')
    if 'prompt' in native:
        if not isinstance(native['prompt'], str) or not native['prompt'].strip():
            raise ValueError('a nonempty native prompt is required')
        if prompts is not None and native['prompt'] not in prompts:
            raise UnsupportedOperation('Krea2 requires a cached prompt; declare it before native preparation')
    resolution = native.get('resolution', 768)
    if type(resolution) is not int or resolution <= 0 or resolution % 16:
        raise ValueError('native Krea2 resolution must be a positive multiple of 16')


def render_callback(native):
    def render(*, prompt, strength, seed, images, resolution=768, output_type='pil'):
        check_request(dict(operation='render', references=bool(images), output_type=output_type,
                           native=dict(prompt=prompt, resolution=resolution)), prompts=native.embeddings)
        with observation(native):
            return native.generate(prompt, strength, seed, resolution)
    return render
