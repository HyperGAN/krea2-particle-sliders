"""Native-only v2 adapter loading: no training cache, game, critic or optimizer."""
from pathlib import Path

from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity
from particle_sliders.lifecycle import Capabilities, UnsupportedOperation

from .native_runtime import CARD_FIELDS, check_request, configuration, observation, own_native, render_callback

PROTOCOL = 'krea2-native-adapter-v2'


def capabilities(config):
    dummy = config.get('dummy', False)
    return Capabilities('adapter_gan', render=not dummy, strength_min=None, strength_max=None,
        base_precisions=('float32',) if dummy else ('bfloat16',), render_outputs=('pil',),
        operations=('response', 'render') if not dummy else ('response',),
        unavailable_reasons=(('render', 'tiny Krea2 fixture has no native sampler/decoder'),))


def validate(options, spec):
    from .lifecycle import _native_arguments, _validate_device, _validate_prompts
    from .shared import validate_shared_runtime
    config = spec.configuration
    if (set(config) != {'dummy', 'card'} or type(config['dummy']) is not bool
            or not isinstance(config['card'], dict)
            or set(config['card']) != (set() if config['dummy'] else set(CARD_FIELDS))):
        raise ValueError('invalid Krea2 native inference configuration')
    if set(options) - {'native_args', 'device', 'base_precision', 'prompts'}:
        raise ValueError('native inference accepts only local loading/device/prompt options, without cache or training settings')
    if spec.configuration['dummy']:
        if options.get('native_args') or options.get('prompts'):
            raise ValueError('tiny native fixture has fixed encodings and no native CLI arguments')
        _validate_device(options, None)
    else:
        args = _native_arguments(options)
        _validate_device(options, args)
        _validate_prompts(options.get('prompts'), required=True)
        if {key: getattr(args, key) for key in CARD_FIELDS} != spec.configuration['card']:
            raise ValueError('native adapter geometry, revision or turbo policy differs')
    validate_shared_runtime()


def preflight(options, request, spec):
    check_request(request, prompts=None if spec.configuration['dummy'] else options['prompts'])
    if request['operation'] == 'response':
        native = request['native']
        if set(native) - {'prompt', 'latent', 'timestep'}:
            raise ValueError('response accepts prompt, native latent field and raw timestep')
        if 'prompt' in native and not isinstance(native['prompt'], str):
            raise ValueError('response prompt must be text')
        if 'prompt' in native and not spec.configuration['dummy'] and native['prompt'] not in options['prompts']:
            raise UnsupportedOperation('declare the response prompt before native preparation')


def model_identity(native, *, dummy):
    from .lifecycle import _product_identity
    if dummy:
        # Tiny fixture construction is deterministic and independently creates no
        # teacher rows. Hash its actual frozen tensors, excluding trained leaves.
        import hashlib
        from .pg09 import tensor_identity
        import json
        frozen = {name: tensor_identity(value) for name, value in
                  list(native.model.named_parameters()) + list(native.model.named_buffers())
                  if '.lora_' not in name}
        identity = dict(qualification='tiny-native-CPU-only',
                        frozen_model_sha256=hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest(),
                        conditioning='deterministic random tiny encodings v1')
    else:
        identity = dict(frozen_model_sha256=native.frozen_identity(),
                        conditioning=native.conditioning_identity,
                        vae_source_sha256=native.class_source(native.pipe.vae),
                        pipeline_source_sha256=native.class_source(native.pipe))
    return _product_identity(dict(identity, native_configuration=configuration(native, dummy=dummy)))


def _response_callback(native, adapter):
    def response(*, prompt, latent, timestep, strength, seed):
        import torch
        if prompt not in native.embeddings:
            raise UnsupportedOperation('native response requires a pre-encoded prompt')
        z = torch.as_tensor(latent, device=native.device, dtype=torch.float32)
        t = torch.as_tensor(timestep, device=native.device, dtype=torch.float32)
        expected = native.args.resolution // native.pipe.vae_scale_factor
        if (tuple(z.shape) != (1, native.model.config.in_channels // native.pipe.patch_size**2,
                               1, expected, expected) or not torch.isfinite(z).all()
                or t.numel() != 1 or not torch.isfinite(t).all() or not 0 <= t.item() <= 1000):
            raise ValueError('finite native [1,C,1,H,W] latent and one raw timestep in [0,1000] required')
        embeds, mask = native.embeddings[prompt]
        grid = native.args.resolution // (native.pipe.vae_scale_factor * native.pipe.patch_size)
        context = dict(embeds=embeds, mask=mask,
            position_ids=native.pipe.prepare_position_ids(embeds.shape[1], grid, grid, 'cpu'))
        return adapter.velocity(context, z, t, scale=strength).detach().clone()
    return response


def prepare(options, spec):
    from .lifecycle import _native_arguments
    from particle_sliders.pg09_images import ImageResponseAdapter, image_lora_parameters
    dummy = spec.configuration['dummy']
    if dummy:
        from .synthetic import staged_native
        native = staged_native()
    else:
        from .cuda_train import CudaUNI
        args = _native_arguments(options)
        native = CudaUNI(args, options['prompts'], Path(args.save_dir), write_prompt_counts=False)
    close = own_native(native)
    try:
        identity = model_identity(native, dummy=dummy)
        adapter = ImageResponseAdapter('krea2', native.model, image_lora_parameters(native.model),
            pipeline=native.pipe, scheduler=native.pipe.scheduler, model_identity=identity)
        return InferenceBinding(adapter.network,
            native_identity(adapter, model_identity=identity, protocol=PROTOCOL), close,
            None if dummy else render_callback(native), _response_callback(native, adapter),
            observation=lambda: observation(native))
    except BaseException:
        close()
        raise


BACKEND = InferenceBackend('krea2', capabilities({}), prepare, validate, response=True,
                           configure_capabilities=capabilities, preflight=preflight)


def specification(native, adapter, *, dummy):
    return BACKEND.specification(adapter, model_identity=model_identity(native, dummy=dummy),
                                protocol=PROTOCOL, configuration=configuration(native, dummy=dummy))
