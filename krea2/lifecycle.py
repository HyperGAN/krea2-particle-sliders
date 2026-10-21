"""Krea2 adapter-GAN factory; preserve native PEFT, caches and sampler policy."""
from copy import deepcopy
from pathlib import Path

from particle_sliders.lifecycle import (
    Backend, Binding, Capabilities, UnsupportedOperation, checkpoint_cleanup,
)


def _native_arguments(options):
    from .train import parse_args
    from .cuda_train import validate_args
    values = options.get('native_args')
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise ValueError('native_args must be a list of native CLI strings')
    args = parse_args(values)
    if args.allow_hub or args.resume or args.load_te_lora or args.dummy or args.print_card:
        raise ValueError('only local fresh native loading; use session resume/artifact arguments')
    validate_args(args)
    return args


def validate(options):
    import math
    allowed = {'cache_file', 'native_args', 'steps', 'seed', 'batch', 'preset', 'lr',
               'base_precision', 'ema_decay', 'dummy'}
    if set(options) - allowed:
        raise ValueError('unknown Krea2 session option')
    if type(options.get('dummy', False)) is not bool:
        raise ValueError('dummy must be a boolean')
    if options.get('dummy'):
        if options.get('base_precision', 'float32') != 'float32':
            raise ValueError('Krea2 tiny fixture is float32 only')
        if 'cache_file' in options or 'native_args' in options:
            raise ValueError('tiny fixtures do not accept native cache/CLI options')
    else:
        path = options.get('cache_file')
        if not isinstance(path, str) or not Path(path).is_file():
            raise ValueError('Krea2 requires an existing local cache_file')
        _native_arguments(options)
    if options.get('preset', 'bispf_optimized') not in ('bispf', 'bispf_optimized'):
        raise ValueError('unsupported Krea2 PG preset')
    lr = options.get('lr', 5e-5)
    if type(lr) not in (int, float) or not math.isfinite(lr) or lr <= 0:
        raise ValueError('Krea2 learning rate must be positive and finite')
    from .shared import validate_shared_runtime
    validate_shared_runtime()


def _cache_identity(cache, args):
    """Check native coordinate and sampler claims before constructing a model."""
    if not isinstance(cache, dict) or not isinstance(cache.get('key'), dict):
        raise ValueError('invalid Krea2 teacher cache')
    expected = dict(model_id=args.model_id, transformer_subfolder=args.transformer_subfolder,
                    revision=args.revision, skeleton=args.skeleton_model,
                    skeleton_revision=args.skeleton_revision, rank=args.rank,
                    resolution=args.resolution, sample_steps=args.sample_steps,
                    guidance=args.sample_guidance, mu=args.mu, particles=None,
                    objective='g_adv;d_adv+public_penalty')
    for key, value in expected.items():
        if key not in cache['key'] or cache['key'][key] != value:
            raise ValueError('Krea2 cache differs from native configuration: ' + key)
    for pool in ('edit', 'preserve'):
        if not isinstance(cache.get(pool), list) or not cache[pool]:
            raise ValueError('nonempty Krea2 edit and preservation cache pools required')
    return deepcopy(cache['key'])


def _product_identity(identity):
    """Bind callbacks/helpers as well as the common factory and PG sources."""
    from importlib.metadata import version
    from .pg09 import file_digest
    return dict(identity, lifecycle_product=dict(
        sources={p.name: file_digest(p) for p in sorted(Path(__file__).parent.iterdir())
                 if p.suffix in ('.py', '.json')},
        versions={name: version(name) for name in ('diffusers', 'peft', 'transformers', 'safetensors')}))


def prepare(options, *, native=None, cache=None):
    import torch
    from .pg09 import build_game, file_digest
    if options.get('dummy'):
        from .synthetic import staged_fixture
        native, cache = staged_fixture()
        checksum = 'deterministic-tiny-Krea2-CPU-fixture-v1'
        identity = {'qualification': 'tiny-native-CPU-only'}
    else:
        from .cuda_train import CudaUNI
        args = _native_arguments(options)
        cache = cache if cache is not None else torch.load(options['cache_file'], map_location='cpu', weights_only=True)
        identity = _cache_identity(cache, args)
        checksum = file_digest(options['cache_file'])
        texts = sorted({record[key] for pool in ('edit', 'preserve') for record in cache[pool]
                        for key in ('prompt', 'positive')})
        if native is None:
            # Common sessions own their packet output destinations. Preparing a
            # session must not write into the legacy CLI's save_dir.
            native = CudaUNI(args, texts, Path(args.save_dir), write_prompt_counts=False)
    # Ownership transfers only after successful native construction. Any later
    # identity/build failure must release the attached PEFT and checkpoint hooks.
    restore = checkpoint_cleanup(native.model)
    modes = [(module, module.training) for module in native.model.modules()]
    closed = False

    def close():
        nonlocal closed
        if closed:
            return
        closed = True
        try:
            restore()
            for module, training in modes:
                module.training = training
        finally:
            if hasattr(native.model, 'unload'):
                native.pipe.transformer = native.model.unload()

    try:
        if not options.get('dummy') and identity.get('frozen_model_sha256') != native.frozen_identity():
            raise ValueError('Krea2 cache differs from staged frozen model')
        game = build_game(native, cache, cache_sha256=checksum, model_identity=_product_identity(identity),
            total_steps=options.get('steps', 4), seed=options.get('seed', 7),
            batch_size=options.get('batch', 2), preset=options.get('preset', 'bispf_optimized'), lr=options.get('lr', 5e-5))
    except BaseException:
        close()
        raise

    def render(*, prompt, strength, seed, images, resolution=768, output_type='pil'):
        if images or output_type != 'pil':
            raise UnsupportedOperation('Krea2 native rendering accepts PIL output without references')
        if prompt not in native.embeddings:
            raise UnsupportedOperation('Krea2 rendering requires a cached prompt; its text encoder is released')
        if type(resolution) is not int or resolution <= 0 or resolution % 16:
            raise ValueError('native Krea2 resolution must be a positive multiple of 16')
        # Native pipeline sampling mutates its scheduler cursor. Use a temporary
        # native scheduler, restoring the actual prepared object even on failure.
        scheduler = native.pipe.scheduler
        try:
            native.pipe.scheduler = deepcopy(scheduler)
            return native.generate(prompt, strength, seed, resolution)
        finally:
            native.pipe.scheduler = scheduler
    return Binding(game, close, None if options.get('dummy') else render)


def capabilities(options):
    return Capabilities('adapter_gan', render=not options.get('dummy', False),
                        strength_min=None, strength_max=None, render_outputs=('pil',),
                        base_precisions=('float32',) if options.get('dummy') else ('bfloat16',))


BACKEND = Backend('krea2', capabilities({}), prepare, validate, capabilities)
