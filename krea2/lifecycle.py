"""Krea2 adapter-GAN factory; preserve native PEFT, caches and sampler policy."""
from pathlib import Path
from particle_sliders.lifecycle import Backend, Binding, Capabilities, checkpoint_cleanup


def validate(options):
    allowed = {'cache_file', 'native_args', 'steps', 'seed', 'batch', 'preset', 'lr',
               'base_precision', 'ema_decay', 'dummy'}
    if set(options) - allowed:
        raise ValueError('unknown Krea2 session option')
    if options.get('dummy'):
        if options.get('base_precision', 'float32') != 'float32':
            raise ValueError('Krea2 tiny fixture is float32 only')
    elif not options.get('cache_file') or not isinstance(options.get('native_args'), list):
        raise ValueError('Krea2 requires local cache_file and native_args (existing native CLI options)')
    import math
    if options.get('preset', 'bispf_optimized') not in ('bispf', 'bispf_optimized'):
        raise ValueError('unsupported Krea2 PG preset')
    if not math.isfinite(options.get('lr', 5e-5)) or options.get('lr', 5e-5) <= 0:
        raise ValueError('Krea2 learning rate must be positive and finite')
    if not options.get('dummy'):
        from .train import parse_args
        from .cuda_train import validate_args
        args = parse_args(options['native_args'])
        if args.allow_hub or args.resume or args.load_te_lora:
            raise ValueError('only local fresh native loading; use session resume/artifact arguments')
        validate_args(args)
    from .shared import validate_shared_runtime
    validate_shared_runtime()


def prepare(options, *, native=None, cache=None):
    import torch
    from .pg09 import build_game, file_digest
    if options.get('dummy'):
        from .synthetic import staged_fixture
        native, cache = staged_fixture()
        checksum = 'deterministic-tiny-Krea2-CPU-fixture-v1'
        identity = {'qualification': 'tiny-native-CPU-only'}
    else:
        from .train import parse_args
        from .cuda_train import CudaUNI, validate_args
        args = parse_args(options['native_args'])
        if args.allow_hub or args.resume or args.load_te_lora:
            raise ValueError('session accepts only local fresh native loading; use session resume/artifact arguments')
        validate_args(args)
        cache = cache if cache is not None else torch.load(options['cache_file'], map_location='cpu', weights_only=True)
        texts = sorted({record[key] for pool in ('edit', 'preserve') for record in cache[pool]
                        for key in ('prompt', 'positive')})
        if native is None:
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)
            native = CudaUNI(args, texts, Path(args.save_dir))
        checksum = file_digest(options['cache_file'])
        identity = cache['key']
        if identity.get('frozen_model_sha256') != native.frozen_identity():
            raise ValueError('Krea2 cache differs from staged frozen model')
    restore = checkpoint_cleanup(native.model)
    def close():
        restore()
        if hasattr(native.model, 'unload'):
            native.pipe.transformer = native.model.unload()
    try:
        game = build_game(native, cache, cache_sha256=checksum, model_identity=identity,
            total_steps=options.get('steps', 4), seed=options.get('seed', 7),
            batch_size=options.get('batch', 2), preset=options.get('preset', 'bispf_optimized'), lr=options.get('lr', 5e-5))
    except BaseException:
        close()
        raise


    def render(*, prompt, strength, seed, images, resolution=768):
        return native.generate(prompt, strength, seed, resolution)
    return Binding(game, close, None if options.get('dummy') else render)


def capabilities(options):
    return Capabilities('adapter_gan', render=not options.get('dummy', False),
                        strength_min=None, strength_max=None,
                        base_precisions=('float32',) if options.get('dummy') else ('bfloat16',))


BACKEND = Backend('krea2', capabilities({}), prepare, validate, capabilities)
