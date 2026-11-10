"""Owner-injected Krea2 opt-in factory; original core and native math unchanged.

No registration, model download, teacher capture, core patch or old-packet
migration. The caller supplies a fresh admitted original PEFT native owner.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile

import torch

from particle_sliders.lifecycle import Binding, SessionConfig, SliderSession, _check_packet
from particle_sliders.inference import (
    InferenceBackend, InferenceBinding, export_inference, native_identity,
    open_inference as core_open_inference,
)
from particle_sliders.pg09 import ConditionalPG09
from particle_sliders.pg09_images import ImageResponseAdapter, image_lora_parameters
from particle_sliders.field_critic import FieldCritic
from krea2 import lifecycle as original
from krea2 import native_inference as original_inference
from krea2.native_runtime import configuration, observation, render_callback
from krea2.pg09 import Krea2Rows, file_digest
from krea2.shared import validate_shared_runtime

from . import PROFILE, policy
from .ownership import Components, NativeOwner

NAME = 'krea2-explicit-fresh95-v1'
PROTOCOL = 'krea2-explicit-fresh95-RAW-native-v1'


def source_identity():
    from . import ownership
    from krea2 import pg09, cuda_train, native_runtime, native_inference, learning_rates
    from particle_sliders import field_critic, pg09_images
    from particlegan import init
    return {name: policy.source_record(module.__file__) for name, module in (
        ('fresh_factory', __import__(__name__, fromlist=['__file__'])),
        ('fresh_policy', policy), ('fresh_ownership', ownership),
        ('original_factory', original), ('original_rows', pg09),
        ('original_native_loader', cuda_train), ('original_native_runtime', native_runtime),
        ('original_inference', native_inference), ('original_rates', learning_rates),
        ('field_critic', field_critic), ('original_image_adapter', pg09_images), ('PG095_init', init))}


@dataclass(frozen=True)
class Checkpoint:
    path: str
    bytes: int
    sha256: str


def file_pin(path):
    path = Path(path).absolute()
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        after = os.fstat(stream.fileno())
    if _stat(before) != _stat(after):
        raise ValueError('packet changed during whole-file pin')
    return Checkpoint(str(path), before.st_size, sha)


def _stat(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _read(pin):
    if (not isinstance(pin, Checkpoint) or type(pin.bytes) is not int or pin.bytes < 1
            or len(pin.sha256) != 64 or any(c not in '0123456789abcdef' for c in pin.sha256)):
        raise ValueError('explicit whole-file Checkpoint required')
    with Path(pin.path).open('rb') as stream:
        before = os.fstat(stream.fileno())
        if before.st_size != pin.bytes or hashlib.file_digest(stream, 'sha256').hexdigest() != pin.sha256:
            raise ValueError('whole-file packet identity differs')
        stream.seek(0)
        packet = torch.load(stream, map_location='cpu', weights_only=True)
        if _stat(before) != _stat(os.fstat(stream.fileno())):
            raise ValueError('packet changed during typed read')
    return packet


def equal_state(left, right):
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.dtype == right.dtype
                and left.shape == right.shape and torch.equal(left.detach().cpu(), right.detach().cpu()))
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal_state(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(equal_state(a, b) for a, b in zip(left, right))
    return left == right


def _write(path, packet):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            torch.save(packet, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Never replace an old FULL/RAW.
        os.unlink(temporary)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return file_pin(path)


def _receipt(receipt, cohort, options, sources):
    if (not isinstance(receipt, dict) or receipt.get('schema') != policy.SCHEMA
            or receipt.get('policy') != policy.POLICIES['krea2'] or receipt.get('product') != 'krea2'
            or receipt.get('cohort_id') != cohort or receipt.get('seed') != options.get('seed', 7)
            or receipt.get('source_identity') != sources or receipt.get('completed_steps') != 0
            or receipt.get('restored_from') is not None or receipt.get('particles') is not None
            or receipt.get('parameter_identity_preserved') is not True
            or receipt.get('selection') != 'raw' or receipt.get('ema_decay') is not None):
        raise ValueError('original new-profile initialization receipt required; no old cohort transfer')
    return deepcopy(receipt)


class ReceiptBackend:
    def __init__(self, backend, receipt, sources):
        self.backend, self.network = backend, backend.network
        self.receipt, self.sources = deepcopy(receipt), deepcopy(sources)

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def metadata(self):
        return dict(self.backend.metadata(), fresh_initialization=dict(
            profile=PROFILE, receipt=deepcopy(self.receipt), sources=deepcopy(self.sources)))


def _admit(mode, cohort, options, factory, admission):
    if not callable(factory) or not callable(admission):
        raise ValueError('explicit owner_factory and owner_admission required before construction')
    request = dict(operation=mode, profile=PROFILE, cohort_id=cohort, options=deepcopy(options),
                   selection='raw', ema_decay=None)
    if admission(deepcopy(request)) is not True:
        raise ValueError('owner refused construction')
    device = torch.device(options.get('device', 'cpu'))
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
        components = factory(deepcopy(request))
    if not isinstance(components, Components):
        raise TypeError('owner_factory must return Components; it owns pre-return failure cleanup')
    return components


def _native_identity(native, dummy):
    return original_inference.model_identity(native, dummy=dummy)


def _check_native(native, options, cache):
    if native.pipe.transformer is not native.model:
        raise ValueError('original native pipe.transformer ownership differs')
    if not options.get('dummy'):
        args = original._native_arguments(options)
        identity = original._cache_identity(cache, args)
        if (identity.get('frozen_model_sha256') != native.frozen_identity()
                or identity['conditioning_identity'] != native.conditioning_identity):
            raise ValueError('original frozen/cache/conditioning identity differs')
        return original._product_identity(identity)
    return original._product_identity(dict(qualification='tiny-native-CPU-only'))


def local_components(request):
    """Concrete original local loader for callers with external owner admission.

    Passed as owner_factory, so the API has already admitted this request. No
    fresh teacher capture or legacy trainer runs. Only an explicitly transferred
    PEFT owner is unloaded at close; foreign pipe.transformer replacements stay.
    """
    options, mode = request['options'], request['operation']
    if request.get('profile') != PROFILE or mode not in ('fresh', 'resume', 'raw-inference'):
        raise ValueError('this factory requires the versioned explicit construction request')
    if options.get('dummy'):
        raise ValueError('local_components is native only; tiny tests declare their own software factory')
    args = original._native_arguments(options)
    if mode == 'raw-inference':
        cache, checksum = None, None
        texts = options['prompts']
    else:
        cache = torch.load(options['cache_file'], map_location='cpu', weights_only=True)
        original._cache_identity(cache, args)  # Original pre-allocation geometry guard.
        checksum = file_digest(options['cache_file'])
        texts = sorted({r[key] for pool in ('edit', 'preserve') for r in cache[pool]
                        for key in ('prompt', 'positive')} | set(options.get('render_prompts', [])))
    from krea2.cuda_train import CudaUNI
    native = CudaUNI(args, texts, Path(args.save_dir), write_prompt_counts=False)
    model = native.model
    released = False
    pending_base = []
    def release():
        nonlocal released
        if released:
            return
        if native.pipe.transformer is not model:
            raise RuntimeError('foreign pipe.transformer retained; owned unload remains pending')
        if not pending_base:
            pending_base.append(model.unload())
        native.pipe.transformer = pending_base[0]
        released = True
    return Components(native, cache, checksum, release)


def _adapter(native, identity):
    return ImageResponseAdapter('krea2', native.model, image_lora_parameters(native.model),
        pipeline=native.pipe, scheduler=native.pipe.scheduler, model_identity=identity)


def _entry_failed(primary, owner):
    try:
        owner.close()
    except BaseException as cleanup:
        failure = BaseExceptionGroup('Krea2 entry and owned cleanup failed', [primary, cleanup])
        failure.cleanup_owner = owner
        raise failure


class OwnedTraining:
    def __init__(self, session, owner, receipt, sources):
        self.session, self.owner = session, owner
        self.initialization, self.sources = deepcopy(receipt), deepcopy(sources)

    @property
    def game(self):
        return self.session.game

    @property
    def manifest(self):
        return self.session.manifest

    def step(self):
        self.owner.check()
        return self.session.step()

    def checkpoint(self, path):
        self.owner.check()
        return _write(path, self.session.checkpoint())

    def export(self, path):
        self.owner.check()
        return _write(path, export_inference(self.session, selection='raw'))

    def close(self):
        self.owner.close()
        self.session.closed = True  # Only after every owned cleanup succeeded.

    def __enter__(self):
        self.owner.check()
        return self

    def __exit__(self, kind, error, traceback):
        try:
            self.close()
        except BaseException as cleanup:
            if error is None:
                raise
            raise BaseExceptionGroup('Krea2 body and cleanup failed', [error, cleanup])


def open_training(options, *, mode, cohort_id, owner_factory, owner_admission,
                  resume=None, before_update=None):
    """Explicit fresh|resume before any factory/component; no native teacher calls.

    owner_factory(request) returns an admitted original CudaUNI + original cache.
    This API constructs original rows/adapter/FieldCritic and the original game.
    The factory must use a newly constructed PEFT adapter on resume as well;
    initializer is skipped and the entire typed saved state supplies its values.
    """
    if (mode not in ('fresh', 'resume') or not isinstance(cohort_id, str) or not cohort_id.strip()
            or (mode == 'fresh' and resume is not None) or (mode == 'resume' and resume is None)):
        raise ValueError('explicit fresh|resume and matching whole FULL intent required')
    options = deepcopy(options)
    if options.get('ema_decay') is not None:
        raise ValueError('weight EMA is disabled')
    options['ema_decay'] = None
    for key, default, low in (('steps', 4, 1), ('batch', 2, 2), ('seed', 7, 0)):
        if type(options.get(key, default)) is not int or options.get(key, default) < low:
            raise ValueError('invalid original budget: ' + key)
    if options.get('batch', 2) > 512 or options.get('seed', 7) >= 2**63 - 100:
        raise ValueError('original PG budget exceeded')
    if before_update is not None and not callable(before_update):
        raise TypeError('before_update must be callable')
    capabilities = original.capabilities(options)
    capabilities.check('train', strength=1., references=False,
                       precision=options.get('base_precision'), output_type='pil')
    original.validate(options)  # Exact core0.13/PG0.9.5 Source lock unchanged.
    original.BACKEND._check_pg(options, capabilities)  # Original profile/domain gate too.
    config, sources = SessionConfig(NAME, cohort_id, **options), source_identity()
    packet = _read(resume) if mode == 'resume' else None
    receipt = None
    if packet is not None:
        _check_packet(packet)
        if (packet['schema'] != SliderSession.training_schema
                or packet['manifest']['config'] != config.as_dict()
                or packet['payload']['ema'] != dict(decay=None, values=None)):
            raise ValueError('FULL new-profile config/noEMA differs')
        fresh = packet['manifest']['game_signature']['backend']['fresh_initialization']
        if fresh['profile'] != PROFILE or fresh['sources'] != sources:
            raise ValueError('FULL original fresh Source profile differs')
        receipt = _receipt(fresh['receipt'], cohort_id, options, sources)
    components = _admit(mode, cohort_id, options, owner_factory, owner_admission)
    owner = NativeOwner(components)
    native = components.native
    try:
        if components.cache is None or not isinstance(components.cache_sha256, str) or not components.cache_sha256:
            raise ValueError('original cache and explicit cache content identity required')
        if not options.get('dummy') and components.cache_sha256 != file_digest(options['cache_file']):
            raise ValueError('original cache file content identity differs')
        identity = _check_native(native, options, components.cache)
        rows = Krea2Rows(native, components.cache, cache_sha256=components.cache_sha256)
        adapter = _adapter(native, dict(identity, frozen_cache=rows.identity()))
        # Preserve original native.params list object by not assigning it here.
        devices = [native.device.index or 0] if native.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(options.get('seed', 7))
            critic = FieldCritic(rows.response_shape[1], rows.condition_width).to(native.device)
        if mode == 'fresh':
            receipt = policy.initialize('krea2', adapter.network, critic, model=native.model,
                construction=policy.Construction('fresh', cohort_id,
                    hashlib.sha256(policy.canonical(options).encode()).hexdigest()),
                options=options, seed=options.get('seed', 7), source_identity=sources, optimizers=())
        owner.seal()
        backend = ReceiptBackend(adapter, receipt, sources)
        # First optimizer construction, only after receipt and component init.
        from krea2.learning_rates import recipe_overrides
        game = ConditionalPG09(backend, rows, critic, particles=None,
            response_domain='continuous_field', response_scale=rows.response_scale,
            preset=options.get('preset', 'bispf_optimized'), batch_size=options.get('batch', 2),
            seed=options.get('seed', 7), total_steps=options.get('steps', 4), sampling='balanced',
            recipe_overrides=recipe_overrides(lr=options.get('lr', 5e-5), d_lr_mult=options.get('d_lr_mult')))
        dummy = bool(options.get('dummy'))
        native_identity_value = _native_identity(native, dummy)
        specification = _inference_backend().specification(backend,
            model_identity=dict(native_identity_value, fresh_initialization=deepcopy(receipt)),
            protocol=PROTOCOL, configuration=dict(original=configuration(native, dummy=dummy),
                profile=PROFILE, cohort_id=cohort_id, options=deepcopy(options),
                initialization=deepcopy(receipt), sources=deepcopy(sources)))
        def admission(request):
            owner.check()
            if before_update is not None:
                return before_update(request)
        binding = Binding(game, owner.close, None if dummy else render_callback(native),
                          inference=specification, before_update=admission,
                          observation=lambda: observation(native))
        session = SliderSession(config, capabilities, binding)
        session._manifest['factory_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if mode == 'resume':
            session.restore(packet)  # Original production complete validation/apply.
            if not equal_state(session.checkpoint(), packet):
                raise RuntimeError('whole FULL manifest/game/optimizers/streams/RNG/last_metrics mismatch')
        return OwnedTraining(session, owner, receipt, sources)
    except BaseException as error:
        _entry_failed(error, owner)
        raise


def _inference_capabilities(config):
    return original_inference.capabilities(config['original'])


def _validate_inference(options, spec):
    config = spec.configuration
    if (set(config) != {'original', 'profile', 'cohort_id', 'options', 'initialization', 'sources'}
            or config['profile'] != PROFILE or config['sources'] != source_identity()
            or config['options'].get('ema_decay') is not None):
        raise ValueError('explicit new-profile RAW configuration differs')
    _receipt(config['initialization'], config['cohort_id'], config['options'], config['sources'])
    from types import SimpleNamespace
    original_inference.validate(options, SimpleNamespace(configuration=config['original']))


def _inference_preflight(options, request, spec):
    from types import SimpleNamespace
    original_inference.preflight(options, request, SimpleNamespace(configuration=spec.configuration['original']))


def _prepare_inference(options, spec, *, components, owner=None):
    """No rows, FieldCritic, game, optimizer or policy.initialize on this route."""
    native, config = components.native, spec.configuration
    owner = NativeOwner(components) if owner is None else owner
    try:
        identity = _native_identity(native, config['original']['dummy'])
        adapter = _adapter(native, identity)
        owner.seal()
        backend = ReceiptBackend(adapter, config['initialization'], config['sources'])
        return InferenceBinding(backend.network,
            native_identity(backend, model_identity=dict(identity, fresh_initialization=config['initialization']), protocol=PROTOCOL),
            owner.close, None if config['original']['dummy'] else render_callback(native),
            original_inference._response_callback(native, adapter), observation=lambda: observation(native))
    except BaseException as error:
        _entry_failed(error, owner)
        raise


def _inference_backend(prepare=_prepare_inference):
    return InferenceBackend(NAME, original_inference.capabilities({'dummy': False}),
        prepare, _validate_inference, configure_capabilities=_inference_capabilities,
        preflight=_inference_preflight)


class OwnedInference:
    def __init__(self, session, owner):
        self.session, self.owner = session, owner

    def response(self, **request):
        self.owner.check()
        return self.session.response(**request)

    def render(self, **request):
        self.owner.check()
        return self.session.render(**request)

    def close(self):
        self.owner.close()
        self.session.closed = True

    __enter__ = OwnedTraining.__enter__
    __exit__ = OwnedTraining.__exit__


def open_inference(raw, options, *, mode, owner_factory, owner_admission):
    if mode != 'raw-inference':
        raise ValueError('explicit raw-inference mode required before native construction')
    packet = _read(raw)
    from particle_sliders.inference import _validate_packet
    _validate_packet(packet)  # Validate the same bracketed read; no second file-read race.
    if packet.get('selection') != 'raw':
        raise ValueError('exact RAW packet required; no EMA')
    from particle_sliders.inference import InferenceSpec
    specification = InferenceSpec(packet['specification'])
    _validate_inference(options, specification)
    components = _admit(mode, specification.configuration['cohort_id'], options, owner_factory, owner_admission)
    owner = NativeOwner(components)
    def prepare(local_options, local_spec):
        return _prepare_inference(local_options, local_spec, components=components, owner=owner)
    # Both functions live in this exact file: the core's strict factory Source
    # digest is identical; no captured core/native Source identity is overridden.
    try:
        session = core_open_inference(packet, options=options, backend=_inference_backend(prepare), operation='response')
        return OwnedInference(session, owner)
    except BaseException as error:
        if not owner.closing:
            _entry_failed(error, owner)
        else:
            error.cleanup_owner = owner
        raise
