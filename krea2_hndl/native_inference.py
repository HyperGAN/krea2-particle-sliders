"""Independent native-only loading for the explicit Krea2 HNDL cohort."""
from particle_sliders.inference import InferenceBackend
from krea2 import native_inference as product
from krea2.native_runtime import configuration
from .runtime import native_factory, validate_runtime

PROTOCOL = 'krea2-hypergan-hndl-native-v1'


def validate(options, spec):
    product.validate(options, spec, runtime_validator=validate_runtime)


def prepare(options, spec):
    return product.prepare(options, spec, native_factory=native_factory, protocol=PROTOCOL,
                           identity_factory=model_identity)


def model_identity(native, *, dummy):
    import hashlib
    from pathlib import Path
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(Path(__file__).parent.iterdir()) if p.suffix in ('.py', '.json')}
    return dict(product.model_identity(native, dummy=dummy), candidate_binding=sources,
                shared_runtime=validate_runtime())


BACKEND = InferenceBackend('krea2-hndl', product.capabilities({}), prepare, validate,
    response=True, configure_capabilities=product.capabilities, preflight=product.preflight)


def specification(native, adapter, *, dummy):
    return BACKEND.specification(adapter, model_identity=model_identity(native, dummy=dummy),
        protocol=PROTOCOL, configuration=configuration(native, dummy=dummy))
