"""Data-independent strict native inference; refuses stale identity before load."""
from pathlib import Path

from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity
from krea2.native_runtime import CARD_FIELDS, configuration, render_callback
from krea2.native_inference import _response_callback, model_identity as base_identity, preflight as base_preflight
from krea2.lifecycle import _validate_prompts

from . import BACKEND_NAME
from .lifecycle import CAPABILITIES, arguments
from .provenance import execution_identity
from .execution import resolution

PROTOCOL = "krea2-native-strict-adapter-v1"


def model_identity(native):
    return dict(base_identity(native, dummy=False), strict_execution=native.strict_execution,
                strict_assets=native.strict_assets)


def specification(native, adapter):
    return BACKEND.specification(adapter, model_identity=model_identity(native), protocol=PROTOCOL,
        configuration=dict(card=configuration(native, dummy=False)["card"], execution=native.strict_execution,
                           assets=native.strict_assets))


def validate(options, spec):
    if set(options) - {"native_args", "device", "base_precision", "prompts", "expected_gpu_uuid"}:
        raise ValueError("strict inference accepts only native loading/device/prompt options")
    if options.get("base_precision", "bfloat16") != "bfloat16":
        raise ValueError("strict inference requires BF16")
    config = spec.configuration
    if set(config) != {"card", "execution", "assets"} or set(config["card"]) != set(CARD_FIELDS):
        raise ValueError("strict native inference configuration differs")
    if spec.identity["protocol"] != PROTOCOL:
        raise ValueError("strict native inference protocol differs")
    args = arguments(options)
    _validate_prompts(options.get("prompts"), required=True)
    from .assets import require_args
    assets = require_args(args)
    assets["full_weight_hashes_verified"] = True
    execution = execution_identity()
    if (config["execution"] != execution or config["assets"] != assets
            or {k: getattr(args, k) for k in CARD_FIELDS} != config["card"]
            or spec.identity["model"].get("strict_execution") != execution
            or spec.identity["model"].get("strict_assets") != assets):
        raise ValueError("strict native source/numerical/assets/card differ before base allocation")
    from .execution import canonical_gpu_uuid
    canonical_gpu_uuid(options.get("expected_gpu_uuid"))


def preflight(options, request, spec):
    # The old route's request checks have no model allocation or training state.
    from types import SimpleNamespace
    base_preflight(options, request, SimpleNamespace(configuration={"dummy": False}))
    if request["operation"] == "render":
        resolution(request["native"].get("resolution", 768))


def prepare(options, spec):
    from particle_sliders.pg09_images import ImageResponseAdapter, image_lora_parameters
    from .native import StrictCudaUNI, observation
    args = arguments(options)
    native = StrictCudaUNI(args, options["prompts"], Path(args.save_dir), expected_gpu_uuid=options["expected_gpu_uuid"])
    adapter = None
    try:
        identity = model_identity(native)
        adapter = ImageResponseAdapter("krea2", native.model, image_lora_parameters(native.model),
            pipeline=native.pipe, scheduler=native.pipe.scheduler, model_identity=identity)
        return InferenceBinding(adapter.network, native_identity(adapter, model_identity=identity, protocol=PROTOCOL),
            lambda: native.strict_close(network=adapter.network), render_callback(native), _response_callback(native, adapter),
            observation=lambda: observation(native))
    except BaseException:
        native.strict_close(network=adapter.network if adapter is not None else None)
        raise


BACKEND = InferenceBackend(BACKEND_NAME, CAPABILITIES, prepare, validate, response=True, preflight=preflight)
