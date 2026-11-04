"""Caller-owned strict numerical policy, scoped RNG and native geometry gates."""
from contextlib import contextmanager
import inspect
import os
import uuid

from . import EXECUTION
LOADER_SEED = 41


def policy():
    import torch
    actual = dict(deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic,
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        fp16_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        bf16_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        flash_sdp=torch.backends.cuda.flash_sdp_enabled(),
        efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(),
        cudnn_sdp=torch.backends.cuda.cudnn_sdp_enabled(), math_sdp=torch.backends.cuda.math_sdp_enabled(),
        math_sdp_low_precision_reduction=torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
        cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
    wanted = dict(deterministic_algorithms=True, deterministic_warn_only=False,
        cudnn_benchmark=False, cudnn_deterministic=True, tf32_matmul=False, tf32_cudnn=False,
        float32_matmul_precision="highest", fp16_reduction=False, bf16_reduction=False,
        flash_sdp=False, efficient_sdp=False, cudnn_sdp=False, math_sdp=True,
        math_sdp_low_precision_reduction=False, cublas_workspace=":4096:8")
    if actual != wanted:
        raise ValueError("Krea2 strict numerical policy differs: " + ", ".join(k for k in wanted if actual[k] != wanted[k]))
    overrides = {k: os.environ.get(k) for k in ("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "NVIDIA_TF32_OVERRIDE")}
    if any(v not in (None, "0") for v in overrides.values()):
        raise ValueError("strict native policy refuses TF32 environment override")
    return dict(profile=EXECUTION, flags=actual, overrides=overrides,
                base_precision="bfloat16 with pinned native FP32 RMSNorm weights",
                trainable_precision="float32", attention="math SDPA only",
                loader_rng=dict(seed=LOADER_SEED, scope="CPU and only owned visible cuda:0"),
                maximum_render_resolution=768, maximum_combined_tokens=4096)


def require_policy(expected):
    if policy() != expected:
        raise ValueError("Krea2 strict execution identity changed")


@contextmanager
def initialization(seed=LOADER_SEED, *, device="cuda:0"):
    import torch
    if device not in ("cpu", "cuda:0") or type(seed) is not int or not 0 <= seed < 2**63 - 100:
        raise ValueError("strict initialization owns only CPU and visible cuda:0")
    with torch.random.fork_rng(devices=[0] if device == "cuda:0" else []):
        torch.random.default_generator.manual_seed(seed)
        if device == "cuda:0":
            torch.cuda.default_generators[0].manual_seed(seed)
        yield


def canonical_gpu_uuid(value):
    if not isinstance(value, str):
        raise ValueError("an explicit GPU UUID string is required")
    try:
        return "GPU-" + str(uuid.UUID(value.removeprefix("GPU-")))
    except ValueError as error:
        raise ValueError("invalid GPU UUID") from error


def admission(expected_gpu_uuid):
    import torch
    expected = canonical_gpu_uuid(expected_gpu_uuid)
    if (os.environ.get("KREA2_STRICT_GUARDIAN") != str(os.getppid())
            or os.environ.get("KREA2_STRICT_PROCESS_CAP_GIB") != "36"
            or canonical_gpu_uuid(os.environ.get("CUDA_VISIBLE_DEVICES")) != expected):
        raise ValueError("strict native loading needs its admitted guardian and exact visible GPU UUID")
    if torch.cuda.device_count() != 1 or canonical_gpu_uuid(str(torch.cuda.get_device_properties(0).uuid)) != expected:
        raise ValueError("strict native actual CUDA UUID differs")
    if torch.cuda.mem_get_info(0)[0] < 41 * 2**30:
        raise MemoryError("native admission requires36GiB cap plus5GiB reserve")


def resolution(value, *, allow_zero=False):
    if type(value) is not int or not (value == 0 and allow_zero or 0 < value <= 768 and value % 16 == 0):
        raise ValueError("strict native resolution must be a multiple of16 through768")


def geometry(latent, context, *, expected_image_tokens=None):
    if (latent.ndim != 3 or context.ndim != 4 or latent.shape[0] != context.shape[0]
            or not 1 <= latent.shape[1] or not 1 <= context.shape[1]
            or latent.shape[1] + context.shape[1] > 4096
            or expected_image_tokens is not None and latent.shape[1] != expected_image_tokens):
        raise ValueError("strict native actual latent/context token geometry exceeds its qualified bounds")


def forward_guards(model, expected):
    """Recheck math and actual input geometry, including checkpoint recomputation."""
    from diffusers.models.transformers.transformer_krea2 import Krea2Attention
    handles = []
    root = model.get_base_model() if hasattr(model, "get_base_model") else model
    signature = inspect.signature(root.forward)
    def check(_module, args, kwargs):
        require_policy(expected)
        values = signature.bind_partial(*args, **kwargs).arguments
        geometry(values["hidden_states"], values["encoder_hidden_states"])
    def attention(_module, args, kwargs):
        require_policy(expected)
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        if hidden is None or hidden.ndim != 3 or not 1 <= hidden.shape[1] <= 4096:
            raise ValueError("strict native attention exceeds4096 actual tokens")
    try:
        handles.append(root.register_forward_pre_hook(check, with_kwargs=True))
        for module in root.modules():
            if isinstance(module, Krea2Attention):
                handles.append(module.register_forward_pre_hook(attention, with_kwargs=True))
    except BaseException:
        for handle in reversed(handles):
            handle.remove()
        raise
    return handles
