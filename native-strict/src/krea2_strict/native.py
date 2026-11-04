"""Strict ownership wrapper around the pinned native Krea2 loader and sampler."""
from contextlib import contextmanager
from pathlib import Path

from krea2.cuda_train import CudaUNI
from krea2.native_runtime import observation as base_observation, own_native

from .assets import inspect_assets, require_args
from .execution import admission, forward_guards, geometry, initialization, require_policy, resolution
from .provenance import execution_identity


def validate_geometry(native):
    import torch
    resolution(native.args.resolution)
    grid = native.args.resolution // (native.pipe.vae_scale_factor * native.pipe.patch_size)
    if grid <= 0:
        raise ValueError("invalid native latent grid")
    for embeds, mask in native.embeddings.values():
        if (embeds.ndim != 4 or mask.ndim != 2 or tuple(mask.shape) != tuple(embeds.shape[:2])
                or embeds.shape[0] != 1 or not 1 <= embeds.shape[1] or grid**2 + embeds.shape[1] > 4096
                or tuple(embeds.shape[2:]) != (native.model.config.num_text_layers, native.model.config.text_hidden_dim)
                or mask.dtype != torch.bool or not torch.isfinite(embeds).all()):
            raise ValueError("strict native actual encoded context/grid exceeds4096 tokens")


def audit_loaded(native):
    import torch
    from diffusers import Krea2Transformer2DModel, Krea2Pipeline
    from krea2.attention import LargeImageAttention
    from diffusers.models.transformers.transformer_krea2 import Krea2Attention, Krea2RMSNorm
    root = native.model.get_base_model()
    if type(root) is not Krea2Transformer2DModel or type(native.pipe) is not Krea2Pipeline:
        raise ValueError("strict native actual transformer/pipeline class differs")
    leaves, fp32_norms = [], []
    norm_weights = {id(module.weight) for module in root.modules() if isinstance(module, Krea2RMSNorm)}
    for name, parameter in native.model.named_parameters():
        trainable = ".lora_A." in name or ".lora_B." in name
        expected_dtype = torch.float32 if trainable or id(parameter) in norm_weights else torch.bfloat16
        if (parameter.device != torch.device("cuda:0") or parameter.requires_grad != trainable
                or parameter.dtype != expected_dtype):
            raise ValueError("strict native actual trainable/base precision/device differs: " + name)
        if trainable:
            leaves.append(name)
        elif id(parameter) in norm_weights:
            fp32_norms.append(name)
    if not leaves:
        raise ValueError("strict native actual LoRA leaves missing")
    attention_count = 0
    for module in root.modules():
        if isinstance(module, Krea2Attention):
            attention_count += 1
            if type(module.processor) is not LargeImageAttention:
                raise ValueError("strict native actual attention implementation differs")
    if not attention_count:
        raise ValueError("strict native attention modules missing")
    if any(embeds.device.type != "cpu" or embeds.dtype != torch.bfloat16
           or mask.device.type != "cpu" for embeds, mask in native.embeddings.values()):
        raise ValueError("strict native actual conditioning precision/device differs")
    return dict(actual_transformer="diffusers.Krea2Transformer2DModel", actual_pipeline="diffusers.Krea2Pipeline",
                lora_leaves=leaves, frozen_fp32_native_norms=fp32_norms, native_attention_modules=attention_count,
                device=dict(name=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0))),
                attention="pinned LargeImageAttention stock path only, actual combined tokens<=4096")


class StrictCudaUNI(CudaUNI):
    def module_identity(self, module):
        # The inherited constructor invokes this on the actual frozen text
        # encoder before encoding or releasing it. Bind its resolved attention
        # and dtype rather than inferring them from package metadata.
        import torch
        from transformers import Qwen3VLForConditionalGeneration
        if type(module) is not Qwen3VLForConditionalGeneration:
            raise ValueError("strict native actual text encoder class differs")
        configs = (module.config, module.config.text_config, module.config.vision_config)
        attention = [getattr(config, "_attn_implementation", None) for config in configs]
        if any(value not in ("sdpa", "eager") for value in attention):
            raise ValueError("strict native actual encoder attention implementation differs")
        if any(p.dtype != torch.bfloat16 or p.requires_grad for p in module.parameters()):
            raise ValueError("strict native actual frozen encoder precision differs")
        self.strict_encoder = dict(native_class="transformers.Qwen3VLForConditionalGeneration",
                                   attention=attention, dtype="bfloat16")
        return CudaUNI.module_identity(module)

    def __init__(self, args, texts, out, *, expected_gpu_uuid):
        require_args(args)
        for name in ("resolution", "sample_resolution", "final_resolution"):
            resolution(getattr(args, name), allow_zero=name == "final_resolution")
        if args.device != 0:
            raise ValueError("strict native route owns only visible cuda:0")
        self.strict_execution = execution_identity()
        self.strict_assets = inspect_assets(args.transformer, args.skeleton_model, full_hashes=True)
        admission(expected_gpu_uuid)
        self._strict_handles = []
        try:
            # Includes frozen construction, prompt encoding and random PEFT
            # initialization. No torch.manual_seed/manual_seed_all is called.
            with initialization():
                super().__init__(args, texts, Path(out), write_prompt_counts=False)
            require_policy(self.strict_execution["numerical"])
            validate_geometry(self)
            self.strict_loaded = audit_loaded(self)
            self.conditioning_identity = dict(self.conditioning_identity,
                strict_execution=self.strict_execution, strict_assets=self.strict_assets,
                strict_loaded=self.strict_loaded, strict_encoder=self.strict_encoder)
            self._strict_handles = forward_guards(self.model, self.strict_execution["numerical"])
        except BaseException:
            self.strict_close()
            raise

    def strict_close(self):
        for handle in reversed(self._strict_handles):
            handle.remove()
        self._strict_handles.clear()
        if hasattr(self, "model") and hasattr(self, "pipe"):
            if not getattr(self, "_strict_closed", False):
                self._strict_closed = True
                own_native(self)()

    def forward(self, *args, **kwargs):
        require_policy(self.strict_execution["numerical"])
        return super().forward(*args, **kwargs)

    def generate(self, prompt, scale, seed, image_resolution):
        require_policy(self.strict_execution["numerical"])
        resolution(image_resolution)
        return super().generate(prompt, scale, seed, image_resolution)


@contextmanager
def observation(native):
    require_policy(native.strict_execution["numerical"])
    with base_observation(native):
        yield
    require_policy(native.strict_execution["numerical"])


def check_cache_geometry(native, cache):
    validate_geometry(native)
    grid = native.args.resolution // (native.pipe.vae_scale_factor * native.pipe.patch_size)
    for pool in ("edit", "preserve"):
        for row in cache[pool]:
            if row["prompt"] not in native.embeddings or row["positive"] not in native.embeddings:
                raise ValueError("strict native cache prompt was not encoded")
            for prompt in (row["prompt"], row["positive"]):
                geometry(row["z"], native.embeddings[prompt][0], expected_image_tokens=grid**2)
