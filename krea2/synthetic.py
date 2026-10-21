"""Random tiny native Krea2 fixture; no pretrained or quality qualification."""
from types import SimpleNamespace
import torch
from diffusers import Krea2Pipeline, Krea2Transformer2DModel, FlowMatchEulerDiscreteScheduler
from peft import LoraConfig, get_peft_model
from .cuda_train import CudaUNI


def staged_fixture():
    torch.manual_seed(23)
    model = Krea2Transformer2DModel(in_channels=16, num_layers=1, attention_head_dim=16,
        num_attention_heads=2, num_key_value_heads=1, intermediate_size=64,
        timestep_embed_dim=32, text_hidden_dim=16, num_text_layers=2,
        text_num_attention_heads=2, text_num_key_value_heads=1, text_intermediate_size=32,
        num_layerwise_text_blocks=1, num_refiner_text_blocks=1, axes_dims_rope=(4, 6, 6))
    model.requires_grad_(False)
    model = get_peft_model(model, LoraConfig(r=2, lora_alpha=3,
        target_modules=['to_q', 'to_k', 'to_v', 'to_out.0']))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.get_base_model().enable_gradient_checkpointing()
    pipe = object.__new__(Krea2Pipeline)
    pipe.register_to_config()
    pipe.patch_size, pipe.vae_scale_factor, pipe.transformer = 2, 8, model
    pipe.scheduler = FlowMatchEulerDiscreteScheduler()
    native = CudaUNI.__new__(CudaUNI)
    native.pipe, native.model, native.device, native.dtype = pipe, model, torch.device('cpu'), torch.float32
    native.args = SimpleNamespace(resolution=32)
    embeds = torch.randn(1, 3, 2, 16)
    mask = torch.tensor([[True, True, False]])
    native.embeddings = {'neutral': (embeds, mask), 'positive': (embeds + .2, mask),
                         'hold': (embeds - .2, mask)}
    native.position_ids = {}
    native.params = [p for p in model.parameters() if p.requires_grad]
    cache = dict(edit=[], preserve=[], key={'fixture': 'real-tiny-Krea2-cpu'})
    for holding, pool in ((False, 'edit'), (True, 'preserve')):
        for i in range(2):
            neutral, positive = ('hold', 'hold') if holding else ('neutral', 'positive')
            z = torch.randn(1, 4, 16)
            t = torch.tensor(700. - 300. * i)
            with torch.no_grad(), native.scale(0):
                baseline = native.forward(neutral, z, t).cpu()
                teacher = native.forward(positive, z, t).cpu()
            cache[pool].append(dict(prompt=neutral, positive=positive, z=z, t=t,
                                    baseline=baseline, teacher=teacher))
    return native, cache
