"""Memory-efficient masked GQA for large Krea images on Ampere GPUs."""
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_krea2 import Krea2Attention, Krea2AttnProcessor


class LargeImageAttention(Krea2AttnProcessor):
    """Keep the stock path through 768px; avoid quadratic allocation at 1536px.

PyTorch's efficient backend supports masks but not enable_gqa. Expanding the
KV heads expresses the same attention operation with enable_gqa=False.
"""
    def __call__(self, attn, hidden_states, attention_mask=None, image_rotary_emb=None):
        if hidden_states.shape[1] <= 4096 or hidden_states.device.type != 'cuda':
            return super().__call__(attn, hidden_states, attention_mask, image_rotary_emb)
        query = attn.to_q(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))
        key = attn.to_k(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
        query, key = attn.norm_q(query), attn.norm_k(key)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
        query, key, value = (t.transpose(1, 2) for t in (query, key, value))
        repeats = attn.num_heads // attn.num_kv_heads
        if repeats != 1:
            key = key.repeat_interleave(repeats, dim=1)
            value = value.repeat_interleave(repeats, dim=1)
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            result = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)
        result = result.transpose(1, 2).flatten(2, 3)
        return attn.to_out[0](result * torch.sigmoid(attn.to_gate(hidden_states)))


def install_large_image_attention(model):
    for module in model.modules():
        if isinstance(module, Krea2Attention):
            module.set_processor(LargeImageAttention())
