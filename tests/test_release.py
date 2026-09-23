"""Numerical release contracts; optional torch keeps CPU smoke installs small."""
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('safetensors')
from safetensors.torch import load_file
from release_tools.lora import compress_factors, export_lora, projection_error


def test_compression_recovers_known_lower_rank_teacher():
    torch.manual_seed(12)
    a = torch.randn(16, 32)
    b = torch.randn(40, 8) @ torch.randn(8, 16)
    h = torch.randn(300, 32) @ a.T
    down, up, mix, error = compress_factors(a, b, h.T @ h)
    assert error < 1e-10
    assert torch.allclose(up @ down, b @ a, atol=3e-5, rtol=3e-5)
    assert torch.allclose(down, mix @ a, atol=1e-5)
    heldout = torch.randn(80, 32) @ a.T
    numerator, denominator = projection_error(b, up, mix, heldout.T @ heldout)
    assert numerator / denominator < 1e-10


def test_alpha_export_preserves_learned_matrices_and_rejects_overwrite(tmp_path):
    name = 'transformer_blocks.0.attn.to_q'
    a, b = torch.randn(8, 32), torch.randn(32, 8)
    out = export_lora(tmp_path, 'weights', 'fixture', {name: (a, b)}, 12., 'abc', 'test')
    state = load_file(str(tmp_path / out['files']['comfyui']))
    assert torch.equal(state['diffusion_model.blocks.0.attn.wq.lora_down.weight'], a)
    assert torch.equal(state['diffusion_model.blocks.0.attn.wq.lora_up.weight'], b)
    assert float(state['diffusion_model.blocks.0.attn.wq.alpha']) == 12
    export_lora(tmp_path, 'weights', 'fixture', {name: (a, b)}, 12., 'abc', 'test')
    with pytest.raises(FileExistsError):
        export_lora(tmp_path, 'weights', 'fixture', {name: (a, b)}, 16., 'abc', 'test')


def test_diffusers_reads_embedded_alpha(tmp_path):
    diffusers = pytest.importorskip('diffusers')
    model = diffusers.Krea2Transformer2DModel(
        in_channels=4, num_layers=1, num_attention_heads=2, attention_head_dim=16,
        num_key_value_heads=1, intermediate_size=64, text_hidden_dim=16,
        text_intermediate_size=32, text_num_attention_heads=1, text_num_key_value_heads=1,
        num_text_layers=2, num_layerwise_text_blocks=1, num_refiner_text_blocks=1,
        axes_dims_rope=(4, 6, 6))
    name = 'transformer_blocks.0.attn.to_q'
    a, b = torch.randn(8, 32), torch.randn(32, 8)
    out = export_lora(tmp_path, 'weights', 'fixture', {name: (a, b)}, 12., 'abc', 'test')
    state, metadata = diffusers.Krea2Pipeline.lora_state_dict(
        str(tmp_path / out['files']['native']), return_lora_metadata=True)
    diffusers.Krea2Pipeline.load_lora_into_transformer(
        state, transformer=model, adapter_name='fixture', metadata=metadata)
    module = model.transformer_blocks[0].attn.to_q
    assert module.scaling['fixture'] == 1.5
    assert torch.allclose(module.get_delta_weight('fixture'), 1.5 * (b @ a))
