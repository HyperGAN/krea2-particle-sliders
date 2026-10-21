"""Native CPU LoRA checkpoint and sampling state contracts; no Hub or GPU."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("peft")
pytest.importorskip("safetensors")

from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.tuners_utils import BaseTunerLayer

from krea2.cuda_train import CudaUNI


def tiny_transformer():
    return diffusers.Krea2Transformer2DModel(
        in_channels=4, num_layers=1, num_attention_heads=2, attention_head_dim=16,
        num_key_value_heads=1, intermediate_size=64, text_hidden_dim=16,
        text_intermediate_size=32, text_num_attention_heads=1, text_num_key_value_heads=1,
        num_text_layers=2, num_layerwise_text_blocks=1, num_refiner_text_blocks=1,
        axes_dims_rope=(4, 6, 6),
    )


def native_backend(tmp_path):
    torch.manual_seed(29)
    base = tiny_transformer().requires_grad_(False)
    model = get_peft_model(base, LoraConfig(
        r=2, lora_alpha=3, target_modules=["to_q", "to_k"],
    ))
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, BaseTunerLayer):
                module.lora_B["default"].weight.normal_(std=0.1)
    backend = CudaUNI.__new__(CudaUNI)
    backend.model = model
    backend.out = tmp_path
    backend.args = SimpleNamespace(name="alpha-fixture", sample_steps=8)
    backend.device = torch.device("cpu")
    backend.dtype = torch.float32
    return backend


def tuner_layers(model):
    return [module for module in model.modules() if isinstance(module, BaseTunerLayer)]


def test_native_checkpoint_preserves_nonunit_alpha_and_game_state(tmp_path):
    backend = native_backend(tmp_path)
    backend.pipe = SimpleNamespace(save_lora_weights=diffusers.Krea2Pipeline.save_lora_weights)
    game_state = {"completed_steps": 7, "game_fixture": torch.tensor([1.0, 2.0])}
    adapter_state = {"adapter_fixture": torch.tensor([3.0])}
    game = SimpleNamespace(state_dict=lambda: game_state, export_state_dict=lambda: adapter_state)

    dest = backend.checkpoint(7, game)

    assert json.loads((dest / "adapter_config.json").read_text())["lora_alpha"] == 3
    saved_game = torch.load(dest / "training_state.pt", map_location="cpu", weights_only=False)
    assert saved_game["completed_steps"] == 7
    assert torch.equal(saved_game["game_fixture"], game_state["game_fixture"])
    saved_adapter = torch.load(dest / "gan_adapter.pt", map_location="cpu", weights_only=False)
    assert torch.equal(saved_adapter["adapter_fixture"], adapter_state["adapter_fixture"])
    latest = json.loads((tmp_path / "latest_checkpoint.json").read_text())
    assert latest == {"step": 7, "path": str(dest.resolve())}

    peft_roundtrip = PeftModel.from_pretrained(tiny_transformer(), dest)
    state, metadata = diffusers.Krea2Pipeline.lora_state_dict(
        dest, weight_name="alpha-fixture.safetensors", return_lora_metadata=True,
    )
    assert metadata is not None
    native_roundtrip = tiny_transformer()
    diffusers.Krea2Pipeline.load_lora_into_transformer(
        state, transformer=native_roundtrip, adapter_name="fixture", metadata=metadata,
    )

    expected = {
        name.removeprefix("base_model.model."): module.get_delta_weight("default")
        for name, module in backend.model.named_modules() if isinstance(module, BaseTunerLayer)
    }
    for model, adapter in ((peft_roundtrip, "default"), (native_roundtrip, "fixture")):
        actual = {
            name.removeprefix("base_model.model."): module.get_delta_weight(adapter)
            for name, module in model.named_modules() if isinstance(module, BaseTunerLayer)
        }
        assert actual.keys() == expected.keys()
        for name in expected:
            assert torch.equal(actual[name], expected[name]), name

    with pytest.raises(FileExistsError):
        backend.checkpoint(7, game)
    with pytest.raises(ValueError, match="step"):
        backend.checkpoint(8, game)
    assert not (tmp_path / "checkpoint-0008").exists()


def test_scale_restores_heterogeneous_and_nested_strengths(tmp_path):
    backend = native_backend(tmp_path)
    layers = tuner_layers(backend.model)
    for index, layer in enumerate(layers):
        layer.scaling["default"] = (index + 1) / 10
    original = [dict(layer.scaling) for layer in layers]

    with backend.scale(0.8):
        assert [layer.scaling["default"] for layer in layers] == pytest.approx(
            [scales["default"] * 0.8 for scales in original])
        with backend.scale(-0.2):
            assert [layer.scaling["default"] for layer in layers] == pytest.approx(
                [scales["default"] * -0.2 for scales in original])
        assert [layer.scaling["default"] for layer in layers] == pytest.approx(
            [scales["default"] * 0.8 for scales in original])
    assert [layer.scaling for layer in layers] == original

    with pytest.raises(RuntimeError, match="fixture"):
        with backend.scale(0.5):
            raise RuntimeError("fixture")
    assert [layer.scaling for layer in layers] == original


def test_scale_zero_disables_edit_and_restores_frozen_parameters(tmp_path):
    backend = native_backend(tmp_path)
    backend.model.requires_grad_(False)
    layers = tuner_layers(backend.model)
    original_grad = {name: parameter.requires_grad for name, parameter in backend.model.named_parameters()}
    original_disabled = [layer.disable_adapters for layer in layers]
    projection = layers[0]
    inputs = torch.randn(1, 3, projection.in_features)
    with torch.no_grad():
        base = projection.base_layer(inputs)
        assert not torch.equal(projection(inputs), base)
        with backend.scale(0):
            assert torch.equal(projection(inputs), base)

    assert [layer.disable_adapters for layer in layers] == original_disabled
    assert {name: parameter.requires_grad for name, parameter in backend.model.named_parameters()} == original_grad


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("fail", [True, False])
def test_generate_restores_model_mode_after_success_or_error(tmp_path, monkeypatch, training, fail):
    backend = native_backend(tmp_path)
    backend.model.train(training)
    tuner_layers(backend.model)[0].base_layer.train(not training)
    original_modes = {name: module.training for name, module in backend.model.named_modules()}
    backend.embeddings = {"fixture": (torch.zeros(1, 2, 2, 16), torch.ones(1, 2, dtype=torch.bool))}
    sentinel = object()

    class VAE:
        dtype = torch.float32
        config = SimpleNamespace(latents_mean=[0.0], latents_std=[1.0])

        def to(self, device):
            return self

        def decode(self, latents, return_dict):
            return (latents.expand(1, 3, 1, 2, 2),)

    class Pipeline:
        vae = VAE()
        image_processor = SimpleNamespace(postprocess=lambda pixels, output_type: [sentinel])

        def __call__(self, **kwargs):
            assert not backend.model.training
            assert kwargs["guidance_scale"] == 0.0
            assert kwargs["num_inference_steps"] == 8
            assert kwargs["generator"].device.type == "cpu"
            if fail:
                raise RuntimeError("sampling fixture")
            return SimpleNamespace(images=torch.zeros(1, 1, 1, 2, 2))

        def _unpack_latents(self, latents, height, width):
            return latents

    backend.pipe = Pipeline()
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    if fail:
        with pytest.raises(RuntimeError, match="sampling fixture"):
            backend.generate("fixture", 0.5, 42, 32)
    else:
        assert backend.generate("fixture", 0.5, 42, 32) is sentinel
    assert backend.model.training is training
    assert {name: module.training for name, module in backend.model.named_modules()} == original_modes
