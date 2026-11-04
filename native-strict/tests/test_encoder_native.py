"""Exact pinned native encoder class; only meta or tiny random-weight models."""
import json
from pathlib import Path
import pytest
import torch
from krea2_strict.execution import initialization
from krea2_strict.native import StrictCudaUNI

SNAPSHOT=Path('/mnt/nvme/ai/caches/huggingface/main/hub/models--krea--Krea-2-Raw/snapshots/6b0ece7fffb640c5e3bcbe0a7f10f66b8e60a603')


def packet():
    return json.loads((SNAPSHOT/'text_encoder/config.json').read_text())


def tiny_config():
    from transformers import Qwen3VLConfig
    value=packet()
    value['text_config'].update(hidden_size=16,intermediate_size=32,num_hidden_layers=36,
        num_attention_heads=2,num_key_value_heads=1,head_dim=8,max_position_embeddings=2048,
        rope_parameters=dict(rope_type='default',rope_theta=5000000.,mrope_interleaved=True,mrope_section=[1,1,2]))
    value['vision_config'].update(depth=2,hidden_size=16,intermediate_size=32,num_heads=2,
        out_hidden_size=16,num_position_embeddings=16,deepstack_visual_indexes=[],patch_size=2,
        spatial_merge_size=1,temporal_patch_size=1)
    return Qwen3VLConfig(**value)


@pytest.fixture
def actual_encoder(tmp_path,strict_flags):
    from transformers import Qwen3VLModel
    with initialization(device='cpu'):
        model=Qwen3VLModel(tiny_config()).to(torch.bfloat16).eval().requires_grad_(False)
        model.save_pretrained(tmp_path/'tiny-native')
    del model
    with initialization(device='cpu'):
        loaded=Qwen3VLModel.from_pretrained(tmp_path/'tiny-native',dtype=torch.bfloat16,
            attn_implementation='sdpa',local_files_only=True).eval().requires_grad_(False)
    assert type(loaded) is Qwen3VLModel and not torch.cuda.is_initialized()
    return loaded


def test_exact_skeleton_native_class_and_all713_meta_key_shapes():
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from transformers import Qwen3VLConfig,Qwen3VLModel
    index=json.loads((SNAPSHOT/'model_index.json').read_text())
    assert index['text_encoder']==['transformers','Qwen3VLModel']
    assert packet()['architectures']==['Qwen3VLModel']
    with init_empty_weights(include_buffers=True),torch.device('meta'):
        model=Qwen3VLModel(Qwen3VLConfig(**packet()))
    with safe_open(SNAPSHOT/'text_encoder/model.safetensors',framework='pt',device='cpu') as handle:
        expected={name:handle.get_slice(name).get_shape() for name in handle.keys()}
        assert {handle.get_slice(name).get_dtype() for name in handle.keys()}=={'BF16'}
    assert expected=={name:list(value.shape) for name,value in model.state_dict().items()}
    assert len(expected)==713 and sum(value.numel() for value in model.parameters())==4437815808
    assert all(value.device.type=='meta' for value in model.parameters())
    assert not torch.cuda.is_initialized()


def test_actual_native_loader_passes_exact_guard_and_binds_identity(actual_encoder):
    instance=StrictCudaUNI.__new__(StrictCudaUNI)
    before=torch.random.get_rng_state().clone()
    digest=instance.module_identity(actual_encoder)
    assert len(digest)==64
    assert instance.strict_encoder==dict(native_class='transformers.Qwen3VLModel',
        attention=['sdpa','sdpa','sdpa'],dtype='bfloat16')
    assert torch.equal(before,torch.random.get_rng_state()) and not torch.cuda.is_initialized()


def test_actual_pinned_pipeline_uses_native12_layer_hidden_state_contract(actual_encoder):
    from transformers import AutoTokenizer
    from diffusers import Krea2Pipeline
    index=json.loads((SNAPSHOT/'model_index.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(SNAPSHOT/'tokenizer',local_files_only=True)
    pipe=Krea2Pipeline(scheduler=None,vae=None,text_encoder=actual_encoder,tokenizer=tokenizer,
        transformer=None,text_encoder_select_layers=index['text_encoder_select_layers'])
    with torch.no_grad():
        hidden,mask=pipe.get_text_hidden_states('A photograph of a red ceramic mug on a wooden table.',32,torch.device('cpu'))
        repeated,repeated_mask=pipe.get_text_hidden_states('A photograph of a red ceramic mug on a wooden table.',32,torch.device('cpu'))
    assert hidden.shape==(1,32,12,16) and hidden.dtype==torch.bfloat16
    assert mask.shape==(1,32) and mask.dtype==torch.bool
    assert torch.equal(hidden,repeated) and torch.equal(mask,repeated_mask)


def test_generation_wrapper_and_subclass_still_refused_by_exact_guard():
    from accelerate import init_empty_weights
    from transformers import Qwen3VLModel,Qwen3VLForConditionalGeneration
    class UnqualifiedSubclass(Qwen3VLModel):pass
    with init_empty_weights(include_buffers=True),torch.device('meta'):
        models=[Qwen3VLForConditionalGeneration(tiny_config()),UnqualifiedSubclass(tiny_config())]
    for model in models:
        with pytest.raises(ValueError,match='text encoder class differs'):
            StrictCudaUNI.__new__(StrictCudaUNI).module_identity(model)


@pytest.mark.parametrize('changed',['dtype','trainable','attention'])
def test_actual_native_class_does_not_bypass_dtype_freeze_or_attention_guards(actual_encoder,changed):
    if changed=='dtype':actual_encoder.to(torch.float32)
    elif changed=='trainable':next(actual_encoder.parameters()).requires_grad_(True)
    else:actual_encoder.config.text_config._attn_implementation='flash_attention_2'
    with pytest.raises(ValueError,match='precision differs' if changed!='attention' else 'attention implementation differs'):
        StrictCudaUNI.__new__(StrictCudaUNI).module_identity(actual_encoder)
