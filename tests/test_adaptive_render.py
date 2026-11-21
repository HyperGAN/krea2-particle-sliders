from types import SimpleNamespace

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from peft import LoraConfig, get_peft_model
from particle_sliders.adaptive_steering import ActivationGate, activation_projection
from particle_sliders.backends import lora_scale_context

from krea2.adaptive import render
from krea2.scheduled_render import render as render_fixed


def fixture(*, fail=False, repeat_first=False):
    values = []

    class Transformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(1, 1, bias=False)
            self.proj.weight.data.fill_(1.0)

        def forward(self, *, hidden_states, timestep):
            values.append(self.proj.scaling["default"])
            if fail:
                raise RuntimeError("denoiser failed")
            return self.proj(hidden_states)

    model = get_peft_model(Transformer(), LoraConfig(r=1, lora_alpha=1, target_modules=["proj"]))
    model.get_base_model().proj.lora_A["default"].weight.data.fill_(1.0)
    model.get_base_model().proj.lora_B["default"].weight.data.fill_(1.0)
    pipeline = SimpleNamespace(do_classifier_free_guidance=False)
    scale = lora_scale_context([("transformer", model)])
    native = SimpleNamespace(args=SimpleNamespace(sample_steps=2, sample_guidance=0),
                             model=model, pipe=pipeline, scale=scale)

    def original(*, strength, **request):
        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(sigmas=[1.0, 0.5])
        scheduler.set_begin_index(0)
        pipeline.scheduler = scheduler
        latents = torch.ones(1, 2, 1)
        with scale(strength):
            for t in scheduler.timesteps:
                inputs = dict(hidden_states=latents, timestep=(t / 1000).reshape(1))
                velocity = model(**inputs)
                if repeat_first:
                    model(**inputs)
                latents = scheduler.step(velocity, t, latents, return_dict=False)[0]
        return latents

    inference = SimpleNamespace(
        owner=SimpleNamespace(check=lambda: None, components=SimpleNamespace(native=native)),
        session=SimpleNamespace(capabilities=SimpleNamespace(check=lambda *a, **k: None)),
        render=original,
    )
    return inference, model, values


def assert_restored(model):
    assert model.get_base_model().proj.scaling == {"default": 1.0}
    assert not model._forward_hooks and not model._forward_pre_hooks


def test_adaptive_gate_uses_current_native_latent_and_records_interval_dose():
    owner, model, values = fixture()
    readout = []

    def alignment(latents, timestep):
        score = activation_projection(latents, torch.ones(1), strength=-0.5).mean()
        readout.append((score.item(), timestep.item()))
        return score

    output, receipt = render(owner, strength=-0.5, gate=ActivationGate("cfp", -0.8),
                             alignment=alignment)
    assert readout == [(-1.0, 1.0), (-0.75, 0.5)]
    assert values == receipt["effective_strengths"] == [-0.5, 0.0]
    assert receipt["fired"] == 1 and receipt["eligible"] == receipt["denoiser_calls"] == 2
    assert receipt["gate_rate"] == 0.5
    assert receipt["sigma_weighted_dose"] == 0.25
    assert [row["probability"] for row in receipt["decisions"]] == [1.0, 0.0]
    assert [row["alignment"] for row in receipt["decisions"]] == [-1.0, -0.75]
    assert receipt["decisions"][-1]["sigma_end"] == 0.0
    fixed, _ = render_fixed(owner, strength=-0.5, step_gains=[1.0, 0.0])
    assert torch.equal(output, fixed)
    assert_restored(model)


def test_random_zero_and_all_on_replay_exactly_with_independent_gate_state():
    owner, model, _ = fixture()
    for rate, strength, reference in [(0.0, 0.25, 0.0), (1.0, 0.25, 0.25)]:
        gate = ActivationGate("random", rate, seed=4)
        result, receipt = render(owner, strength=strength, gate=gate)
        assert torch.equal(result, owner.render(strength=reference))
        assert receipt["effective_strengths"] == [reference, reference]
        assert all(row["alignment"] is None for row in receipt["decisions"])
        with pytest.raises(ValueError, match="fresh gate"):
            render(owner, strength=strength, gate=gate)
        assert_restored(model)


def test_repeated_interval_fails_without_second_gate_advance():
    owner, model, _ = fixture(repeat_first=True)
    gate = ActivationGate("random", 0.5, seed=4)
    with pytest.raises(RuntimeError, match="exactly one forward per Euler interval"):
        render(owner, strength=0.25, gate=gate)
    assert gate.usage.eligible == 1
    assert_restored(model)


def test_failure_removes_owned_adaptive_hooks_and_restores_scale():
    owner, model, _ = fixture(fail=True)
    with pytest.raises(RuntimeError, match="denoiser failed"):
        render(owner, strength=0.25, gate=ActivationGate("random", 1.0, seed=4))
    assert_restored(model)
