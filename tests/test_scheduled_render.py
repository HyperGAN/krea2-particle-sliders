from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from krea2.scheduled_render import render


def fixture(*, fail=False):
    values = []
    current = 0.25

    @contextmanager
    def scale(value):
        nonlocal current
        previous = current
        current = value
        try:
            yield
        finally:
            current = previous

    class Model(torch.nn.Module):
        def forward(self, value):
            values.append(current)
            if fail:
                raise RuntimeError("native failure")
            return value

    model = Model()
    native = SimpleNamespace(args=SimpleNamespace(sample_steps=2), model=model, scale=scale)

    def original(*, strength, **request):
        with scale(strength):
            model(torch.zeros(1))
            model(torch.zeros(1))
        return "image"

    inference = SimpleNamespace(
        owner=SimpleNamespace(check=lambda: None, components=SimpleNamespace(native=native)),
        session=SimpleNamespace(capabilities=SimpleNamespace(check=lambda *a, **k: None)),
        render=original,
    )
    return inference, model, values


def test_original_denoiser_receives_native_strength_per_call():
    owner, model, values = fixture()
    image, receipt = render(owner, strength=0.5, step_gains=[1.5, 0.5], prompt="room")
    assert image == "image"
    assert values == [0.75, 0.25]
    assert receipt == {"effective_strengths": values, "denoiser_calls": 2}
    assert not model._forward_hooks and not model._forward_pre_hooks


def test_native_failure_removes_both_owned_hooks():
    owner, model, _ = fixture(fail=True)
    with pytest.raises(RuntimeError, match="native failure"):
        render(owner, strength=0.5, step_gains=[1.5, 0.5], prompt="room")
    assert not model._forward_hooks and not model._forward_pre_hooks
