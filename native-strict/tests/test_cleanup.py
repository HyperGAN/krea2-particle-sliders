from contextlib import contextmanager
from types import SimpleNamespace
import pytest
import torch

from krea2_strict.native import StrictCudaUNI


class Owned(torch.nn.Linear):
    def __init__(self, label, events, *, fail=False):
        super().__init__(2, 2)
        self.label, self.events, self.fail = label, events, fail
    def to(self, device):
        self.events.append((self.label, str(device)))
        if self.fail:
            raise RuntimeError("owned move failed")
        return super().to(device)


def test_partial_pipe_without_model_releases_only_owned_modules_and_rng():
    events = []
    external = Owned("external", events)
    expected = {k: v.clone() for k, v in external.state_dict().items()}
    native = StrictCudaUNI.__new__(StrictCudaUNI)
    native.pipe = SimpleNamespace(transformer=Owned("transformer", events),
        text_encoder=Owned("encoder", events), vae=Owned("vae", events))
    native.device = torch.device("cpu")
    native.position_ids = {"owned": torch.ones(2)}
    before = torch.get_rng_state()
    native.strict_close(); native.strict_close()
    assert events == [("transformer", "cpu"), ("encoder", "cpu"), ("vae", "cpu")]
    assert native.position_ids == {}
    assert torch.equal(before, torch.get_rng_state())
    assert all(torch.equal(v, expected[k]) for k, v in external.state_dict().items())
    assert not torch.cuda.is_initialized()


def test_failure_attempts_other_owned_releases_and_can_retry():
    events = []
    native = StrictCudaUNI.__new__(StrictCudaUNI)
    native.pipe = SimpleNamespace(transformer=Owned("transformer", events, fail=True),
        text_encoder=Owned("encoder", events), vae=Owned("vae", events))
    native.device = torch.device("cpu")
    before = torch.get_rng_state()
    with pytest.raises(RuntimeError, match="owned move"):
        native.strict_close()
    assert events == [("transformer", "cpu"), ("encoder", "cpu"), ("vae", "cpu")]
    with pytest.raises(RuntimeError, match="closing or closed"):
        native.forward()
    native.pipe.transformer.fail = False
    native.strict_close()
    assert native._strict_closed
    assert torch.equal(before, torch.get_rng_state())


def test_owned_allocator_context_restores_caller_device_even_on_error(monkeypatch):
    native = StrictCudaUNI.__new__(StrictCudaUNI)
    native.device = torch.device("cuda:0")
    state = {"selected": 7}
    calls = []
    @contextmanager
    def owned_device(device):
        previous = state["selected"]
        state["selected"] = device.index
        calls.append(("enter", device.index))
        try:
            yield
        finally:
            state["selected"] = previous
            calls.append(("restore", previous))
    monkeypatch.setattr(torch.cuda, "device", owned_device)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: (_ for _ in ()).throw(RuntimeError("allocator failure")))
    before = torch.get_rng_state()
    with pytest.raises(RuntimeError, match="allocator failure"):
        native.strict_close()
    assert state["selected"] == 7 and calls == [("enter", 0), ("restore", 7)]
    assert torch.equal(before, torch.get_rng_state())
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)


def test_close_moves_network_optimizer_aliases_and_pre_binding_checkpoint_restore(monkeypatch):
    from krea2.synthetic import staged_fixture
    from krea2.native_runtime import own_native
    from krea2_strict.game import build_game
    # Numerical policy is checked separately; this bounded test isolates close.
    monkeypatch.setattr("krea2_strict.execution.require_policy", lambda *_: None)
    monkeypatch.setattr(torch, "manual_seed", torch.random.default_generator.manual_seed)
    with torch.random.fork_rng(devices=[]):
        old, cache = staged_fixture()
    native = StrictCudaUNI.__new__(StrictCudaUNI)
    native.__dict__.update(old.__dict__)
    native.strict_execution = {"numerical": {"test": "cleanup-only tiny CPU"}}
    root = native.model.get_base_model()
    original = root._gradient_checkpointing_func
    native._strict_restore = own_native(native)
    game = build_game(native, cache, cache_sha256="tiny", model_identity={"test": "cleanup"},
                      total_steps=2, batch_size=2, seed=7)
    assert root._gradient_checkpointing_func is not original
    game.step()
    trainable_aliases = list(game.network.parameters())
    before = torch.get_rng_state()
    native.strict_close(game=game)
    assert root._gradient_checkpointing_func is original
    assert all(p.device.type == "cpu" and (p.grad is None or p.grad.device.type == "cpu") for p in trainable_aliases)
    assert all(p.device.type == "cpu" for p in game.critic.parameters())
    for optimizer in (game.opt_g, game.opt_d):
        for state in optimizer.state.values():
            assert all(not isinstance(v, torch.Tensor) or v.device.type == "cpu" for v in state.values())
    assert torch.equal(before, torch.get_rng_state())
    assert not torch.cuda.is_initialized()
