from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch

from krea2_strict.execution import policy, forward_guards


@pytest.fixture
def tiny(strict_flags, monkeypatch):
    # The historical fixture deliberately remains unchanged. Its test-only
    # constructor seed is scoped to CPU here; the new product never calls it.
    monkeypatch.setattr(torch, "manual_seed", torch.random.default_generator.manual_seed)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda *_: pytest.fail("CUDA-wide seed"))
    from krea2.synthetic import staged_fixture
    with torch.random.fork_rng(devices=[]):
        native, cache = staged_fixture()
    native.strict_execution = dict(numerical=policy(), sources={"tiny_fixture": True})
    native.strict_assets = {"full_weight_hashes_verified": True, "fixture": "random tiny real Krea2 CPU"}
    handles = forward_guards(native.model, native.strict_execution["numerical"])
    try:
        yield native, cache
    finally:
        for h in handles:
            h.remove()


def test_actual_tiny_shared_adapter_teacher_parity(tiny):
    from krea2_strict.cache import verify_native_parity
    native, cache = tiny
    receipt = verify_native_parity(native, cache)
    assert receipt["all_rows_exact"]
    assert receipt["unique_stored_rows"] == 4
    cache["edit"][0]["teacher"][0, 0, 0] += .125
    with pytest.raises(ValueError, match="OFF parity"):
        verify_native_parity(native, cache)


def game(native, cache, *, lr=5e-5):
    from krea2_strict.game import build_game
    return build_game(native, cache, cache_sha256="actual-tiny-cache", model_identity={"fixture": "strict-cpu-software-seam"},
        total_steps=2, batch_size=2, seed=7, lr=lr, d_lr_mult=2)


def test_actual_shared_game_exact_resume_rates_and_raw_response(tiny, monkeypatch):
    from particle_sliders.pg09 import _hash
    native, cache = tiny
    monkeypatch.setattr(torch, "manual_seed", lambda *_: pytest.fail("CUDA-wide seed"))
    before = torch.get_rng_state().clone()
    full = game(native, cache)
    assert torch.equal(before, torch.get_rng_state())
    actual_groups = {name: [g["lr"] for g in optimizer.param_groups]
                     for name, optimizer in (("generator", full.opt_g), ("discriminator", full.opt_d))}
    assert actual_groups["generator"] == [5e-5]
    assert actual_groups["discriminator"] and set(actual_groups["discriminator"]) == {1e-4}
    frozen = {n: p.detach().clone() for n, p in native.model.named_parameters() if ".lora_" not in n}
    full.step(); middle = deepcopy(full.state_dict()); full.step(); expected = deepcopy(full.state_dict())
    full.load_state_dict(middle); full.step()
    assert _hash(full.state_dict()) == _hash(expected)
    row = full.rows[0]
    with torch.no_grad():
        response = full.backend.velocity(row.row.neutral, row.row.latent, row.row.timestep, scale=0)
    assert torch.equal(response.cpu(), row.neutral.cpu())
    for name, p in native.model.named_parameters():
        if name in frozen:
            assert torch.equal(p, frozen[name])
    # Configuration is part of the full game packet; an LR change is a new
    # cohort, not a way to relabel an old optimizer state.
    other = game(native, cache, lr=1e-4)
    before = _hash(other.state_dict())
    with pytest.raises(ValueError, match="recipe"):
        other.load_state_dict(middle)
    assert _hash(other.state_dict()) == before


def test_old_cache_refused_before_factory_even_with_claimed_conditioning():
    from krea2_strict.cache import validate_cache, coordinate_identity
    args = SimpleNamespace(model_id="bbox", transformer_subfolder="transformer", revision="r",
        skeleton_model="raw", skeleton_revision="s", rank=2, resolution=32,
        sample_steps=8, sample_guidance=0., mu=1.15)
    cache = dict(key=dict(coordinate_identity(args), conditioning_identity={"legacy": "encoding"}),
                 edit=[dict(prompt="x", positive="y")], preserve=[dict(prompt="z", positive="z")])
    with pytest.raises(ValueError, match="fresh capture required"):
        validate_cache(cache, args, {"new": "execution"}, {"new": "assets"})


def test_strict_constructor_refuses_policy_before_cuda_or_model(monkeypatch):
    import krea2_strict.native as native
    monkeypatch.setattr(native, "require_args", lambda args: {})
    monkeypatch.setattr(native, "execution_identity", lambda: (_ for _ in ()).throw(ValueError("numerical differs")))
    monkeypatch.setattr(native, "admission", lambda *_: pytest.fail("CUDA admission reached"))
    monkeypatch.setattr(native.CudaUNI, "__init__", lambda *a, **kw: pytest.fail("model allocated"))
    args = SimpleNamespace(resolution=512, sample_resolution=768, final_resolution=0, device=0)
    with pytest.raises(ValueError, match="numerical differs"):
        native.StrictCudaUNI(args, ["x"], "/unused", expected_gpu_uuid="GPU-unused")


def test_strict_constructor_failure_restores_rng_and_partial_ownership(strict_flags, monkeypatch):
    import krea2_strict.native as native
    monkeypatch.setattr(native, "require_args", lambda args: {})
    execution = dict(numerical=policy())
    monkeypatch.setattr(native, "execution_identity", lambda: execution)
    monkeypatch.setattr(native, "inspect_assets", lambda *a, **kw: {"full_weight_hashes_verified": True})
    monkeypatch.setattr(native, "admission", lambda *_: None)
    from contextlib import contextmanager
    @contextmanager
    def cpu_only():
        from krea2_strict.execution import initialization
        with initialization(device="cpu"):
            yield
    monkeypatch.setattr(native, "initialization", cpu_only)
    closes = []
    monkeypatch.setattr(native, "own_native", lambda n: lambda: closes.append(True))
    def fail(self, *a, **kw):
        self.pipe = SimpleNamespace()
        self.model = torch.nn.Linear(2, 2)
        raise RuntimeError("PEFT constructor failed")
    monkeypatch.setattr(native.CudaUNI, "__init__", fail)
    args = SimpleNamespace(resolution=512, sample_resolution=768, final_resolution=0, device=0,
                           transformer="t", skeleton_model="s")
    before = torch.get_rng_state()
    with pytest.raises(RuntimeError, match="PEFT constructor"):
        native.StrictCudaUNI(args, ["x"], "/unused", expected_gpu_uuid="GPU-unused")
    assert closes == [True]
    assert torch.equal(before, torch.get_rng_state())


def test_inference_source_mismatch_precedes_allocation(strict_flags, monkeypatch):
    import krea2_strict.inference as inference
    import krea2_strict.assets as assets
    from krea2.native_runtime import CARD_FIELDS
    args = SimpleNamespace(**{key: key for key in CARD_FIELDS})
    monkeypatch.setattr(inference, "arguments", lambda options: args)
    monkeypatch.setattr(assets, "require_args", lambda args: {})
    monkeypatch.setattr(inference, "execution_identity", lambda: {"current": "strict-source"})
    spec = SimpleNamespace(configuration=dict(card={key: key for key in CARD_FIELDS},
        execution={"old": "source"}, assets={"full_weight_hashes_verified": True}),
        identity={"protocol": inference.PROTOCOL})
    with pytest.raises(ValueError, match="before base allocation"):
        inference.validate(dict(prompts=["x"]), spec)


def test_actual_tiny_native_only_inference_factory_has_no_game_or_rows(tiny, monkeypatch):
    import krea2_strict.inference as inference
    import krea2_strict.native as loader
    import krea2_strict.game as training
    import krea2_strict.cache as teachers
    from particle_sliders.pg09 import _hash
    native, cache = tiny
    closes = []
    native.strict_close = lambda **kwargs: closes.append(True)
    monkeypatch.setattr(loader, "StrictCudaUNI", lambda *a, **kw: native)
    monkeypatch.setattr(inference, "arguments", lambda options: SimpleNamespace(save_dir="/unused"))
    monkeypatch.setattr(inference, "base_identity", lambda native, dummy: dict(
        fixture="random tiny actual Krea2 CPU", frozen=_hash({n: p for n, p in native.model.named_parameters() if ".lora_" not in n})))
    monkeypatch.setattr(training, "build_game", lambda *a, **kw: pytest.fail("inference built a game"))
    monkeypatch.setattr(teachers, "verify_native_parity", lambda *a, **kw: pytest.fail("inference read teacher rows"))
    bound = inference.prepare(dict(prompts=["neutral"], expected_gpu_uuid="test-only-no-CUDA"), None)
    try:
        assert not any(hasattr(bound, name) for name in ("game", "rows", "critic", "optimizer"))
        for strength in (0., .5, 1.):
            actual = bound.response(prompt="neutral", latent=native.pipe._unpack_latents(cache["edit"][0]["z"],32,32),
                                    timestep=cache["edit"][0]["t"], strength=strength, seed=7)
            assert actual.shape == (1,4,1,4,4)
            assert torch.isfinite(actual).all()
        with bound.observation():
            pass
    finally:
        bound.close()
    assert closes == [True]


@pytest.mark.parametrize("value", [True, 0., -1., float("nan"), float("inf")])
def test_bad_lr_refused_before_rows(value):
    from krea2_strict.game import build_game
    with pytest.raises(ValueError, match="finite and positive"):
        build_game(None, {}, cache_sha256="unused", model_identity={}, total_steps=2, lr=value)
