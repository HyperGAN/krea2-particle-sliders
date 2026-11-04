from types import SimpleNamespace
import os
import pytest
import torch

from krea2_strict.execution import (admission, canonical_gpu_uuid, forward_guards, geometry,
    initialization, policy, require_policy, resolution)


def test_resolved_defaults_are_raw_and_preserve_inherited_lr():
    from krea2_strict.lifecycle import config
    cfg = config()
    assert cfg.backend == "krea2-native-strict"
    assert cfg.options["lr"] == 5e-5
    assert cfg.options["ema_decay"] is None


def test_policy_inspection_is_pure_and_detects_change(strict_flags):
    expected = policy()
    assert policy() == expected
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    with pytest.raises(ValueError, match="bf16_reduction"):
        require_policy(expected)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


@pytest.mark.parametrize("kind", ["warn", "tf32", "precision", "flash", "efficient", "cudnn", "lowmath", "fp16"])
def test_strict_numerical_refusals(strict_flags, kind):
    changes = {
        "warn": lambda: torch.use_deterministic_algorithms(True, warn_only=True),
        "tf32": lambda: setattr(torch.backends.cuda.matmul, "allow_tf32", True),
        "precision": lambda: torch.set_float32_matmul_precision("high"),
        "flash": lambda: torch.backends.cuda.enable_flash_sdp(True),
        "efficient": lambda: torch.backends.cuda.enable_mem_efficient_sdp(True),
        "cudnn": lambda: setattr(torch.backends.cudnn, "benchmark", True),
        "lowmath": lambda: torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True),
        "fp16": lambda: setattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction", True)}
    changes[kind]()
    with pytest.raises(ValueError, match="numerical policy"):
        policy()


def test_override_refusal(strict_flags, monkeypatch):
    monkeypatch.setenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "1")
    with pytest.raises(ValueError, match="override"):
        policy()


@pytest.mark.parametrize("failure", [False, True])
def test_cpu_initialization_scopes_rng_on_success_and_failure(monkeypatch, failure):
    monkeypatch.setattr(torch, "manual_seed", lambda *a: pytest.fail("CUDA-wide torch.manual_seed"))
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda *a: pytest.fail("CUDA-wide seed"))
    external = torch.get_rng_state().clone()
    values = []
    for _ in range(2):
        try:
            with initialization(41, device="cpu"):
                values.append(torch.randn(8))
                if failure:
                    raise RuntimeError("failed constructor")
        except RuntimeError:
            pass
        assert torch.equal(external, torch.get_rng_state())
    assert torch.equal(*values)


def test_cuda_initialization_only_seeds_owned_generator(monkeypatch):
    from contextlib import contextmanager
    calls = []
    @contextmanager
    def fork(*, devices):
        calls.append(("fork", devices))
        before = torch.get_rng_state()
        try:
            yield
        finally:
            torch.set_rng_state(before)
    generators = [SimpleNamespace(manual_seed=lambda s: calls.append(("owned", s))),
                  SimpleNamespace(manual_seed=lambda s: pytest.fail("other GPU seeded"))]
    monkeypatch.setattr(torch.random, "fork_rng", fork)
    monkeypatch.setattr(torch.cuda, "default_generators", generators)
    monkeypatch.setattr(torch, "manual_seed", lambda *a: pytest.fail("CUDA-wide seed"))
    with initialization():
        pass
    assert calls == [("fork", [0]), ("owned", 41)]


@pytest.mark.parametrize("value", [769, 1024, 0, -16, True, 17])
def test_render_cap(value):
    with pytest.raises(ValueError, match="resolution"):
        resolution(value)


def test_actual_combined_geometry_guard():
    geometry(torch.empty(1, 3584, 16), torch.empty(1, 512, 2, 8))
    with pytest.raises(ValueError, match="geometry"):
        geometry(torch.empty(1, 3585, 16), torch.empty(1, 512, 2, 8))
    with pytest.raises(ValueError, match="geometry"):
        geometry(torch.empty(1, 8, 16), torch.empty(1, 3, 2, 8), expected_image_tokens=4)


@pytest.mark.parametrize("actual", ["72c1b506-891d-b8bc-b353-e020585e1c47", "GPU-72c1b506-891d-b8bc-b353-e020585e1c47"])
def test_truthful_torch_and_nvidia_uuid_forms(monkeypatch, actual):
    declared = "GPU-72c1b506-891d-b8bc-b353-e020585e1c47"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", declared)
    monkeypatch.setenv("KREA2_STRICT_GUARDIAN", str(os.getppid()))
    monkeypatch.setenv("KREA2_STRICT_PROCESS_CAP_GIB", "36")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(uuid=actual))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (42 * 2**30, 48 * 2**30))
    admission(declared)


def test_uuid_mismatch_refuses_before_memory(monkeypatch):
    declared = "GPU-72c1b506-891d-b8bc-b353-e020585e1c47"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", declared)
    monkeypatch.setenv("KREA2_STRICT_GUARDIAN", str(os.getppid()))
    monkeypatch.setenv("KREA2_STRICT_PROCESS_CAP_GIB", "36")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(uuid="00000000-0000-0000-0000-000000000000"))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: pytest.fail("memory queried after wrong UUID"))
    with pytest.raises(ValueError, match="UUID differs"):
        admission(declared)


@pytest.mark.parametrize("bad", [None, True, "", "0", "GPU-invalid", "GPU-GPU-72c1b506-891d-b8bc-b353-e020585e1c47"])
def test_invalid_uuid_without_cuda(bad):
    with pytest.raises(ValueError):
        canonical_gpu_uuid(bad)


def test_forward_rechecks_policy_and_actual_geometry(strict_flags):
    class Host(torch.nn.Module):
        def forward(self, hidden_states, encoder_hidden_states):
            return hidden_states
    host = Host()
    handles = forward_guards(host, policy())
    try:
        assert host(torch.zeros(1, 4, 8), torch.zeros(1, 3, 2, 8)).shape == (1, 4, 8)
        with pytest.raises(ValueError, match="geometry"):
            host(torch.zeros(1, 4095, 8), torch.zeros(1, 3, 2, 8))
        torch.backends.cudnn.benchmark = True
        with pytest.raises(ValueError, match="policy"):
            host(torch.zeros(1, 4, 8), torch.zeros(1, 3, 2, 8))
    finally:
        for h in handles:
            h.remove()
