import os
import pytest
import torch


@pytest.fixture
def strict_flags(monkeypatch):
    attrs = [(torch.backends.cuda.matmul, "allow_tf32", False),
             (torch.backends.cudnn, "allow_tf32", False),
             (torch.backends.cudnn, "benchmark", False),
             (torch.backends.cudnn, "deterministic", True),
             (torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction", False),
             (torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction", False)]
    old = [(owner, key, getattr(owner, key)) for owner, key, _ in attrs]
    before = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.get_float32_matmul_precision(), torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(), torch.backends.cuda.cudnn_sdp_enabled(),
        torch.backends.cuda.math_sdp_enabled(), torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed())
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_float32_matmul_precision("highest")
    for owner, key, value in attrs:
        setattr(owner, key, value)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.delenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", raising=False)
    monkeypatch.delenv("NVIDIA_TF32_OVERRIDE", raising=False)
    try:
        yield
        assert not torch.cuda.is_initialized()
    finally:
        torch.use_deterministic_algorithms(before[0], warn_only=before[1])
        torch.set_float32_matmul_precision(before[2])
        for owner, key, value in old:
            setattr(owner, key, value)
        torch.backends.cuda.enable_flash_sdp(before[3])
        torch.backends.cuda.enable_mem_efficient_sdp(before[4])
        torch.backends.cuda.enable_cudnn_sdp(before[5])
        torch.backends.cuda.enable_math_sdp(before[6])
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(before[7])
