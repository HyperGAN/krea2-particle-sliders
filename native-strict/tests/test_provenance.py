import pytest
import torch


def test_exact_installed_public_base_core_and_candidate_closure():
    from krea2_strict.provenance import verify_sources
    actual = verify_sources()
    assert actual["base_commit"] == "9d3ad435daa230fe7746bc0888abad8fcf2a7913"
    assert actual["core"]["commit"] == "ddb41423bcefc1fe76cd94d0f9b367353b3ea137"
    assert not torch.cuda.is_initialized()


def test_declared_actual_native_dependency_sources():
    from krea2_strict.provenance import runtime_sources
    actual = runtime_sources()
    assert actual["diffusers"]["version"] == "0.40.0.dev0"
    assert actual["transformers"]["version"] == "5.15.0"
    assert not torch.cuda.is_initialized()


def test_product_helper_mutation_is_refused_before_cuda(monkeypatch):
    from krea2_strict.provenance import verify_sources
    from krea2_strict import provenance
    real = provenance.index
    def changed(root):
        value = real(root)
        if root.name == "krea2_strict":
            value["execution.py"] = "0" * 64
        return value
    monkeypatch.setattr(provenance, "index", changed)
    with pytest.raises(ValueError, match="source bytes"):
        verify_sources()
    assert not torch.cuda.is_initialized()


def test_base_import_shadow_refused_before_model(monkeypatch):
    import krea2
    from krea2_strict.provenance import verify_sources
    monkeypatch.setattr(krea2, "__file__", "/tmp/shadow/krea2/__init__.py")
    with pytest.raises(ValueError, match="shadowed"):
        verify_sources()
