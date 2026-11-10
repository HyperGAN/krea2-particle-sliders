"""Explicit new-cohort API. Historical Krea2 entrypoints remain unchanged."""
PROFILE = "krea2-active-PEFT-PG095-downQR-sqrt5-FieldCriticKEEP-no-bank-v1"


def open_training(*args, **kwargs):
    from .api import open_training as implementation
    return implementation(*args, **kwargs)


def open_inference(*args, **kwargs):
    from .api import open_inference as implementation
    return implementation(*args, **kwargs)
