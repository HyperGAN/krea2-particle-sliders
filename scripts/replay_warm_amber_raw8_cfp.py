"""Replay the measured CFP panel on the already-owned archived Krea2 RAW8."""

import hashlib
import io
import json
import os
from math import fsum, isfinite
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from krea2 import adaptive
from particle_sliders import adaptive_steering
from particle_sliders.lifecycle import Capabilities


EVIDENCE = Path(__file__).resolve().parents[1] / "evidence/warm-amber-raw8-cfp-heldout.json"
EVIDENCE_SHA256 = "d4016834a55137dbbe73b698515e81429810a208515b9d91bb802d92c5841a63"
ADAPTIVE_RENDER_SHA256 = "4c6c18d8f7154eb967ad60c0ff7dc301922f825cf13f300f8903081f55481fda"
SHARED_GATE_SHA256 = "5fc2a5256a935eca9cecc365d9ac62e5e94cdc375d2ab33c5dc26eb344241814"
PROMPTS = {
    "dev0-seed29001": "A photograph of a blue bicycle beside a stone wall.",
    "dev1-seed29002": "A photograph of a white teapot on a kitchen counter.",
    "dev2-seed29005": "A photograph of an orange cat beside a green door.",
    "dev3-seed29004": "A photograph of a wooden bridge over a quiet stream.",
}


def _read_evidence():
    payload = EVIDENCE.read_bytes()
    if hashlib.sha256(payload).hexdigest() != EVIDENCE_SHA256:
        raise ValueError("the measured CFP evidence digest does not match")
    return json.loads(payload)


def _verify_sources():
    for module, expected in ((adaptive, ADAPTIVE_RENDER_SHA256),
                             (adaptive_steering, SHARED_GATE_SHA256)):
        source = module.__loader__.get_data(module.__file__)
        if hashlib.sha256(source).hexdigest() != expected:
            raise ValueError(f"{module.__name__} differs from the measured CFP source")


def _verify_native_raw(native, raw_path, evidence, check):
    cohort = evidence["cohort"]
    base = cohort["base"]
    sampler = evidence["native_sampler"]
    check()
    settings = (native.args.model_id, native.args.revision, native.args.sample_steps,
                native.args.sample_guidance, native.args.mu, native.args.resolution)
    expected = (base["model_id"], base["revision"], sampler["steps"],
                sampler["guidance"], sampler["mu"], sampler["resolution"][0])
    if (settings != expected or native.device.type != "cuda" or native.dtype != torch.bfloat16
            or native.pipe.transformer is not native.model or native.pipe.scheduler.order != 1):
        raise ValueError("the owned Krea2 runtime differs from the measured sampler")
    with Path(raw_path).open("rb") as source:
        if (os.fstat(source.fileno()).st_size != cohort["raw"]["bytes"]
                or hashlib.file_digest(source, "sha256").hexdigest() != cohort["raw"]["sha256"]):
            raise ValueError("the saved Warm Amber RAW8 digest or byte count does not match")
        source.seek(0)
        raw = torch.load(source, map_location="cpu", weights_only=True)
    config = raw["config"]
    if (raw["completed_steps"] != cohort["regularizer_updates"]
            or config["session_ema_decay"] is not None or config["weight_selection"] != "raw"):
        raise ValueError("the archived state must have eight RAW updates without EMA")
    backend = config["backend"]
    geometry = backend["native_geometry"]
    conditioning = dict(backend["model_identity"]["frozen_cache"]["cache_key"]["conditioning_identity"])
    conditioning.pop("strict_execution")
    check()
    if (native.conditioning_identity != conditioning
            or native.frozen_identity() != base["frozen_model_sha256"]
            or native.pipe.patch_size != geometry["patch_size"]
            or native.pipe.vae_scale_factor != geometry["vae_scale_factor"]):
        raise ValueError("the measured transformer, VAE or conditioning identity does not match")
    names = backend["adapter_names"]
    parameters = {name: value for name, value in native.model.named_parameters()
                  if ".lora_A." in name or ".lora_B." in name}
    if (parameters.keys() != set(names)
            or raw["network"].keys() != {f"weights.{index}" for index in range(len(names))}):
        raise ValueError("the owned Krea2 LoRA leaves differ from the archived RAW8")
    for index, name in enumerate(names):
        check()
        live = parameters[name].detach()
        saved = raw["network"][f"weights.{index}"]
        if live.dtype != saved.dtype or not torch.equal(live.cpu(), saved):
            raise ValueError("the owned Krea2 LoRA does not contain the exact archived RAW8 weights")


def _gate_seed(row):
    material = json.dumps([row["case"], row["seed"], PROMPTS[row["case"]]], ensure_ascii=False).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _alignment_callback(fit, *, device, strength):
    direction = torch.tensor(fit["direction"], dtype=torch.float32, device=device)
    shape = tuple(fit["packed_BF16_shape"])

    def alignment(packed, _timestep):
        if packed.shape != shape or packed.dtype != torch.bfloat16 or packed.device != direction.device:
            raise ValueError("the packed activations differ from the measured CFP readout")
        pooled = packed.mean(dim=(0, 1), dtype=torch.float64).float()
        return adaptive_steering.activation_projection(pooled, direction, strength=strength)

    return alignment


def _match_receipt(receipt, row, sampler):
    sigmas = sampler["actual_sigmas"]
    widths = [start - end for start, end in zip(sigmas[:-1], sigmas[1:], strict=True)]
    decisions = receipt["decisions"]
    saved = row["cfp"]
    dose = receipt["sigma_weighted_dose"]
    if (receipt["eligible"] != sampler["steps"] or receipt["fired"] != saved["firings"]
            or [step["sigma_start"] for step in decisions] != sigmas[:-1]
            or [step["sigma_end"] for step in decisions] != sigmas[1:]
            or [step["alignment"] for step in decisions] != saved["alignment_scores"]
            or [step["fired"] for step in decisions] != saved["fired_mask"]
            or receipt["effective_strengths"] != saved["effective_interval_strengths"]
            or dose != row["common_sigma_weighted_dose"]):
        raise ValueError("the realized CFP trace differs from the measured case")
    strength = dose / fsum(widths)
    if strength != row["constant"]["effective_strength"]:
        raise ValueError("the derived constant strength differs from the measured case")
    return strength


def _png_matches(image, expected):
    encoded = io.BytesIO()
    image.save(encoded, format="PNG")
    payload = encoded.getvalue()
    return len(payload) == expected["bytes"] and hashlib.sha256(payload).hexdigest() == expected["sha256"]


def replay_warm_amber_raw8_cfp(native, *, owner, raw_path, case_id, archived_off=None):
    """Return the images, native receipt and archived PNG matches for one measured case."""
    check = owner.check
    check()
    _verify_sources()
    evidence = _read_evidence()
    rows = {row["case"]: row for row in evidence["cases"]}
    row = rows[case_id]
    sampler = evidence["native_sampler"]
    strength = sampler["nominal_strength"]
    expected_constant = row["constant"]["effective_strength"]
    if not (isfinite(strength) and 0 < strength and 0 <= expected_constant <= strength):
        raise ValueError("the measured CFP and constant strengths are outside the qualified range")
    capabilities = Capabilities("lora", render=True)
    for value in (0.0, strength, expected_constant):
        capabilities.check("render", strength=value, references=False, output_type="pil")
    calibration = evidence["calibration"]
    gate_seed = _gate_seed(row)
    if gate_seed != row["cfp"]["gate_seed"]:
        raise ValueError("the exact measured prompt and seed do not match the recorded request")
    _verify_native_raw(native, raw_path, evidence, check)

    off = None
    if archived_off is not None:
        payload = Path(archived_off).read_bytes()
        if (len(payload) != row["OFF"]["bytes"]
                or hashlib.sha256(payload).hexdigest() != row["OFF"]["sha256"]):
            raise ValueError("the archived OFF PNG differs from the measured case")
        with Image.open(io.BytesIO(payload)) as source:
            off = source.copy()

    def native_render(*, prompt, seed, resolution, strength):
        check()
        return native.generate(prompt, strength, seed, resolution)

    inference = SimpleNamespace(
        owner=SimpleNamespace(check=check, components=SimpleNamespace(native=native)),
        session=SimpleNamespace(capabilities=capabilities),
        render=native_render,
    )
    request = dict(prompt=PROMPTS[case_id], seed=row["seed"], resolution=sampler["resolution"][0])
    gate = adaptive_steering.ActivationGate("cfp", calibration["accepted"]["threshold"], seed=gate_seed)
    alignment = _alignment_callback(calibration["feature_fit"], device=native.device, strength=strength)
    cfp, receipt = adaptive.render(inference, gate=gate, strength=strength, alignment=alignment, **request)
    constant_strength = _match_receipt(receipt, row, sampler)
    constant = native_render(strength=constant_strength, **request)
    if off is None:
        off = native_render(strength=0.0, **request)
    return dict(
        images={"cfp": cfp, "constant": constant, "off": off},
        receipt=receipt,
        constant_strength=constant_strength,
        archived_png_match={"cfp": _png_matches(cfp, row["cfp"]["output"]),
                            "constant": _png_matches(constant, row["constant"]["output"]),
                            "off": archived_off is not None or _png_matches(off, row["OFF"])},
    )
