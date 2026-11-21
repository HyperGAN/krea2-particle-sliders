"""Activation-dependent allocation for the owned Krea2 RAW denoising path."""
from contextlib import ExitStack
from math import isfinite

import torch
from particle_sliders.adaptive_steering import ActivationGate


@torch.no_grad()
def render(inference, *, gate: ActivationGate, strength, alignment=None, **request):
    """Gate native LoRA strength once per guidance-zero Euler interval.

    ``alignment(latents, timestep)`` reads one calibrated signed projection
    from the packed [1, position, channel] input and the normalized timestep
    tensor before the denoiser. Treat the inputs as read-only. Pass a fresh
    shared gate per render. The matched random control takes no readout.
    """
    inference.owner.check()
    if gate.usage.eligible:
        raise ValueError("Krea2 adaptive rendering requires a fresh gate per request")
    if type(strength) not in (int, float) or not isfinite(strength) or strength == 0:
        raise ValueError("Krea2 adaptive edit strength must be finite and nonzero")
    if alignment is not None and not callable(alignment):
        raise TypeError("Krea2 alignment must be a callable")
    native = inference.owner.components.native
    steps = native.args.sample_steps
    if type(steps) is not int or steps <= 0:
        raise ValueError("Krea2 adaptive rendering requires a positive native step count")
    if native.args.sample_guidance != 0:
        raise ValueError("Krea2 adaptive rendering requires guidance zero")
    for value in (0.0, strength):
        inference.session.capabilities.check(
            "render", strength=value, references=False, output_type="pil",
        )

    trace = []
    scheduler = None
    sigmas = None
    scope = ExitStack()

    def before(_model, _args, kwargs):
        nonlocal scheduler, sigmas
        index = len(trace)
        active = native.pipe.scheduler
        if native.pipe.do_classifier_free_guidance or active.order != 1:
            raise RuntimeError("Krea2 adaptive rendering requires one conditional Euler forward per interval")
        if index == 0:
            scheduler = active
            sigmas = active.sigmas.detach().tolist()
            if (len(sigmas) != steps + 1 or not all(isfinite(sigma) for sigma in sigmas)
                    or not all(left > right for left, right in zip(sigmas, sigmas[1:]))):
                raise RuntimeError("Krea2 native sigma grid differs from the adaptive schedule")
        expected_step = index if index else None
        if (index >= steps or active is not scheduler or active.begin_index != 0
                or active.step_index != expected_step):
            raise RuntimeError("Krea2 adaptive gate requires exactly one forward per Euler interval")
        latents, timestep = kwargs.get("hidden_states"), kwargs.get("timestep")
        if (not isinstance(latents, torch.Tensor) or latents.ndim != 3 or latents.shape[0] != 1
                or not isinstance(timestep, torch.Tensor) or timestep.shape != (1,)):
            raise RuntimeError("Krea2 adaptive rendering requires one packed latent and timestep")
        score = None if alignment is None else float(alignment(latents.detach(), timestep.detach()))
        if score is not None and not isfinite(score):
            raise ValueError("Krea2 alignment must return a finite scalar")
        decision = gate.step(score)
        applied = strength if decision.fired else 0.0
        scope.enter_context(native.scale(applied))
        trace.append(dict(interval=index, sigma_start=sigmas[index], sigma_end=sigmas[index + 1],
                          alignment=score, probability=decision.probability, fired=decision.fired,
                          effective_strength=applied))

    def after(_model, _args, _output):
        scope.close()

    with ExitStack() as handles:
        handles.callback(native.model.register_forward_pre_hook(before, with_kwargs=True).remove)
        handles.callback(native.model.register_forward_hook(after, always_call=True).remove)
        handles.callback(scope.close)
        image = inference.render(strength=0.0, **request)
    if len(trace) != steps or scheduler.step_index != steps:
        raise RuntimeError("Krea2 finished before applying adaptive strength to every Euler interval")
    usage = gate.usage
    return image, dict(
        effective_strengths=[row["effective_strength"] for row in trace],
        denoiser_calls=len(trace), decisions=trace, fired=usage.fired, eligible=usage.eligible,
        gate_rate=usage.rate,
        sigma_weighted_dose=sum((row["sigma_start"] - row["sigma_end"])
                                * abs(row["effective_strength"]) for row in trace),
    )
