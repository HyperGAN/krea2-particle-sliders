"""Schedule the original scoped Krea2 pipeline without changing its RAW identity."""
from contextlib import ExitStack
from math import isfinite
from sys import float_info


def render(inference, *, step_gains, strength=1.0, **request):
    """Apply one gain per native guidance-zero denoiser forward."""
    inference.owner.check()
    native = inference.owner.components.native
    steps = native.args.sample_steps
    if not isinstance(step_gains, (list, tuple)) or len(step_gains) != steps:
        raise ValueError("Krea2 requires one step gain per native sampling step")
    if type(strength) not in (int, float) or not isfinite(strength):
        raise ValueError("Krea2 nominal strength must be finite")
    if any(type(gain) not in (int, float) or not 0 <= gain <= float_info.max
           for gain in step_gains):
        raise ValueError("Krea2 step gains must be finite and nonnegative")
    values = tuple(strength * gain for gain in step_gains)
    for value in values:
        if not isfinite(value):
            raise ValueError("Krea2 effective strengths must be finite")
        inference.session.capabilities.check("render", strength=value, references=False,
                                             output_type="pil")

    step = 0
    scope = ExitStack()

    def before(_model, _args):
        nonlocal step
        if step >= steps:
            raise RuntimeError("Krea2 made more denoiser calls than scheduled")
        scope.enter_context(native.scale(values[step]))
        step += 1

    def after(_model, _args, _output):
        scope.close()

    with ExitStack() as handles:
        handles.callback(native.model.register_forward_pre_hook(before).remove)
        handles.callback(native.model.register_forward_hook(after, always_call=True).remove)
        handles.callback(scope.close)
        image = inference.render(strength=strength, **request)
    if step != steps:
        raise RuntimeError("Krea2 finished before applying all scheduled strengths")
    return image, {"effective_strengths": list(values), "denoiser_calls": step}
