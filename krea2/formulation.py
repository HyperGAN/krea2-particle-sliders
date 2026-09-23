"""Lock Krea2 training to the shared winning formulation.

Hub id, Comfy node, turbo sample numbers, and prompt cards stay in this
repo. Gmix architecture and the formulation overlay come from
``particle-sliders-core``. This module does not copy ``RoutedMLP``,
``GradRegularizer``, or ``locked_shared``, and it does not vendor ParticleGAN.
"""

from __future__ import annotations

PIN = (
    "particle-sliders-core @ git+https://github.com/HyperGAN/particle-sliders.git"
    "@a119ca1ecd3d5d6c437065839d22739b04f2f4d8"
    "#subdirectory=packages/particle-sliders-core"
)


def lock_winning_formulation():
    """Return the shared stamp after ``require`` accepts it.

    Products call ``winning_formulation()``. They do not restate the game.
    """
    try:
        from particle_sliders import winning_formulation
    except ImportError as exc:
        raise ImportError(
            "Training requires particle-sliders-core "
            f"({PIN}). Install it from requirements.txt."
        ) from exc
    stamp = winning_formulation()
    stamp.require(stamp.as_dict())
    return stamp


def formulation_record(stamp) -> dict[str, object]:
    """JSON-safe identity of the stamp this process locked. Not a second recipe."""
    spec = stamp.as_dict()
    return {
        "pin": PIN,
        "architecture_id": stamp.architecture_id,
        "family": stamp.family,
        "formulation_id": stamp.formulation_id,
        "formulation_provisional": bool(stamp.formulation_provisional),
        "recipe_name": spec["recipe_name"],
        "critic": spec["critic"],
        "parts": spec["parts"],
        "particle_dim": spec["particle_dim"],
        "adapter_rank": spec["adapter_rank"],
        "noise_decay_steps": spec["noise_decay_steps"],
        "noise_hold_ratio": spec["noise_hold_ratio"],
        "edit_noise_ratio": spec["edit_noise_ratio"],
    }


def cpu_gmix_step(stamp, *, step: int = 0) -> dict[str, float | str | bool]:
    """One CPU step through the stamp's gmix builders.

    The modules, losses, noise curve, and b_cap penalty stay in
    particle-sliders-core / ParticleGAN. This function only calls them.
    """
    import torch

    torch.manual_seed(0)
    spec = stamp.spec
    bridge = stamp.bridge()
    particles = torch.nn.Parameter(
        torch.randn(int(spec["parts"]), int(spec["particle_dim"]))
    )
    hidden = torch.randn(2, int(spec["adapter_rank"]))
    edit = bridge(hidden, particles)
    targets = torch.randn(4, 8)
    neutrals = targets - 0.2
    critic = stamp.critic(targets, neutrals=neutrals)
    real_x = critic.normalize(targets).detach()
    fake_x = critic.normalize(neutrals).detach()
    real = critic(real_x)
    fake = critic(fake_x)
    d_loss_fn, g_loss_fn, vic_fn = stamp.losses()
    d_loss = d_loss_fn(real, fake)
    g_loss = g_loss_fn(real, fake)
    vic = vic_fn(particles)
    regularizer = stamp.regularizer()
    penalty, penalty_stats = regularizer.penalty(
        critic, real_x, fake_x, step=int(regularizer.lazy_k)
    )
    loss = g_loss + vic + edit.square().mean() + penalty
    loss.backward()
    sigma = float(stamp.noise_std_at(int(step), float(critic.edit_rms.detach())))
    grad = particles.grad
    return {
        "gmix_d_loss": float(d_loss.detach()),
        "gmix_g_loss": float(g_loss.detach()),
        "gmix_vic": float(vic.detach()),
        "gmix_penalty": float(penalty.detach()),
        "gmix_penalty_applied": bool(penalty_stats.get("applied")),
        "gmix_noise_std": sigma,
        "gmix_edit_mean_abs": float(edit.detach().abs().mean()),
        "gmix_particle_grad_norm": float(grad.detach().norm()) if grad is not None else 0.0,
        "gmix_regularizer_arm": str(regularizer.arm),
        "gmix_regularizer_coeff": float(regularizer.coeff),
        "gmix_regularizer_kappa": float(regularizer.kappa),
        "gmix_regularizer_lazy_k": int(regularizer.lazy_k),
    }
