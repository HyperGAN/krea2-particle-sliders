"""UNI teachers for the distilled turbo-bbox card.

CFG 0 leaves the + teacher as ``v(pos)``. Raw CFG 4.5 composes
``v(pos) + 4.5 * (v(pos) - v(''))``. Scale 0 is always ``v(neu)``.
Minus is not a teacher. These helpers do not import torch.
"""

from __future__ import annotations

from krea2.defaults import TURBO_MU


def cfg_compose(v_cond: list[float], v_uncond: list[float], guidance: float) -> list[float]:
    if guidance and float(guidance) > 0.0:
        scale = float(guidance)
        return [c + scale * (c - u) for c, u in zip(v_cond, v_uncond)]
    return list(v_cond)


def plus_neu_teachers(
    v_pos: list[float],
    v_neu: list[float],
    v_uncond: list[float] | None = None,
    *,
    guidance: float = 0.0,
) -> tuple[list[float], list[float]]:
    if v_uncond is not None and float(guidance) > 0.0:
        return cfg_compose(v_pos, v_uncond, guidance), list(v_neu)
    return list(v_pos), list(v_neu)


def scheduler_mu(*, is_distilled: bool, mu: float | None = None) -> float | None:
    """Pinned mu, else 1.15 when distilled, else None (caller shifts)."""
    if mu is not None:
        return float(mu)
    if is_distilled:
        return float(TURBO_MU)
    return None


def mse(pred: list[float], target: list[float]) -> float:
    if len(pred) != len(target) or not pred:
        raise ValueError("velocity vectors must be the same non-empty length")
    return sum((p - t) ** 2 for p, t in zip(pred, target)) / len(pred)
