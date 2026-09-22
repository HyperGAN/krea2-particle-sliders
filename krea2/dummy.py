"""CPU stand-in for the turbo-bbox UNI step. Never loads Hub weights."""

from __future__ import annotations

import hashlib
import struct
import zlib
from pathlib import Path

from krea2.prompts import SliderPrompt, unused_words_for, word_tokens
from krea2.uni import mse, plus_neu_teachers


class Loss:
    def __init__(self, value: float, param: "Param"):
        self.value = float(value)
        self.param = param

    def detach(self) -> float:
        return self.value

    def backward(self) -> None:
        self.param.grad = self.value


class Param:
    def __init__(self) -> None:
        self.data = 0.1
        self.grad = 0.0


def _vec(text: str, dim: int = 4) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [((digest[i] / 255.0) * 2.0 - 1.0) for i in range(dim)]


class DummyBackend:
    """Deterministic velocities. Scale 1 adds the trainable scalar."""

    def __init__(self, *, seed: int = 0, lora_targets: str = "dit") -> None:
        self.dim = 4
        self.seed = int(seed)
        self.lora_targets = lora_targets
        self.encoder_lora = lora_targets in ("te", "dit+te", "text_encoder")
        self.param = Param()
        self.loaded_te_lora: str | None = None
        self._scale = 0.0

    def begin_step(self) -> None:
        return None

    def sample_latents(self, device: object = None) -> list[float]:
        del device
        return _vec(f"latent-{self.seed}", self.dim)

    def set_adapter_scale(self, scale: float) -> None:
        self._scale = float(scale)

    def trainable_parameters(self) -> list[Param]:
        return [self.param]

    def predict_v(
        self,
        prompt: str,
        z: list[float],
        *,
        scale: float = 0.0,
        pin_unused: bool = False,
        neu_prompt: str | None = None,
        unused_words: list[str] | None = None,
    ) -> list[float]:
        del z, pin_unused, neu_prompt, unused_words
        base = _vec(prompt or "", self.dim)
        shift = float(scale) * self.param.data
        return [value + shift for value in base]

    def encode_text(self, prompt: str, *, frozen: bool = False) -> tuple[list[float], list[str]]:
        del frozen
        tokens = word_tokens(prompt) or [""]
        return _vec(prompt or "", self.dim), tokens

    def load_te_adapter(self, path: str | Path) -> None:
        self.loaded_te_lora = str(path)

    def generate(
        self,
        prompt: str,
        *,
        seed: int = 0,
        num_steps: int = 2,
        guidance: float = 0.0,
        scale: float = 0.0,
        height: int = 8,
        width: int = 8,
        **kwargs: object,
    ) -> "TinyImage":
        del kwargs
        tone = int(abs(hash((prompt, seed, num_steps, guidance, scale))) % 200) + 20
        return TinyImage(width=width, height=height, tone=tone)


class TinyImage:
    def __init__(self, *, width: int, height: int, tone: int) -> None:
        self.width = int(width)
        self.height = int(height)
        self.tone = int(tone)

    def save(self, path: str | Path) -> None:
        _write_png(Path(path), self.width, self.height, self.tone)


def _write_png(path: Path, width: int, height: int, tone: int) -> None:
    raw = b"".join(
        b"\x00" + bytes([tone, (tone // 2) & 255, 255 - tone]) * width
        for _ in range(height)
    )

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


def step_loss(
    backend: DummyBackend,
    prompt: SliderPrompt,
    z: list[float],
    *,
    guidance: float,
    hold_weight: float,
    lm_target: str,
) -> tuple[Loss, dict[str, float]]:
    """One UNI step on the distilled card. Row ``guidance_scale`` is ignored."""
    del lm_target
    with_scale0 = backend.predict_v(prompt.positive, z, scale=0.0)
    v_neu = backend.predict_v(prompt.neutral, z, scale=0.0)
    v_uncond = backend.predict_v(prompt.unconditional, z, scale=0.0)
    tgt_plus, tgt_zero = plus_neu_teachers(
        with_scale0, v_neu, v_uncond, guidance=guidance
    )
    pred_plus = backend.predict_v(prompt.positive, z, scale=1.0)
    pred_zero = backend.predict_v(prompt.neutral, z, scale=0.0)
    value = mse(pred_plus, tgt_plus) + mse(pred_zero, tgt_zero)
    unused = set(unused_words_for(prompt))
    pos_tokens = set(word_tokens(prompt.positive))
    # Concept words are not held. Unused attributes contribute a constant.
    hold = 0.0 if not unused else 0.01 * len(unused - pos_tokens)
    if hold_weight > 0.0:
        value = value + float(hold_weight) * hold
    stats = {
        "loss": float(value),
        "hold": float(hold),
        "minus_teacher": 0.0,
    }
    return Loss(value, backend.param), stats
