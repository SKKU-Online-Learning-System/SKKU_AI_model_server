"""Render the voice_clone reference clip.

QWEN_TTS_MODE=voice_clone conditions every request on one recording so the timbre
stays stable across the several requests a turn is split into. The reference here
is rendered from a CustomVoice speaker rather than supplied by a person, so the
built-in timbre is kept while gaining clone-mode stability.

Several takes are rendered and the one with the least narrowband energy is kept,
because takes vary noticeably in how clean they are.
"""

from __future__ import annotations

import os
import wave
from pathlib import Path

import numpy as np
import torch
from faster_qwen3_tts import FasterQwen3TTS

SAMPLE_RATE = 24_000
TAKES = int(os.getenv("QWEN_TTS_REFERENCE_TAKES", "6"))
MODEL = os.getenv(
    "QWEN_TTS_REFERENCE_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
)
SPEAKER = os.getenv("QWEN_TTS_REFERENCE_SPEAKER", "ryan")
LANGUAGE = os.getenv("QWEN_TTS_LANGUAGE", "Korean")
TEXT = os.getenv(
    "QWEN_TTS_REFERENCE_TEXT",
    "안녕하세요. 오늘은 운영체제의 가상 메모리에 대해 함께 살펴보겠습니다. "
    "천천히 설명드리니 편하게 들어주세요.",
)
OUTPUT = Path(
    os.getenv(
        "QWEN_TTS_REF_WAV",
        str(Path(__file__).resolve().parent.parent / "samples" / "qwen_ryan_reference.wav"),
    )
)


def narrowband_score(samples: np.ndarray) -> float:
    """Strength of the strongest narrow spectral peak, relative to the band."""
    size, hop = 1024, 256
    window = np.hanning(size)
    frames = (len(samples) - size) // hop + 1
    spectra = np.abs(
        np.array(
            [np.fft.rfft(samples[i * hop : i * hop + size] * window) for i in range(frames)]
        )
    )
    freqs = np.fft.rfftfreq(size, 1 / SAMPLE_RATE)
    band = (freqs > 1500) & (freqs < 9000)
    in_band = spectra[:, band] + 1e-6
    peakiness = in_band.max(axis=1) / np.median(in_band, axis=1)
    energy = spectra.sum(axis=1)
    voiced = energy > np.percentile(energy, 60) * 0.2
    if not voiced.any():
        return float("inf")
    return float(np.percentile(peakiness[voiced], 99.9))


def render(model: FasterQwen3TTS) -> np.ndarray:
    parts = [
        np.asarray(chunk, dtype=np.float32).reshape(-1)
        for chunk, _rate, _timing in model.generate_custom_voice_streaming(
            text=TEXT, speaker=SPEAKER, language=LANGUAGE, chunk_size=4
        )
    ]
    return np.concatenate(parts) * 32767


def main() -> None:
    model = FasterQwen3TTS.from_pretrained(
        MODEL,
        device="cuda",
        dtype=torch.bfloat16,
        attn_implementation=os.getenv("QWEN_TTS_ATTENTION_BACKEND", "sdpa"),
        cache_dir=os.getenv("HF_HOME") or None,
    )
    render(model)  # warm up and capture the CUDA graph

    best: tuple[np.ndarray, float] | None = None
    for take in range(TAKES):
        samples = render(model)
        score = narrowband_score(samples)
        loud = np.flatnonzero(np.abs(samples) > 200)
        samples = samples[max(0, loud[0] - 480) : loud[-1] + 1]
        print(f"take {take}: {len(samples) / SAMPLE_RATE:5.2f}s  narrowband={score:6.0f}")
        if best is None or score < best[1]:
            best = (samples, score)

    assert best is not None
    samples, score = best
    samples = samples * min(1.0, 22000 / max(np.abs(samples).max(), 1))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(OUTPUT), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(np.clip(samples, -32768, 32767).astype("<i2").tobytes())

    print(f"\nkept the take scoring {score:.0f} -> {OUTPUT}")
    print("Set QWEN_TTS_REF_TEXT to exactly this transcript:")
    print(f"  {TEXT}")


if __name__ == "__main__":
    main()
