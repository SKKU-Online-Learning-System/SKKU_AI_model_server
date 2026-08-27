from __future__ import annotations

import io
import math
import wave
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


ASR_SAMPLE_RATE = 16_000
TTS_OUTPUT_SAMPLE_RATE = 24_000


class AudioValidationError(ValueError):
    pass


@dataclass(frozen=True)
class DecodedAudio:
    samples: np.ndarray
    sample_rate: int

    @property
    def duration_ms(self) -> int:
        if self.sample_rate <= 0:
            return 0
        return round(len(self.samples) * 1000 / self.sample_rate)


def _mono(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples)
    if samples.ndim == 1:
        return samples
    if samples.ndim == 2:
        return samples.mean(axis=1)
    raise AudioValidationError("Audio must be mono or stereo.")


def resample_audio(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate <= 0 or target_rate <= 0:
        raise AudioValidationError("Invalid sample rate.")
    samples = np.asarray(samples, dtype=np.float32)
    if source_rate == target_rate:
        return samples
    divisor = math.gcd(source_rate, target_rate)
    up = target_rate // divisor
    down = source_rate // divisor
    return resample_poly(samples, up, down).astype(np.float32, copy=False)


def decode_upload(data: bytes, filename: str | None, content_type: str | None) -> DecodedAudio:
    if not data:
        raise AudioValidationError("Empty audio file.")

    is_wav = (
        (filename or "").lower().endswith(".wav")
        or (content_type or "").lower() in {"audio/wav", "audio/x-wav", "audio/wave"}
        or data[:4] == b"RIFF"
    )

    if is_wav:
        try:
            samples, sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        except Exception as exc:
            raise AudioValidationError(f"Invalid WAV audio: {exc}") from exc
        samples = _mono(samples).astype(np.float32, copy=False)
    else:
        if len(data) % 2:
            raise AudioValidationError("Raw PCM16 payload must have an even number of bytes.")
        pcm = np.frombuffer(data, dtype="<i2")
        if pcm.size == 0:
            raise AudioValidationError("Empty PCM16 audio.")
        samples = pcm.astype(np.float32) / 32768.0
        sample_rate = ASR_SAMPLE_RATE

    samples = resample_audio(samples, sample_rate, ASR_SAMPLE_RATE)
    if not np.isfinite(samples).all():
        raise AudioValidationError("Audio contains non-finite samples.")
    return DecodedAudio(samples=samples, sample_rate=ASR_SAMPLE_RATE)


def float_to_pcm16(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    samples = np.clip(samples, -1.0, 1.0)
    return np.rint(samples * 32767.0).astype("<i2")


def encode_pcm16(samples: np.ndarray, source_rate: int, target_rate: int = TTS_OUTPUT_SAMPLE_RATE) -> bytes:
    mono = _mono(np.asarray(samples, dtype=np.float32))
    normalized = resample_audio(mono, source_rate, target_rate)
    return float_to_pcm16(normalized).tobytes()


def encode_wav_pcm16(
    samples: np.ndarray, source_rate: int, target_rate: int = TTS_OUTPUT_SAMPLE_RATE
) -> bytes:
    pcm = encode_pcm16(samples, source_rate=source_rate, target_rate=target_rate)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(target_rate)
        wav_file.writeframes(pcm)
    return buffer.getvalue()
