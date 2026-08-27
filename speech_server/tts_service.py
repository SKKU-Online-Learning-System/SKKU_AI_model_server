from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import numpy as np

from .config import Settings


@dataclass(frozen=True)
class TTSResult:
    samples: np.ndarray
    sample_rate: int
    inference_ms: int

    @property
    def audio_duration_ms(self) -> int:
        return round(len(self.samples) * 1000 / self.sample_rate) if self.sample_rate else 0


class TTSService(Protocol):
    def load(self) -> None: ...

    def supported_speakers(self) -> list[str]: ...

    def synthesize(
        self, text: str, speaker: str, language: str, instruct: str | None = None
    ) -> TTSResult: ...


class QwenTTSService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None

    def load(self) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        self.model = Qwen3TTSModel.from_pretrained(
            self.settings.tts_model,
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )

    def supported_speakers(self) -> list[str]:
        if self.model is None:
            return []
        speakers = self.model.get_supported_speakers()
        return [str(s) for s in speakers]

    def synthesize(
        self, text: str, speaker: str, language: str, instruct: str | None = None
    ) -> TTSResult:
        if self.model is None:
            raise RuntimeError("TTS model is not loaded.")

        kwargs = {
            "text": text,
            "language": language,
            "speaker": speaker,
        }
        if instruct:
            kwargs["instruct"] = instruct

        start = perf_counter()
        wavs, sample_rate = self.model.generate_custom_voice(**kwargs)
        elapsed_ms = round((perf_counter() - start) * 1000)
        samples = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        return TTSResult(samples=samples, sample_rate=int(sample_rate), inference_ms=elapsed_ms)
