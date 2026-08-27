from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import numpy as np

from .config import Settings


@dataclass(frozen=True)
class ASRResult:
    text: str
    language: str
    inference_ms: int


class ASRService(Protocol):
    def load(self) -> None: ...

    def transcribe(self, samples: np.ndarray, sample_rate: int, language: str | None) -> ASRResult: ...


class QwenASRService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None

    def load(self) -> None:
        import torch
        from qwen_asr import Qwen3ASRModel

        self.model = Qwen3ASRModel.from_pretrained(
            self.settings.asr_model,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            max_inference_batch_size=1,
            max_new_tokens=self.settings.asr_max_new_tokens,
        )

    def transcribe(
        self, samples: np.ndarray, sample_rate: int, language: str | None = None
    ) -> ASRResult:
        if self.model is None:
            raise RuntimeError("ASR model is not loaded.")

        start = perf_counter()
        results = self.model.transcribe(
            audio=(samples, sample_rate),
            language=language or None,
        )
        elapsed_ms = round((perf_counter() - start) * 1000)
        result = results[0]
        return ASRResult(
            text=result.text,
            language=result.language or (language or ""),
            inference_ms=elapsed_ms,
        )
