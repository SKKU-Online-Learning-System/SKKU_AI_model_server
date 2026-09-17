from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import numpy as np

from .config import Settings

logger = logging.getLogger(__name__)


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
        self.attention_backend = "unknown"

    def _load_model(self, *, attention_backend: str):
        import torch
        from qwen_asr import Qwen3ASRModel

        logger.info(
            "Loading Qwen3 ASR model=%s attention_backend=%s dtype=bfloat16",
            self.settings.asr_model,
            attention_backend,
        )
        return Qwen3ASRModel.from_pretrained(
            self.settings.asr_model,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            max_inference_batch_size=1,
            max_new_tokens=self.settings.asr_max_new_tokens,
            attn_implementation=attention_backend,
        )

    def load(self) -> None:
        import torch

        backend = self.settings.asr_attention_backend
        if backend == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
            logger.warning(
                "ASR_ATTENTION_BACKEND=flash_attention_2 but flash_attn is not installed; "
                "falling back to sdpa"
            )
            backend = "sdpa"

        try:
            self.model = self._load_model(attention_backend=backend)
        except Exception:
            if backend != "flash_attention_2":
                raise
            logger.exception("FlashAttention 2 ASR loading failed; retrying with PyTorch SDPA")
            self.model = None
            torch.cuda.empty_cache()
            backend = "sdpa"
            self.model = self._load_model(attention_backend=backend)

        self.attention_backend = backend
        logger.info("Qwen3 ASR ready attention_backend=%s", self.attention_backend)

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
