from __future__ import annotations

import numpy as np

from speech_server.asr_service import ASRResult
from speech_server.tts_service import TTSResult


class FakeASR:
    def load(self) -> None:
        return None

    def transcribe(self, samples, sample_rate, language=None) -> ASRResult:
        return ASRResult(text="테스트 음성입니다.", language=language or "Korean", inference_ms=10)


class FakeTTS:
    def load(self) -> None:
        return None

    def supported_speakers(self) -> list[str]:
        return ["Sohee", "Ryan"]

    def synthesize(self, text, speaker, language, instruct=None) -> TTSResult:
        samples = np.zeros(24_000, dtype=np.float32)
        return TTSResult(samples=samples, sample_rate=24_000, inference_ms=20)
