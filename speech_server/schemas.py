from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok", "loading", "error"]
    ready: bool
    gpu: str
    asr_model: str
    tts_model: str
    error: str | None = None


class TranscriptionResponse(BaseModel):
    text: str
    language: str
    audio_duration_ms: int
    inference_ms: int


class SpeechRequest(BaseModel):
    input: str = Field(min_length=1, max_length=20_000)
    voice: str | None = None
    language: str | None = None
    response_format: Literal["pcm", "wav"] = "pcm"
    instruct: str | None = Field(default=None, max_length=1_000)
