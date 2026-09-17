from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    host: str = Field("0.0.0.0", validation_alias="MODEL_SERVER_HOST")
    port: int = Field(8010, validation_alias="SPEECH_PORT")
    gpu_id: str = Field("5", validation_alias="SPEECH_GPU_ID")

    asr_model: str = Field("Qwen/Qwen3-ASR-0.6B", validation_alias="ASR_MODEL")
    tts_model: str = Field(
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", validation_alias="TTS_MODEL"
    )
    tts_language: str = Field("Korean", validation_alias="TTS_LANGUAGE")
    tts_speaker: str = Field("Sohee", validation_alias="TTS_SPEAKER")
    tts_attention_backend: Literal["flash_attention_2", "sdpa", "eager"] = Field(
        "flash_attention_2", validation_alias="TTS_ATTENTION_BACKEND"
    )
    # Measured on the A5000: 492ms (sdpa) -> 351ms (FA2) for a 2.2s Korean
    # utterance. Falls back to sdpa when flash_attn is unavailable.
    asr_attention_backend: Literal["flash_attention_2", "sdpa", "eager"] = Field(
        "flash_attention_2", validation_alias="ASR_ATTENTION_BACKEND"
    )

    # The streaming Qwen3-TTS runtime on its own port supersedes this in-process
    # TTS. Loading both wastes GPU 5 and slows ASR through contention, so this
    # service can be started ASR-only.
    tts_enabled: bool = Field(True, validation_alias="SPEECH_TTS_ENABLED")

    api_key: str = Field("", validation_alias="MODEL_SERVER_API_KEY")
    verbose: bool = Field(False, validation_alias="MODEL_SERVER_VERBOSE")
    asr_max_new_tokens: int = Field(512, validation_alias="ASR_MAX_NEW_TOKENS")
    max_audio_bytes: int = Field(50 * 1024 * 1024, validation_alias="MAX_AUDIO_BYTES")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
