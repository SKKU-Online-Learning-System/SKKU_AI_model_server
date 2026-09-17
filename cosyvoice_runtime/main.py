from __future__ import annotations

import asyncio
import hmac
import logging
import os
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterator, Protocol

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("cosyvoice_runtime")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    model_dir: Path
    prompt_wav: Path
    prompt_text: str
    speaker: str = "cosyvoice"
    fp16: bool = True
    warmup_text: str = "안녕하세요."
    api_key: str = ""
    # First-chunk latency is dominated by how many speech tokens the flow decoder
    # waits for: token_hop_len + prompt_token_pad + pre_lookahead_len. Lowering the
    # hop shortens that wait at the cost of more (cheaper) decoder invocations.
    # 0 keeps the model's trained default.
    token_hop_len: int = 0

    @classmethod
    def from_env(cls) -> "Settings":
        hf_home = Path(os.getenv("HF_HOME", str(Path.home() / ".cache/huggingface")))
        return cls(
            model_dir=Path(
                os.getenv(
                    "COSYVOICE_MODEL_DIR",
                    str(hf_home / "Fun-CosyVoice3-0.5B-2512"),
                )
            ),
            prompt_wav=Path(os.getenv("COSYVOICE_PROMPT_WAV", "")),
            prompt_text=os.getenv("COSYVOICE_PROMPT_TEXT", ""),
            speaker=os.getenv("COSYVOICE_SPEAKER", "cosyvoice"),
            fp16=_env_bool("COSYVOICE_FP16", True),
            warmup_text=os.getenv("COSYVOICE_WARMUP_TEXT", "안녕하세요."),
            api_key=os.getenv("MODEL_SERVER_API_KEY", ""),
            token_hop_len=int(os.getenv("COSYVOICE_TOKEN_HOP_LEN", "0") or 0),
        )


class SpeechRequest(BaseModel):
    input: str = Field(min_length=1, max_length=20_000)
    voice: str | None = None
    language: str | None = None
    response_format: str = "pcm"
    instruct: str | None = None
    stream: bool = True
    # Optional per-request override. token2wav costs ~600ms almost regardless of
    # how many tokens it is given, so a small hop buys a faster first packet at
    # the price of throughput. Worth it only for the first chunk of a turn.
    hop_len: int | None = Field(default=None, ge=2, le=100)


class TTSBackend(Protocol):
    sample_rate: int

    def load(self) -> None: ...

    def stream_pcm(self, text: str, hop_len: int | None = None) -> Iterator[bytes]: ...


class CosyVoiceBackend:
    sample_rate = 24_000

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None
        self._initial_hop_len = 25
        self._lock = threading.Lock()

    def load(self) -> None:
        if not self.settings.model_dir.is_dir():
            raise FileNotFoundError(f"CosyVoice model directory not found: {self.settings.model_dir}")
        if not self.settings.prompt_wav.is_file() or not self.settings.prompt_text:
            raise ValueError("COSYVOICE_PROMPT_WAV and COSYVOICE_PROMPT_TEXT are required")

        from cosyvoice.cli.cosyvoice import AutoModel

        self.model = AutoModel(model_dir=str(self.settings.model_dir), fp16=self.settings.fp16)
        self.sample_rate = int(self.model.sample_rate)
        self._initial_hop_len = self.settings.token_hop_len or int(self.model.model.token_hop_len)
        self.model.add_zero_shot_spk(
            self.settings.prompt_text,
            str(self.settings.prompt_wav),
            self.settings.speaker,
        )
        if self.settings.warmup_text:
            list(self.stream_pcm(self.settings.warmup_text))
        logger.info(
            "CosyVoice ready model_dir=%s speaker=%s sample_rate=%d token_hop_len=%d",
            self.settings.model_dir,
            self.settings.speaker,
            self.sample_rate,
            self._initial_hop_len,
        )

    def stream_pcm(self, text: str, hop_len: int | None = None) -> Iterator[bytes]:
        if self.model is None:
            raise RuntimeError("CosyVoice model is not loaded")

        with self._lock:
            # The upstream runtime grows this value across requests, which makes
            # later first chunks much larger. Restore the model's trained chunk size.
            self.model.model.token_hop_len = hop_len or self._initial_hop_len
            outputs = self.model.inference_zero_shot(
                text,
                "",
                "",
                zero_shot_spk_id=self.settings.speaker,
                stream=True,
                text_frontend=False,
            )
            for output in outputs:
                samples = output["tts_speech"].detach().float().cpu().numpy().reshape(-1)
                yield (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def create_app(
    settings: Settings | None = None,
    backend: TTSBackend | None = None,
) -> FastAPI:
    cfg = settings or Settings.from_env()
    tts = backend or CosyVoiceBackend(cfg)
    state = {"ready": False, "error": None}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            await asyncio.to_thread(tts.load)
            state["ready"] = True
        except Exception as exc:
            logger.exception("CosyVoice loading failed")
            state["error"] = f"{type(exc).__name__}: {exc}"
        yield

    app = FastAPI(title="SKKU CosyVoice TTS Server", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def optional_bearer_auth(request: Request, call_next):
        if cfg.api_key:
            actual = request.headers.get("authorization", "")
            if not hmac.compare_digest(actual, f"Bearer {cfg.api_key}"):
                return Response(
                    status_code=401,
                    content='{"detail":"Unauthorized"}',
                    media_type="application/json",
                )
        return await call_next(request)

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok" if state["ready"] else "error",
            "ready": state["ready"],
            "model": str(cfg.model_dir),
            "error": state["error"],
        }

    @app.post("/v1/audio/speech")
    async def synthesize(body: SpeechRequest):
        if not state["ready"]:
            raise HTTPException(status_code=503, detail=state["error"] or "Model is loading")
        if body.response_format != "pcm":
            raise HTTPException(status_code=422, detail="CosyVoice streaming supports pcm only")
        if body.instruct:
            raise HTTPException(status_code=422, detail="CosyVoice instruct mode is not enabled")
        if body.voice and body.voice.casefold() != cfg.speaker.casefold():
            raise HTTPException(status_code=422, detail=f"Unsupported voice: {body.voice}")

        headers = {
            "X-Audio-Sample-Rate": str(tts.sample_rate),
            "X-Audio-Channels": "1",
            "X-Audio-Sample-Format": "pcm_s16le",
        }
        chunks = tts.stream_pcm(body.input, body.hop_len)
        if body.stream:
            return StreamingResponse(
                chunks,
                media_type=f"audio/L16;rate={tts.sample_rate};channels=1",
                headers={**headers, "X-Streaming": "true"},
            )

        start = perf_counter()
        payload = b"".join(chunks)
        inference_ms = round((perf_counter() - start) * 1000)
        duration_ms = round(len(payload) / 2 * 1000 / tts.sample_rate)
        return Response(
            content=payload,
            media_type=f"audio/L16;rate={tts.sample_rate};channels=1",
            headers={
                **headers,
                "X-Inference-Ms": str(inference_ms),
                "X-Audio-Duration-Ms": str(duration_ms),
            },
        )

    return app


app = create_app()
