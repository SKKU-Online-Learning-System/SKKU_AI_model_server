"""Isolated streaming Qwen3-TTS runtime.

Serves the same ``POST /v1/audio/speech`` streaming contract as the CosyVoice
runtime so the application can switch backends with ``TTS_BASE_URL`` alone.

Qwen3-TTS uses a 12 Hz speech tokenizer (CosyVoice runs at 25 Hz), so half as
many autoregressive steps are needed per second of audio. ``faster-qwen3-tts``
adds CUDA graph capture on top of that.
"""

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

logger = logging.getLogger("qwen_tts_runtime")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # "voice_clone" conditions every request on the same reference recording,
    # which is what keeps the timbre stable across the several requests a single
    # turn is split into. A CustomVoice speaker id is a much weaker conditioning:
    # measured across requests, the spectral distance was 1.52x the variation
    # inside one utterance and loudness moved by up to 1.67x, both audible as the
    # voice changing mid-answer. Cloning brought those to 0.65x and 1.25x.
    # Requires a *-Base model and a reference recording.
    mode: str = "voice_clone"
    model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    ref_wav: Path | None = None
    ref_text: str = ""
    speaker: str = "ryan"
    language: str = "Korean"
    attn_implementation: str = "sdpa"
    # Frames emitted per streaming chunk at 12 Hz. 12 == 1s of audio; smaller
    # yields the first packet sooner at the cost of more decoder invocations.
    chunk_size: int = 4
    # Qwen3-TTS pads roughly 0.6-0.95s of silence onto both ends of every
    # utterance. Left in, that silence is added latency at the start of a turn
    # and an unnatural pause at every chunk seam. Amplitude below this counts as
    # silence; 0 disables trimming.
    silence_threshold: int = 200
    # Keep this much audio before the first non-silent sample so a soft onset is
    # not clipped.
    silence_preroll_ms: int = 20
    # Keep this much of the trailing silence. Chunks are joined back to back by the
    # caller, so removing all of it deletes the pause a clause boundary needs and
    # butts two independently generated clips straight together.
    silence_keep_ms: int = 120
    # Each request realises its own loudness, so consecutive requests in one turn
    # differed by up to 1.4x (about 3dB), heard as the voice jumping. Normalising
    # every request towards the same target removes that step. Estimated from the
    # first emitted block so no latency is added; 0 disables.
    target_rms_dbfs: float = -20.0
    # Bound the correction so a quiet or loud opening cannot swing the whole
    # utterance, and keep headroom against clipping.
    max_gain: float = 2.5
    peak_ceiling: float = 0.89
    # Keep the model default. Lowering it destabilises the talker instead of
    # calming it: measured over 6 runs per setting, a 5-character fragment came
    # back as pure silence 1/6 of the time at 0.7 and 3/6 at 0.3, and 0.3 once
    # produced 21s of audio for 5 characters. Use `instruct` and the speaker
    # choice to shape delivery, not temperature.
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.05
    # Free-form style instruction understood by the CustomVoice models.
    instruct: str = ""
    warmup_text: str = "안녕하세요."
    cache_dir: Path | None = None
    api_key: str = ""

    @classmethod
    def from_env(cls) -> "Settings":
        cache = os.getenv("HF_HOME")
        return cls(
            mode=os.getenv("QWEN_TTS_MODE", "voice_clone"),
            model=os.getenv("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
            ref_wav=Path(ref) if (ref := os.getenv("QWEN_TTS_REF_WAV", "")) else None,
            ref_text=os.getenv("QWEN_TTS_REF_TEXT", ""),
            speaker=os.getenv("QWEN_TTS_SPEAKER", "ryan"),
            language=os.getenv("QWEN_TTS_LANGUAGE", "Korean"),
            attn_implementation=os.getenv("QWEN_TTS_ATTENTION_BACKEND", "sdpa"),
            chunk_size=int(os.getenv("QWEN_TTS_CHUNK_SIZE", "4") or 4),
            silence_threshold=int(os.getenv("QWEN_TTS_SILENCE_THRESHOLD", "200") or 0),
            silence_preroll_ms=int(os.getenv("QWEN_TTS_SILENCE_PREROLL_MS", "20") or 0),
            silence_keep_ms=int(os.getenv("QWEN_TTS_SILENCE_KEEP_MS", "120") or 0),
            target_rms_dbfs=float(os.getenv("QWEN_TTS_TARGET_RMS_DBFS", "-20") or 0),
            temperature=float(os.getenv("QWEN_TTS_TEMPERATURE", "0.9") or 0.9),
            top_k=int(os.getenv("QWEN_TTS_TOP_K", "50") or 50),
            top_p=float(os.getenv("QWEN_TTS_TOP_P", "1.0") or 1.0),
            repetition_penalty=float(os.getenv("QWEN_TTS_REPETITION_PENALTY", "1.05") or 1.05),
            instruct=os.getenv("QWEN_TTS_INSTRUCT", ""),
            warmup_text=os.getenv("QWEN_TTS_WARMUP_TEXT", "안녕하세요."),
            cache_dir=Path(cache) if cache else None,
            api_key=os.getenv("MODEL_SERVER_API_KEY", ""),
        )


class SpeechRequest(BaseModel):
    input: str = Field(min_length=1, max_length=20_000)
    voice: str | None = None
    language: str | None = None
    response_format: str = "pcm"
    instruct: str | None = None
    stream: bool = True
    temperature: float | None = Field(default=None, ge=0.05, le=2.0)
    top_k: int | None = Field(default=None, ge=1, le=2048)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    repetition_penalty: float | None = Field(default=None, ge=1.0, le=2.0)
    # Backend-specific streaming granularity; frames per chunk for this runtime.
    hop_len: int | None = Field(default=None, ge=1, le=100)


class TTSBackend(Protocol):
    sample_rate: int

    def load(self) -> None: ...

    def stream_pcm(
        self, text: str, hop_len: int | None = None, speaker: str | None = None,
        language: str | None = None, instruct: str | None = None,
        temperature: float | None = None, top_k: int | None = None,
        top_p: float | None = None, repetition_penalty: float | None = None,
    ) -> Iterator[bytes]: ...


class QwenTTSBackend:
    sample_rate = 24_000

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None
        # One GPU, one autoregressive decode loop with a captured CUDA graph.
        self._lock = threading.Lock()

    def load(self) -> None:
        import torch
        from faster_qwen3_tts import FasterQwen3TTS

        if self.settings.mode == "voice_clone":
            if self.settings.ref_wav is None or not self.settings.ref_wav.is_file():
                raise ValueError(
                    "QWEN_TTS_MODE=voice_clone requires QWEN_TTS_REF_WAV to point at "
                    "a readable reference recording"
                )
            if not self.settings.ref_text.strip():
                raise ValueError(
                    "QWEN_TTS_MODE=voice_clone requires QWEN_TTS_REF_TEXT to be the "
                    "exact transcript of the reference recording"
                )
        elif self.settings.mode != "custom_voice":
            raise ValueError(f"Unknown QWEN_TTS_MODE: {self.settings.mode}")

        logger.info(
            "Loading Qwen3-TTS mode=%s model=%s attn=%s chunk_size=%d",
            self.settings.mode,
            self.settings.model,
            self.settings.attn_implementation,
            self.settings.chunk_size,
        )
        self.model = FasterQwen3TTS.from_pretrained(
            self.settings.model,
            device="cuda",
            dtype=torch.bfloat16,
            attn_implementation=self.settings.attn_implementation,
            cache_dir=str(self.settings.cache_dir) if self.settings.cache_dir else None,
        )
        if self.settings.warmup_text:
            # Also captures the CUDA graph, which is what the first call would pay for.
            list(self.stream_pcm(self.settings.warmup_text))
        logger.info(
            "Qwen3-TTS ready mode=%s voice=%s language=%s sample_rate=%d",
            self.settings.mode,
            self.settings.ref_wav.name
            if self.settings.mode == "voice_clone" and self.settings.ref_wav
            else self.settings.speaker,
            self.settings.language,
            self.sample_rate,
        )

    def stream_pcm(
        self,
        text: str,
        hop_len: int | None = None,
        speaker: str | None = None,
        language: str | None = None,
        instruct: str | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
    ) -> Iterator[bytes]:
        if self.model is None:
            raise RuntimeError("Qwen3-TTS model is not loaded")

        shared = {
            "text": text,
            "language": language or self.settings.language,
            "chunk_size": hop_len or self.settings.chunk_size,
            "temperature": temperature or self.settings.temperature,
            "top_k": top_k or self.settings.top_k,
            "top_p": top_p or self.settings.top_p,
            "repetition_penalty": repetition_penalty or self.settings.repetition_penalty,
        }
        with self._lock:
            if self.settings.mode == "voice_clone":
                outputs = self.model.generate_voice_clone_streaming(
                    ref_audio=str(self.settings.ref_wav),
                    ref_text=self.settings.ref_text,
                    instruct=instruct or self.settings.instruct or None,
                    **shared,
                )
            else:
                outputs = self.model.generate_custom_voice_streaming(
                    speaker=speaker or self.settings.speaker,
                    instruct=instruct or self.settings.instruct or None,
                    **shared,
                )
            pcm = (
                (np.clip(np.asarray(s, dtype=np.float32).reshape(-1), -1.0, 1.0) * 32767)
                .astype("<i2")
                for s, _sr, _timing in outputs
            )
            yield from self._normalise(self._trim_silence(pcm))

    def _normalise(self, blocks: Iterator[bytes]) -> Iterator[bytes]:
        """Scale a whole request towards a fixed loudness.

        The gain is estimated from the voiced audio seen so far, refined as more
        arrives and then locked, so nothing is buffered and the first packet is not
        delayed. Estimating from only the first block was not enough: a soft onset
        mis-scaled the whole request and made consecutive requests differ more, not
        less. Gain changes are ramped within a block so they are not steps.
        """
        target = self.settings.target_rms_dbfs
        if target == 0:
            yield from blocks
            return

        wanted = (10 ** (target / 20)) * 32768
        ceiling = self.settings.peak_ceiling * 32767
        lowest, highest = 1 / self.settings.max_gain, self.settings.max_gain
        lock_after = self.sample_rate            # one second of voiced audio
        energy, counted, gain, locked = 0.0, 0, 1.0, False
        emitted = False

        for block in blocks:
            samples = np.frombuffer(block, dtype="<i2").astype(np.float32)
            previous = gain
            if not locked and samples.size:
                voiced = samples[np.abs(samples) > self.settings.silence_threshold]
                if voiced.size:
                    energy += float((voiced.astype(np.float64) ** 2).sum())
                    counted += voiced.size
                    measured = (energy / counted) ** 0.5
                    gain = min(max(wanted / max(measured, 1.0), lowest), highest)
                if counted >= lock_after:
                    locked = True

            if gain == previous or not emitted:
                # Nothing has been played yet, so there is no level to step from:
                # apply the gain flat rather than swelling into it.
                scaled = samples * gain
            else:
                ramp = np.linspace(previous, gain, samples.size, dtype=np.float32)
                scaled = samples * ramp
            emitted = emitted or samples.size > 0

            peak = float(np.abs(scaled).max()) if scaled.size else 0.0
            if peak > ceiling:
                scaled *= ceiling / peak
            yield scaled.astype("<i2").tobytes()

    def _trim_silence(self, chunks: Iterator[np.ndarray]) -> Iterator[bytes]:
        """Drop the model's leading and trailing silence, keeping pauses inside.

        Only the silence run at the very end of the stream is removed: it is held
        back until the next chunk proves it was an internal pause.
        """
        threshold = self.settings.silence_threshold
        if threshold <= 0:
            for chunk in chunks:
                yield chunk.tobytes()
            return

        preroll = int(self.sample_rate * self.settings.silence_preroll_ms / 1000)
        started = False
        held = np.empty(0, dtype="<i2")
        for chunk in chunks:
            buffer = np.concatenate((held, chunk)) if held.size else chunk
            loud = np.flatnonzero(np.abs(buffer.astype(np.int32)) > threshold)
            if loud.size == 0:
                if started:
                    # A pause inside the utterance: keep all of it, because the
                    # next chunk may prove it was not the trailing silence.
                    held = buffer
                else:
                    # Leading silence: only a pre-roll's worth can ever be needed.
                    held = buffer[-preroll:] if preroll else np.empty(0, dtype="<i2")
                continue
            if not started:
                buffer = buffer[max(0, loud[0] - preroll) :]
                loud = loud - max(0, loud[0] - preroll)
                started = True
            emit, held = buffer[: loud[-1] + 1], buffer[loud[-1] + 1 :]
            if emit.size:
                yield emit.tobytes()
        # Whatever is still held is trailing silence; keep a short tail so the
        # boundary with the next chunk keeps its pause.
        keep = int(self.sample_rate * self.settings.silence_keep_ms / 1000)
        if started and keep and held.size:
            yield held[:keep].tobytes()


def create_app(
    settings: Settings | None = None,
    backend: TTSBackend | None = None,
) -> FastAPI:
    cfg = settings or Settings.from_env()
    tts = backend or QwenTTSBackend(cfg)
    state = {"ready": False, "error": None}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            await asyncio.to_thread(tts.load)
            state["ready"] = True
        except Exception as exc:
            logger.exception("Qwen3-TTS loading failed")
            state["error"] = f"{type(exc).__name__}: {exc}"
        yield

    app = FastAPI(title="SKKU Qwen3-TTS Server", version="0.1.0", lifespan=lifespan)

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
            "model": cfg.model,
            "error": state["error"],
        }

    @app.post("/v1/audio/speech")
    async def synthesize(body: SpeechRequest):
        if not state["ready"]:
            raise HTTPException(status_code=503, detail=state["error"] or "Model is loading")
        if body.response_format != "pcm":
            raise HTTPException(status_code=422, detail="This runtime supports pcm only")

        headers = {
            "X-Audio-Sample-Rate": str(tts.sample_rate),
            "X-Audio-Channels": "1",
            "X-Audio-Sample-Format": "pcm_s16le",
        }
        chunks = tts.stream_pcm(
            body.input, body.hop_len, body.voice, body.language, body.instruct,
            body.temperature, body.top_k, body.top_p, body.repetition_penalty,
        )
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
