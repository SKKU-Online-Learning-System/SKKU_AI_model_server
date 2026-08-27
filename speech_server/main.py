from __future__ import annotations

import asyncio
import hmac
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response

from .audio import (
    TTS_OUTPUT_SAMPLE_RATE,
    AudioValidationError,
    decode_upload,
    encode_pcm16,
    encode_wav_pcm16,
)
from .config import Settings, get_settings
from .runtime import InferenceRuntime
from .schemas import HealthResponse, SpeechRequest, TranscriptionResponse

logger = logging.getLogger("speech_server")


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _resolve_supported_speaker(requested: str, supported: list[str]) -> str:
    """Resolve a requested Qwen3-TTS speaker name case-insensitively.

    Qwen3-TTS exposes get_supported_speakers() as lower-cased names while its
    generation API accepts speaker names case-insensitively. Keep the model
    server contract aligned with the upstream behavior so documented names such
    as ``Sohee`` are accepted even when the runtime reports ``sohee``.
    """
    if not supported:
        return requested
    by_casefold = {str(name).casefold(): str(name) for name in supported}
    resolved = by_casefold.get(requested.casefold())
    if resolved is None:
        raise ValueError(
            f"Unsupported speaker '{requested}'. Supported: {', '.join(supported)}"
        )
    return resolved


def create_app(settings: Settings | None = None, runtime: InferenceRuntime | None = None) -> FastAPI:
    cfg = settings or get_settings()
    _configure_logging(cfg.verbose)
    rt = runtime or InferenceRuntime(cfg)
    inference_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime = rt
        app.state.load_task = asyncio.create_task(asyncio.to_thread(rt.load))
        yield
        load_task = app.state.load_task
        if not load_task.done():
            await load_task

    app = FastAPI(title="SKKU Speech Model Server", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def optional_bearer_auth(request: Request, call_next):
        if cfg.api_key:
            expected = f"Bearer {cfg.api_key}"
            actual = request.headers.get("authorization", "")
            if not hmac.compare_digest(actual, expected):
                return Response(status_code=401, content='{"detail":"Unauthorized"}', media_type="application/json")
        return await call_next(request)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        status = rt.status
        state = "ok" if status.ready else ("loading" if status.loading else "error")
        return HealthResponse(
            status=state,
            ready=status.ready,
            gpu=status.gpu,
            asr_model=cfg.asr_model,
            tts_model=cfg.tts_model,
            error=status.error,
        )

    def require_ready() -> None:
        if not rt.status.ready:
            raise HTTPException(status_code=503, detail=rt.status.error or "Models are still loading.")

    @app.post("/v1/audio/transcriptions", response_model=TranscriptionResponse)
    async def transcribe(
        file: UploadFile = File(...),
        language: str | None = Form(default=None),
    ) -> TranscriptionResponse:
        require_ready()
        data = await file.read(cfg.max_audio_bytes + 1)
        if len(data) > cfg.max_audio_bytes:
            raise HTTPException(status_code=413, detail="Audio upload is too large.")
        try:
            audio = decode_upload(data, filename=file.filename, content_type=file.content_type)
        except AudioValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        async with inference_lock:
            result = await asyncio.to_thread(
                rt.asr.transcribe, audio.samples, audio.sample_rate, language
            )

        rtf = result.inference_ms / max(audio.duration_ms, 1)
        logger.info(
            "asr audio_duration_ms=%d inference_ms=%d rtf=%.4f",
            audio.duration_ms,
            result.inference_ms,
            rtf,
        )
        return TranscriptionResponse(
            text=result.text,
            language=result.language,
            audio_duration_ms=audio.duration_ms,
            inference_ms=result.inference_ms,
        )

    @app.post("/v1/audio/speech")
    async def synthesize(body: SpeechRequest) -> Response:
        require_ready()
        requested_speaker = body.voice or cfg.tts_speaker
        language = body.language or cfg.tts_language
        supported = rt.tts.supported_speakers()
        try:
            speaker = _resolve_supported_speaker(requested_speaker, supported)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        async with inference_lock:
            result = await asyncio.to_thread(
                rt.tts.synthesize,
                body.input,
                speaker,
                language,
                body.instruct,
            )

        if body.response_format == "pcm":
            payload = encode_pcm16(result.samples, result.sample_rate, TTS_OUTPUT_SAMPLE_RATE)
            media_type = "audio/L16;rate=24000;channels=1"
        else:
            payload = encode_wav_pcm16(result.samples, result.sample_rate, TTS_OUTPUT_SAMPLE_RATE)
            media_type = "audio/wav"

        output_duration_ms = round((len(payload) / 2) * 1000 / TTS_OUTPUT_SAMPLE_RATE)
        if body.response_format == "wav":
            output_duration_ms = round(len(result.samples) * 1000 / result.sample_rate)
        rtf = result.inference_ms / max(output_duration_ms, 1)
        logger.info(
            "tts text_chars=%d generation_ms=%d audio_duration_ms=%d rtf=%.4f speaker=%s language=%s",
            len(body.input),
            result.inference_ms,
            output_duration_ms,
            rtf,
            speaker,
            language,
        )
        return Response(
            content=payload,
            media_type=media_type,
            headers={
                "X-Inference-Ms": str(result.inference_ms),
                "X-Audio-Duration-Ms": str(output_duration_ms),
                "X-Audio-Sample-Rate": str(TTS_OUTPUT_SAMPLE_RATE),
                "X-Audio-Channels": "1",
                "X-Audio-Sample-Format": "pcm_s16le",
            },
        )

    return app


app = create_app()
