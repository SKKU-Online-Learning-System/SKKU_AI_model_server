"""Isolated streaming Qwen3-TTS runtime.

Serves ``POST /v1/audio/speech`` as the application TTS backend.

Qwen3-TTS uses a 12 Hz speech tokenizer, and ``faster-qwen3-tts`` adds CUDA
graph capture for low-latency streaming.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
import subprocess
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from time import monotonic, perf_counter
from typing import Callable, Generator, Iterator, Protocol

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("qwen_tts_runtime")


def change_speed(chunks: Iterator[bytes], speed: float, sample_rate: int) -> Iterator[bytes]:
    """Change PCM tempo with FFmpeg's pitch-preserving streaming filter."""
    if not 0.5 <= speed <= 2.0:
        raise ValueError("speed must be between 0.5 and 2.0")
    if speed == 1.0:
        yield from chunks
        return

    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        "-filter:a",
        f"atempo={speed}",
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "pipe:1",
    ]
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("QWEN_TTS_SPEED requires ffmpeg") from exc

    failure: list[BaseException] = []

    def feed() -> None:
        try:
            assert process.stdin is not None
            for chunk in chunks:
                process.stdin.write(chunk)
        except BrokenPipeError:
            pass
        except BaseException as exc:
            failure.append(exc)
        finally:
            if process.stdin is not None:
                process.stdin.close()

    feeder = threading.Thread(target=feed, name="qwen-tts-atempo", daemon=True)
    feeder.start()
    completed = False
    try:
        assert process.stdout is not None
        while block := process.stdout.read(8192):
            yield block
        completed = True
    finally:
        if process.stdout is not None:
            process.stdout.close()
        if not completed and process.poll() is None:
            process.terminate()
        feeder.join(timeout=5)
        if feeder.is_alive():
            process.kill()
            feeder.join()
        process.wait()

    if failure:
        raise failure[0]
    if process.returncode:
        error = process.stderr.read().decode(errors="replace").strip() if process.stderr else ""
        raise RuntimeError(f"ffmpeg atempo failed: {error or process.returncode}")


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
    language: str = "Auto"
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
    #
    # This is a ceiling, not a guarantee: only silence the model actually produced
    # can be kept, and measured over 20 requests it produces a median of 35ms, not
    # the 600-950ms the leading pad suggests. `release_fade_ms` covers the rest.
    silence_keep_ms: int = 120
    # A request whose last block is still voiced ends on a step from full speech to
    # digital zero, which is heard as the sentence being chopped rather than ended
    # -- measured, the final 10ms of a request averaged amplitude 2178 out of a
    # 7000 peak. Fading the last samples out turns that step into a release.
    #
    # It costs no latency: the fade is taken from the tail of the stream, so every
    # block including the first is still emitted as soon as it is produced. The
    # request simply ends this much later. 0 disables.
    release_fade_ms: int = 40
    # Each request realises its own loudness, so consecutive requests in one turn
    # differed by up to 1.4x (about 3dB), heard as the voice jumping. Normalising
    # every request towards the same target removes that step. Estimated from the
    # first emitted block so no latency is added; 0 disables.
    target_rms_dbfs: float = -20.0
    # Normalising each request towards the same target equalises their averages,
    # and measured end to end it does: two requests of one turn came out 0.74dB
    # apart. The step that is left is local to the join. A request ends on a
    # sentence-final fall and the next one opens at full utterance-initial level,
    # and because the two were planned independently that step is bigger than the
    # one the model puts at a sentence boundary inside a single request: 2.87dB
    # against 2.03dB over 17 seams and 143 boundaries (Mann-Whitney p=0.03).
    #
    # So only the excess over a natural boundary is corrected, by attenuating the
    # opening of a continuing request and releasing that over `join_ramp_ms`.
    # `join_allowance_db` is the natural step, which must survive: flattening the
    # join to 0dB would be as wrong as leaving it. 0 disables the correction.
    join_allowance_db: float = 2.0
    join_ramp_ms: int = 700
    # Never correct by more than this; a wild measurement must not duck the reply.
    join_max_cut_db: float = 6.0
    # Continuity state is keyed by the caller's turn id; bound it so a long-lived
    # server cannot accumulate one entry per turn.
    continuity_slots: int = 8
    # Bound the correction so a quiet or loud opening cannot swing the whole
    # utterance, and keep headroom against clipping.
    max_gain: float = 2.5
    peak_ceiling: float = 0.89
    # Below the model default, to narrow how far each request's realised loudness
    # can wander. Paired over 10 turns, the level difference between the two
    # requests of one turn fell from 1.60dB to 0.99dB (Wilcoxon p=0.02). It does
    # NOT steady the speaking rate: rate mismatch was unchanged (p=0.63), so do
    # not reach for temperature if tempo is what drifts.
    #
    # An earlier note here warned that lowering temperature makes the talker
    # degenerate on short input. Re-measured for this model in voice_clone mode
    # over 36 runs on fragments of 2 to 32 characters, 0.6/20 produced no silent
    # and no runaway output, so that warning does not hold here. It was measured
    # against the CustomVoice path; re-run that check before assuming it transfers
    # back if `mode` ever changes.
    temperature: float = 0.6
    top_k: int = 20
    top_p: float = 1.0
    repetition_penalty: float = 1.05
    # Free-form style instruction understood by the CustomVoice models.
    instruct: str = ""
    # Exact output tempo multiplier. FFmpeg atempo preserves pitch.
    speed: float = 1.0
    warmup_text: str = "안녕하세요."
    # Fallback for a response that is never closed: how much audio it may buffer,
    # and how long the queue may stay full before the reader counts as gone. Only a
    # client that has stopped reading entirely can reach the timeout on localhost.
    stream_queue_blocks: int = 8
    stream_stall_timeout: float = 5.0
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
            language=os.getenv("QWEN_TTS_LANGUAGE", "Auto"),
            attn_implementation=os.getenv("QWEN_TTS_ATTENTION_BACKEND", "sdpa"),
            chunk_size=int(os.getenv("QWEN_TTS_CHUNK_SIZE", "4") or 4),
            silence_threshold=int(os.getenv("QWEN_TTS_SILENCE_THRESHOLD", "200") or 0),
            silence_preroll_ms=int(os.getenv("QWEN_TTS_SILENCE_PREROLL_MS", "20") or 0),
            silence_keep_ms=int(os.getenv("QWEN_TTS_SILENCE_KEEP_MS", "120") or 0),
            release_fade_ms=int(os.getenv("QWEN_TTS_RELEASE_FADE_MS", "40") or 0),
            target_rms_dbfs=float(os.getenv("QWEN_TTS_TARGET_RMS_DBFS", "-20") or 0),
            join_allowance_db=float(os.getenv("QWEN_TTS_JOIN_ALLOWANCE_DB", "2.0") or 0),
            join_ramp_ms=int(os.getenv("QWEN_TTS_JOIN_RAMP_MS", "700") or 0),
            temperature=float(os.getenv("QWEN_TTS_TEMPERATURE", "0.6") or 0.6),
            top_k=int(os.getenv("QWEN_TTS_TOP_K", "20") or 20),
            top_p=float(os.getenv("QWEN_TTS_TOP_P", "1.0") or 1.0),
            repetition_penalty=float(os.getenv("QWEN_TTS_REPETITION_PENALTY", "1.05") or 1.05),
            instruct=os.getenv("QWEN_TTS_INSTRUCT", ""),
            speed=float(os.getenv("QWEN_TTS_SPEED", "1.0") or 1.0),
            warmup_text=os.getenv("QWEN_TTS_WARMUP_TEXT", "안녕하세요."),
            stream_queue_blocks=int(os.getenv("QWEN_TTS_STREAM_QUEUE_BLOCKS", "8") or 8),
            stream_stall_timeout=float(os.getenv("QWEN_TTS_STREAM_STALL_TIMEOUT", "5") or 5),
            cache_dir=Path(cache) if cache else None,
            api_key=os.getenv("MODEL_SERVER_API_KEY", ""),
        )


class SpeechRequest(BaseModel):
    input: str = Field(min_length=1, max_length=20_000)
    voice: str | None = None
    language: str | None = None
    response_format: str = "pcm"
    instruct: str | None = None
    speed: float | None = Field(default=None, ge=0.5, le=2.0)
    stream: bool = True
    temperature: float | None = Field(default=None, ge=0.05, le=2.0)
    top_k: int | None = Field(default=None, ge=1, le=2048)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    repetition_penalty: float | None = Field(default=None, ge=1.0, le=2.0)
    # Backend-specific streaming granularity; frames per chunk for this runtime.
    hop_len: int | None = Field(default=None, ge=1, le=100)
    # Opaque caller-chosen id shared by every request of one spoken turn. Requests
    # carrying the same id are levelled against each other so the voice does not
    # step at the joins. Omit it and each request is normalised on its own.
    continuity_id: str | None = Field(default=None, max_length=128)


def guarded_stream(
    render: Callable[[], Iterator[bytes]],
    lock: threading.Lock,
    *,
    queue_blocks: int,
    stall_timeout: float,
) -> Iterator[bytes]:
    """Run ``render`` under ``lock`` on a private thread and yield its blocks.

    The GPU lock must not be held across a ``yield`` of the iterator handed to
    ``StreamingResponse``. Starlette cancels ``stream_response`` when the client
    disconnects and never closes the body iterator, so such a generator is left
    suspended inside its ``with`` and the lock is never released: every later
    request then blocks after the response headers and streams zero audio. The
    agent abandons TTS streams on every barge-in, so this is the normal case, not
    an edge case. Owning the lock on a thread the response cannot strand keeps its
    release independent of whether the iterator is ever finalised.
    """
    blocks: Queue = Queue(maxsize=queue_blocks)
    stop = threading.Event()
    done = threading.Event()
    failure: list[Exception] = []

    def produce() -> None:
        try:
            with lock:
                for block in render():
                    deadline = monotonic() + stall_timeout
                    while not stop.is_set():
                        try:
                            blocks.put(block, timeout=0.05)
                            break
                        except Full:
                            if monotonic() >= deadline:
                                # Nothing has read the response for a whole stall
                                # window: the client is gone, stop synthesising.
                                stop.set()
                    if stop.is_set():
                        break
        except Exception as exc:  # re-raised on the consumer side
            failure.append(exc)
        finally:
            # The lock is already released, so a consumer that never comes back
            # cannot strand it.
            done.set()

    threading.Thread(target=produce, name="tts-render", daemon=True).start()
    try:
        while True:
            try:
                block = blocks.get(timeout=0.05)
            except Empty:
                if done.is_set():
                    break
                continue
            yield block
        if failure:
            raise failure[0]
    finally:
        stop.set()


class ClosingStreamingResponse(StreamingResponse):
    """A ``StreamingResponse`` that always closes the iterator it was handed.

    Starlette cancels ``stream_response`` on client disconnect and leaves the body
    iterator suspended, so ``guarded_stream``'s consumer never reaches its cleanup
    and the render thread only notices once its queue has been full for a whole
    stall window. Closing the source here demotes that timeout to the safety net it
    should be: a barge-in frees the GPU on the next block instead of seconds later.
    """

    def __init__(self, content: Generator[bytes, None, None], **kwargs) -> None:
        self._source = content
        super().__init__(content, **kwargs)

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._source.close()


class TTSBackend(Protocol):
    sample_rate: int

    def load(self) -> None: ...

    def stream_pcm(
        self, text: str, hop_len: int | None = None, speaker: str | None = None,
        language: str | None = None, instruct: str | None = None,
        temperature: float | None = None, top_k: int | None = None,
        top_p: float | None = None, repetition_penalty: float | None = None,
        continuity_id: str | None = None,
        speed: float | None = None,
    ) -> Iterator[bytes]: ...


class QwenTTSBackend:
    sample_rate = 24_000

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None
        # One GPU, one autoregressive decode loop with a captured CUDA graph.
        self._lock = threading.Lock()
        # turn id -> dBFS of the tail of that turn's previous request.
        self._continuity: dict[str, float] = {}

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
        continuity_id: str | None = None,
        speed: float | None = None,
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

        def render() -> Iterator[bytes]:
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
            rendered = self._normalise(self._trim_silence(pcm), continuity_id)
            effective_speed = speed if speed is not None else self.settings.speed
            return change_speed(rendered, effective_speed, self.sample_rate)

        return guarded_stream(
            render,
            self._lock,
            queue_blocks=self.settings.stream_queue_blocks,
            stall_timeout=self.settings.stream_stall_timeout,
        )

    def _remember_tail(self, continuity_id: str | None, tail_db: float | None) -> None:
        if not continuity_id or tail_db is None:
            return
        self._continuity.pop(continuity_id, None)
        self._continuity[continuity_id] = tail_db
        while len(self._continuity) > self.settings.continuity_slots:
            self._continuity.pop(next(iter(self._continuity)))

    @staticmethod
    def _voiced_dbfs(samples: np.ndarray, threshold: int) -> float | None:
        voiced = samples[np.abs(samples) > threshold]
        if voiced.size < 256:
            return None
        return 20 * np.log10(max(float(np.sqrt((voiced.astype(np.float64) ** 2).mean())), 1.0) / 32768)

    def _normalise(
        self, blocks: Iterator[bytes], continuity_id: str | None = None
    ) -> Iterator[bytes]:
        """Scale a whole request towards a fixed loudness.

        The gain is estimated from the voiced audio seen so far, refined as more
        arrives and then locked, so nothing is buffered and the first packet is not
        delayed. Estimating from only the first block was not enough: a soft onset
        mis-scaled the whole request and made consecutive requests differ more, not
        less. Gain changes are ramped within a block so they are not steps.
        """
        # Loudness targeting and the join correction are independent: turning the
        # first off with target_rms_dbfs=0 must not silently take the second with
        # it, so the loop still runs and only the gain estimation is skipped.
        target = self.settings.target_rms_dbfs
        levelling = target != 0
        wanted = (10 ** (target / 20)) * 32768
        ceiling = self.settings.peak_ceiling * 32767
        lowest, highest = 1 / self.settings.max_gain, self.settings.max_gain
        lock_after = self.sample_rate            # one second of voiced audio
        energy, counted, gain, locked = 0.0, 0, 1.0, False
        emitted = False
        # Join correction against the tail of this turn's previous request. It is
        # decided from the first voiced block and released over `join_ramp_ms`;
        # nothing is buffered, so a continuing request is no slower than any other.
        tail_db = self._continuity.get(continuity_id) if continuity_id else None
        ramp_total = int(self.sample_rate * self.settings.join_ramp_ms / 1000)
        ramp_done = ramp_total if tail_db is None or self.settings.join_allowance_db <= 0 else 0
        cut = 1.0
        tail_window = np.empty(0, dtype=np.float32)
        keep_tail = int(self.sample_rate * 0.4)

        for block in blocks:
            samples = np.frombuffer(block, dtype="<i2").astype(np.float32)
            previous = gain
            if levelling and not locked and samples.size:
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

            if ramp_done < ramp_total and scaled.size:
                if ramp_done == 0:
                    head_db = self._voiced_dbfs(scaled, self.settings.silence_threshold)
                    if head_db is None:
                        # Silence so far; the join has not started yet.
                        yield scaled.astype("<i2").tobytes()
                        continue
                    excess = head_db - tail_db - self.settings.join_allowance_db
                    excess = min(max(excess, 0.0), self.settings.join_max_cut_db)
                    cut = 10 ** (-excess / 20)
                start = ramp_done
                ramp_done = min(ramp_total, start + scaled.size)
                positions = np.arange(start, start + scaled.size, dtype=np.float32)
                # cut at the join, unity by the end of the ramp
                envelope = cut + (1.0 - cut) * np.clip(positions / max(ramp_total, 1), 0.0, 1.0)
                scaled = scaled * envelope

            peak = float(np.abs(scaled).max()) if scaled.size else 0.0
            if peak > ceiling:
                scaled *= ceiling / peak
            if scaled.size:
                tail_window = np.concatenate((tail_window, scaled))[-keep_tail:]
            yield scaled.astype("<i2").tobytes()

        self._remember_tail(
            continuity_id, self._voiced_dbfs(tail_window, self.settings.silence_threshold)
        )

    def _trim_silence(self, chunks: Iterator[np.ndarray]) -> Iterator[bytes]:
        """Drop the model's leading and trailing silence, keeping pauses inside.

        Only the silence run at the very end of the stream is removed: it is held
        back until the next chunk proves it was an internal pause. The last
        `release_fade_ms` of speech is held back too, and faded out once the stream
        turns out to have ended there, so a request that stops mid-vowel releases
        instead of stepping to zero.
        """
        threshold = self.settings.silence_threshold
        if threshold <= 0:
            for chunk in chunks:
                yield chunk.tobytes()
            return

        preroll = int(self.sample_rate * self.settings.silence_preroll_ms / 1000)
        fade = int(self.sample_rate * self.settings.release_fade_ms / 1000)
        # The tail of the emitted audio, withheld so it can still be faded. Only
        # the end of the stream is delayed by it; every block is still emitted as
        # soon as the block after it arrives, and the first one immediately.
        pending = np.empty(0, dtype="<i2")
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
                if pending.size:
                    emit = np.concatenate((pending, emit))
                pending = emit[-fade:] if fade else np.empty(0, dtype="<i2")
                emit = emit[: len(emit) - len(pending)]
                if emit.size:
                    yield emit.tobytes()
        if not started:
            return
        # The stream ended here, so the withheld tail is a release rather than more
        # speech: ramp it to zero. When the model already decayed on its own this is
        # inaudible; when it stopped mid-vowel it is what keeps that from clicking.
        release = []
        if pending.size:
            ramp = np.linspace(1.0, 0.0, pending.size, dtype=np.float32)
            release.append((pending.astype(np.float32) * ramp).astype("<i2"))
        # Whatever is still held is trailing silence; keep a short tail so the
        # boundary with the next chunk keeps its pause.
        keep = int(self.sample_rate * self.settings.silence_keep_ms / 1000)
        if keep and held.size:
            release.append(held[:keep])
        if release:
            yield np.concatenate(release).tobytes()


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
            body.continuity_id, body.speed,
        )
        if body.stream:
            return ClosingStreamingResponse(
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
