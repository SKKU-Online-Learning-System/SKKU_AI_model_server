from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from qwen_tts_runtime.main import Settings, create_app


def _silence(n: int) -> np.ndarray:
    return np.zeros(n, dtype="<i2")


def _tone(n: int, amplitude: int = 8000) -> np.ndarray:
    return (np.sin(np.arange(n) / 4.0) * amplitude).astype("<i2")


class FakeQwenTTS:
    sample_rate = 24_000

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def load(self) -> None:
        return None

    def stream_pcm(
        self, text, hop_len=None, speaker=None, language=None, instruct=None,
        temperature=None, top_k=None, top_p=None, repetition_penalty=None,
        continuity_id=None, speed=None,
    ):
        self.calls.append(
            {
                "text": text,
                "hop_len": hop_len,
                "speaker": speaker,
                "language": language,
                "temperature": temperature,
                "continuity_id": continuity_id,
                "speed": speed,
            }
        )
        yield _silence(2400).tobytes()
        yield _tone(2400).tobytes()
        yield _silence(2400).tobytes()


def settings(**overrides) -> Settings:
    return Settings(cache_dir=Path("cache"), **overrides)


def test_streams_pcm_chunks_and_passes_request_options() -> None:
    backend = FakeQwenTTS()
    with TestClient(create_app(settings(), backend)) as client:
        with client.stream(
            "POST",
            "/v1/audio/speech",
            json={
                "input": "안녕하세요",
                "voice": "Sohee",
                "language": "Korean",
                "response_format": "pcm",
                "stream": True,
                "hop_len": 2,
                "temperature": 0.6,
                "continuity_id": "turn-7",
                "speed": 1.25,
            },
        ) as response:
            assert response.status_code == 200
            assert response.headers["X-Audio-Sample-Rate"] == "24000"
            assert response.headers["X-Streaming"] == "true"
            body = b"".join(response.iter_bytes())

    assert len(body) == 3 * 2400 * 2
    assert backend.calls == [
        {
            "text": "안녕하세요",
            "hop_len": 2,
            "speaker": "Sohee",
            "language": "Korean",
            "temperature": 0.6,
            "continuity_id": "turn-7",
            "speed": 1.25,
        }
    ]


def test_non_streaming_response_reports_duration() -> None:
    with TestClient(create_app(settings(), FakeQwenTTS())) as client:
        response = client.post(
            "/v1/audio/speech",
            json={"input": "안녕하세요", "response_format": "pcm", "stream": False},
        )
    assert response.status_code == 200
    assert response.headers["X-Audio-Duration-Ms"] == "300"
    assert "X-Inference-Ms" in response.headers


def test_numeric_speed_changes_duration_without_changing_pitch() -> None:
    from qwen_tts_runtime.main import change_speed

    sample_rate = 24_000
    seconds = np.arange(sample_rate) / sample_rate
    tone = (np.sin(2 * np.pi * 440 * seconds) * 8000).astype("<i2")
    faster = np.frombuffer(
        b"".join(change_speed(iter([tone.tobytes()]), 1.25, sample_rate)), dtype="<i2"
    )

    assert 0.75 * sample_rate < len(faster) < 0.85 * sample_rate
    spectrum = np.abs(np.fft.rfft(faster))
    frequency = np.fft.rfftfreq(len(faster), 1 / sample_rate)[spectrum.argmax()]
    assert abs(frequency - 440) < 10


def test_speed_range_is_validated() -> None:
    with TestClient(create_app(settings(), FakeQwenTTS())) as client:
        response = client.post(
            "/v1/audio/speech",
            json={"input": "안녕하세요", "response_format": "pcm", "speed": 2.1},
        )
    assert response.status_code == 422


def test_unsupported_response_format_is_rejected() -> None:
    with TestClient(create_app(settings(), FakeQwenTTS())) as client:
        response = client.post(
            "/v1/audio/speech",
            json={"input": "안녕하세요", "response_format": "wav"},
        )
    assert response.status_code == 422


def test_silence_trimming_keeps_speech_and_internal_pauses() -> None:
    """The real backend pads ~0.9s of silence onto both ends of every utterance."""
    from qwen_tts_runtime.main import QwenTTSBackend

    backend = QwenTTSBackend(settings())
    frames = [
        _silence(2400),                                  # leading silence
        _tone(2400),                                     # speech
        _silence(2400),                                  # pause inside the utterance
        _tone(2400),                                     # more speech
        _silence(2400),                                  # trailing silence
    ]
    out = b"".join(backend._trim_silence(iter(frames)))
    samples = np.frombuffer(out, dtype="<i2")

    loud = np.flatnonzero(np.abs(samples.astype(np.int32)) > 200)
    preroll = int(24_000 * 20 / 1000)

    # A full frame of leading silence is reduced to at most the pre-roll, and the
    # trailing silence is capped at silence_keep_ms rather than removed, so the
    # seam with the next chunk keeps its pause.
    keep = int(24_000 * 120 / 1000)
    assert loud[0] <= preroll
    assert len(samples) - 1 - loud[-1] <= keep
    # The pause inside the utterance survives: there is a silent run of about one
    # frame between the two bursts of speech.
    quiet_run = np.max(np.diff(loud))
    assert 2000 <= quiet_run <= 2500


def test_normalisation_matches_loudness_across_requests() -> None:
    """Consecutive requests in one turn must not step in level."""
    from qwen_tts_runtime.main import QwenTTSBackend

    backend = QwenTTSBackend(settings())

    def render(amplitude: int) -> np.ndarray:
        # Enough audio for the estimator to settle and lock (1s of voiced signal).
        blocks = [_tone(8000, amplitude).tobytes() for _ in range(5)]
        out = b"".join(backend._normalise(iter(blocks)))
        return np.frombuffer(out, dtype="<i2").astype(np.float32)

    quiet, loud = render(3000), render(12000)
    ratio = np.sqrt((loud**2).mean()) / np.sqrt((quiet**2).mean())
    # A 4x input difference is pulled to within a few percent.
    assert 0.9 <= ratio <= 1.1
    # Once locked the gain is held, so the tail does not drift in level.
    third, last = loud[-16000:-8000], loud[-8000:]
    assert abs(np.sqrt((third**2).mean()) / np.sqrt((last**2).mean()) - 1) < 0.01


def test_normalisation_never_clips() -> None:
    from qwen_tts_runtime.main import QwenTTSBackend

    backend = QwenTTSBackend(settings())
    out = b"".join(backend._normalise(iter([_tone(4800, 400).tobytes()])))
    samples = np.frombuffer(out, dtype="<i2")
    assert np.abs(samples).max() <= int(0.89 * 32767) + 1


def test_normalisation_can_be_disabled() -> None:
    from qwen_tts_runtime.main import QwenTTSBackend

    backend = QwenTTSBackend(settings(target_rms_dbfs=0))
    block = _tone(4800, 3000).tobytes()
    assert b"".join(backend._normalise(iter([block]))) == block


def test_silence_keep_can_be_disabled() -> None:
    from qwen_tts_runtime.main import QwenTTSBackend

    # release_fade_ms too, so this covers the keep behaviour on its own
    backend = QwenTTSBackend(settings(silence_keep_ms=0, release_fade_ms=0))
    frames = [_silence(2400), _tone(2400), _silence(2400)]
    out = np.frombuffer(b"".join(backend._trim_silence(iter(frames))), dtype="<i2")
    loud = np.flatnonzero(np.abs(out.astype(np.int32)) > 200)
    assert len(out) - 1 - loud[-1] < 16


def test_release_fade_ramps_a_request_that_stops_mid_speech() -> None:
    """A request whose audio is still voiced at its last sample must not step to 0.

    The model stops mid-vowel often enough that the previous behaviour -- full
    speech straight into digital silence -- was audible as the sentence being
    chopped rather than ending.
    """
    from qwen_tts_runtime.main import QwenTTSBackend

    cfg = settings(silence_keep_ms=0, release_fade_ms=40)
    backend = QwenTTSBackend(cfg)
    fade = int(backend.sample_rate * cfg.release_fade_ms / 1000)

    # no trailing silence at all: the tone runs to the final sample
    out = np.frombuffer(
        b"".join(backend._trim_silence(iter([_tone(9600, 6000)]))), dtype="<i2"
    )
    assert len(out) == 9600                      # nothing dropped
    assert abs(int(out[-1])) < 200               # ends at rest, not mid-swing
    head = np.abs(out[: 9600 - fade]).max()
    assert head > 5000                           # speech before the fade untouched
    # and the ramp is monotone rather than a step
    tail_env = [np.abs(out[9600 - fade + i : 9600 - fade + i + 240]).max()
                for i in range(0, fade - 240, 240)]
    assert all(a >= b for a, b in zip(tail_env, tail_env[1:]))


def test_release_fade_can_be_disabled() -> None:
    from qwen_tts_runtime.main import QwenTTSBackend

    backend = QwenTTSBackend(settings(silence_keep_ms=0, release_fade_ms=0))
    out = np.frombuffer(
        b"".join(backend._trim_silence(iter([_tone(9600, 6000)]))), dtype="<i2"
    )
    assert np.abs(out[-240:]).max() > 5000


def test_abandoned_stream_hands_the_gpu_lock_back() -> None:
    """Starlette drops the body iterator on disconnect without closing it.

    A generator that held the lock across its yields stayed suspended inside the
    ``with``, so the lock was never released and every later request streamed
    zero audio after the headers.
    """
    import threading
    import time

    from qwen_tts_runtime.main import guarded_stream

    lock = threading.Lock()

    def render():
        for _ in range(64):
            yield b"\x00\x00"

    stream = guarded_stream(render, lock, queue_blocks=2, stall_timeout=0.2)
    assert next(stream) == b"\x00\x00"
    assert lock.locked()

    # Hold the reference the way a cancelled starlette task does, so the iterator
    # is never closed and cannot rely on garbage collection to release the lock.
    abandoned = [stream]

    deadline = time.monotonic() + 5.0
    while lock.locked() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not lock.locked()
    assert abandoned


def test_guarded_stream_yields_every_block_and_reports_render_errors() -> None:
    import threading

    import pytest

    from qwen_tts_runtime.main import guarded_stream

    lock = threading.Lock()
    blocks = [bytes([i]) * 2 for i in range(20)]

    assert list(guarded_stream(lambda: iter(blocks), lock, queue_blocks=2, stall_timeout=5.0)) == blocks
    assert not lock.locked()

    def failing():
        yield b"\x01\x01"
        raise RuntimeError("talker exploded")

    stream = guarded_stream(failing, lock, queue_blocks=4, stall_timeout=5.0)
    assert next(stream) == b"\x01\x01"
    with pytest.raises(RuntimeError, match="talker exploded"):
        list(stream)
    assert not lock.locked()


def _dbfs(pcm: bytes) -> float:
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
    voiced = samples[np.abs(samples) > 200]
    return 20 * np.log10(max(float(np.sqrt((voiced**2).mean())), 1.0) / 32768)


def _blocks(amplitude: int, count: int = 6, n: int = 4800) -> list[bytes]:
    return [_tone(n, amplitude).tobytes() for _ in range(count)]


def test_join_correction_only_removes_the_excess_over_a_natural_boundary() -> None:
    """A request continuing a turn must not open louder than the one it follows.

    The step a sentence boundary carries anyway is kept: the opening is pulled
    down to `join_allowance_db` above the previous tail rather than to the tail
    itself, and the correction is released over the ramp so only the join moves.
    """
    from qwen_tts_runtime.main import QwenTTSBackend

    # levelling off, so this measures the join correction on its own
    cfg = Settings(cache_dir=Path("cache"), target_rms_dbfs=0.0,
                   join_allowance_db=2.0, join_ramp_ms=1000)
    backend = QwenTTSBackend(cfg)
    tenth = int(backend.sample_rate * 0.1) * 2

    quiet = b"".join(backend._normalise(iter(_blocks(3000)), "turn-a"))
    # 6dB louder than the request it follows, of which 4dB is excess
    loud = b"".join(backend._normalise(iter(_blocks(6000)), "turn-a"))

    step = _dbfs(loud[:tenth]) - _dbfs(quiet[-tenth:])
    assert abs(step - cfg.join_allowance_db) < 1.5
    # released afterwards: by the end the request plays at its own level
    assert abs(_dbfs(loud[-tenth:]) - _dbfs(b"".join(_blocks(6000)))) < 0.5


def test_join_correction_is_skipped_without_a_shared_continuity_id() -> None:
    from qwen_tts_runtime.main import QwenTTSBackend

    cfg = Settings(cache_dir=Path("cache"), target_rms_dbfs=0.0, join_allowance_db=2.0)
    backend = QwenTTSBackend(cfg)

    b"".join(backend._normalise(iter(_blocks(3000)), "turn-a"))
    other = b"".join(backend._normalise(iter(_blocks(12000)), "turn-b"))
    standalone = b"".join(backend._normalise(iter(_blocks(12000)), None))

    assert _dbfs(other) == _dbfs(standalone)
