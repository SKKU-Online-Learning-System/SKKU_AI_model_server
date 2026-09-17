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
    ):
        self.calls.append(
            {
                "text": text,
                "hop_len": hop_len,
                "speaker": speaker,
                "language": language,
                "temperature": temperature,
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

    backend = QwenTTSBackend(settings(silence_keep_ms=0))
    frames = [_silence(2400), _tone(2400), _silence(2400)]
    out = np.frombuffer(b"".join(backend._trim_silence(iter(frames))), dtype="<i2")
    loud = np.flatnonzero(np.abs(out.astype(np.int32)) > 200)
    assert len(out) - 1 - loud[-1] < 16
