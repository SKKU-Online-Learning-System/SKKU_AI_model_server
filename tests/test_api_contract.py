from __future__ import annotations

import io
import wave

from fastapi.testclient import TestClient

from speech_server.config import Settings
from speech_server.main import create_app
from speech_server.runtime import InferenceRuntime

from tests.fakes import FakeASR, FakeTTS


def pcm_wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(b"\x00\x00" * 1600)
    return buf.getvalue()


def test_asr_response_schema(client):
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.wav", pcm_wav(), "audio/wav")},
        data={"language": "Korean"},
    )
    assert response.status_code == 200
    assert response.json() == {
        "text": "테스트 음성입니다.",
        "language": "Korean",
        "audio_duration_ms": 100,
        "inference_ms": 10,
    }


def test_malformed_audio(client):
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("bad.wav", b"not a wav", "audio/wav")},
    )
    assert response.status_code == 400


def test_tts_pcm_contract(client):
    response = client.post(
        "/v1/audio/speech",
        json={"input": "안녕하세요", "voice": "Sohee", "language": "Korean", "response_format": "pcm"},
    )
    assert response.status_code == 200
    assert response.headers["x-audio-sample-rate"] == "24000"
    assert response.headers["x-audio-channels"] == "1"
    assert response.headers["x-audio-sample-format"] == "pcm_s16le"
    assert len(response.content) == 48000


def test_invalid_speaker(client):
    response = client.post(
        "/v1/audio/speech",
        json={"input": "안녕하세요", "voice": "Unknown", "response_format": "wav"},
    )
    assert response.status_code == 422


def test_invalid_output_format(client):
    response = client.post(
        "/v1/audio/speech",
        json={"input": "안녕하세요", "response_format": "mp3"},
    )
    assert response.status_code == 422


def test_optional_authentication():
    settings = Settings(api_key="secret", verbose=False)
    runtime = InferenceRuntime(settings, asr_service=FakeASR(), tts_service=FakeTTS())
    app = create_app(settings, runtime)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 401
        assert client.get("/health", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert client.get("/health", headers={"Authorization": "Bearer secret"}).status_code == 200
