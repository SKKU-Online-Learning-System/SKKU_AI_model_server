from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from cosyvoice_runtime.main import Settings, create_app


class FakeCosyVoice:
    sample_rate = 24_000

    def load(self) -> None:
        return None

    def stream_pcm(self, text: str, hop_len: int | None = None):
        assert text == "안녕하세요"
        self.hop_len = hop_len
        yield b"\x00\x00" * 240
        yield b"\x01\x00" * 240


def test_streams_pcm_chunks() -> None:
    settings = Settings(Path("model"), Path("voice.wav"), "transcript")
    with TestClient(create_app(settings, FakeCosyVoice())) as client:
        with client.stream(
            "POST",
            "/v1/audio/speech",
            json={"input": "안녕하세요", "voice": "cosyvoice", "stream": True},
        ) as response:
            assert response.status_code == 200
            assert response.headers["x-streaming"] == "true"
            assert response.headers["x-audio-sample-rate"] == "24000"
            assert b"".join(response.iter_bytes()) == b"\x00\x00" * 240 + b"\x01\x00" * 240


def test_rejects_unknown_voice() -> None:
    settings = Settings(Path("model"), Path("voice.wav"), "transcript")
    with TestClient(create_app(settings, FakeCosyVoice())) as client:
        response = client.post(
            "/v1/audio/speech",
            json={"input": "안녕하세요", "voice": "Sohee"},
        )
        assert response.status_code == 422
