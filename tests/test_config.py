from speech_server.config import Settings


def test_defaults():
    settings = Settings(_env_file=None)
    assert settings.port == 8010
    assert settings.gpu_id == "5"
    assert settings.tts_language == "Auto"
    assert settings.tts_speaker == "Sohee"
    assert settings.tts_attention_backend == "flash_attention_2"


def test_environment_override(monkeypatch):
    monkeypatch.setenv("SPEECH_PORT", "9000")
    monkeypatch.setenv("TTS_SPEAKER", "Ryan")
    monkeypatch.setenv("TTS_ATTENTION_BACKEND", "sdpa")
    settings = Settings(_env_file=None)
    assert settings.port == 9000
    assert settings.tts_speaker == "Ryan"
    assert settings.tts_attention_backend == "sdpa"
