from speech_server.config import Settings


def test_defaults():
    settings = Settings(_env_file=None)
    assert settings.port == 8010
    assert settings.gpu_id == "5"
    assert settings.tts_language == "Korean"
    assert settings.tts_speaker == "Sohee"


def test_environment_override(monkeypatch):
    monkeypatch.setenv("SPEECH_PORT", "9000")
    monkeypatch.setenv("TTS_SPEAKER", "Ryan")
    settings = Settings(_env_file=None)
    assert settings.port == 9000
    assert settings.tts_speaker == "Ryan"
