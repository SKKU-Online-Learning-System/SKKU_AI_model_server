from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from speech_server.config import Settings
from speech_server.main import create_app
from speech_server.runtime import InferenceRuntime
from tests.fakes import FakeASR, FakeTTS


@pytest.fixture
def settings() -> Settings:
    return Settings(api_key="", verbose=False)


@pytest.fixture
def client(settings: Settings):
    runtime = InferenceRuntime(settings, asr_service=FakeASR(), tts_service=FakeTTS())
    app = create_app(settings, runtime)
    with TestClient(app) as test_client:
        for _ in range(100):
            if runtime.status.ready:
                break
            time.sleep(0.01)
        assert runtime.status.ready
        yield test_client
