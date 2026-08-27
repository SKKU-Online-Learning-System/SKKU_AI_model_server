import pytest

from speech_server.main import _resolve_supported_speaker


def test_resolves_documented_sohee_name_case_insensitively() -> None:
    supported = ["aiden", "ono_anna", "sohee", "vivian"]

    assert _resolve_supported_speaker("Sohee", supported) == "sohee"
    assert _resolve_supported_speaker("SOHEE", supported) == "sohee"
    assert _resolve_supported_speaker("sohee", supported) == "sohee"


def test_rejects_unknown_speaker() -> None:
    with pytest.raises(ValueError, match="Unsupported speaker 'Unknown'"):
        _resolve_supported_speaker("Unknown", ["sohee", "vivian"])


def test_passthrough_when_runtime_does_not_publish_speakers() -> None:
    assert _resolve_supported_speaker("Sohee", []) == "Sohee"
