import io
import wave

import numpy as np

from speech_server.audio import ASR_SAMPLE_RATE, decode_upload, encode_pcm16, encode_wav_pcm16


def make_wav(sample_rate=8000, channels=2):
    samples = (np.zeros((sample_rate // 10, channels)) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(samples.tobytes())
    return buf.getvalue()


def test_wav_is_resampled_and_mixed_to_mono():
    audio = decode_upload(make_wav(), "test.wav", "audio/wav")
    assert audio.sample_rate == ASR_SAMPLE_RATE
    assert audio.samples.ndim == 1
    assert 1500 <= len(audio.samples) <= 1700


def test_pcm_and_wav_outputs_are_pcm16_24k():
    samples = np.zeros(2400, dtype=np.float32)
    pcm = encode_pcm16(samples, 24000)
    assert len(pcm) == 4800
    wav = encode_wav_pcm16(samples, 24000)
    with wave.open(io.BytesIO(wav), "rb") as f:
        assert f.getframerate() == 24000
        assert f.getnchannels() == 1
        assert f.getsampwidth() == 2
