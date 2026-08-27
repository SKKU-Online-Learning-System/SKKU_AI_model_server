from __future__ import annotations

import argparse
import json
import os
import tempfile
import wave
from pathlib import Path

import httpx


def headers() -> dict[str, str]:
    key = os.getenv("MODEL_SERVER_API_KEY", "")
    return {"Authorization": f"Bearer {key}"} if key else {}


def raise_for_status_with_body(response: httpx.Response) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        body = response.text[:2000]
        raise RuntimeError(
            f"{response.request.method} {response.request.url} returned "
            f"HTTP {response.status_code}: {body}"
        ) from exc


def check_json(response: httpx.Response) -> dict:
    raise_for_status_with_body(response)
    return response.json()


def chat(base_url: str, model: str, messages: list[dict], **extra) -> dict:
    payload = {"model": model, "messages": messages, **extra}
    with httpx.Client(timeout=180, headers=headers()) as client:
        return check_json(client.post(f"{base_url}/chat/completions", json=payload))


def check_streaming(base_url: str, model: str) -> None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "가상 메모리를 한 문장으로 설명해줘."}],
        "stream": True,
        "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    got_chunk = False
    with httpx.Client(timeout=180, headers=headers()) as client:
        with client.stream("POST", f"{base_url}/chat/completions", json=payload) as response:
            if response.is_error:
                response.read()
                raise_for_status_with_body(response)
            for line in response.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    got_chunk = True
                    break
    assert got_chunk, "No streaming chunk received"


def check_tool_call(base_url: str, model: str) -> None:
    tools = [{
        "type": "function",
        "function": {
            "name": "get_course_room",
            "description": "강의실을 조회한다.",
            "parameters": {
                "type": "object",
                "properties": {"course": {"type": "string"}},
                "required": ["course"],
            },
        },
    }]
    result = chat(
        base_url,
        model,
        [{"role": "user", "content": "운영체제 강의실을 get_course_room 함수로 조회해줘."}],
        tools=tools,
        tool_choice="auto",
        max_tokens=128,
        chat_template_kwargs={"enable_thinking": False},
    )
    calls = result["choices"][0]["message"].get("tool_calls")
    assert calls, f"Expected tool call, got: {json.dumps(result, ensure_ascii=False)[:1000]}"


def save_pcm_response_as_wav(path: Path, pcm: bytes, sample_rate: int = 24000) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--asr-wav", default=os.getenv("ASR_SAMPLE_WAV", ""))
    args = parser.parse_args()

    text_port = int(os.getenv("TEXT_PORT", "8001"))
    voice_port = int(os.getenv("VOICE_PORT", "8002"))
    speech_port = int(os.getenv("SPEECH_PORT", "8010"))
    text_model = os.getenv("TEXT_MODEL", "Qwen/Qwen3.8-27B")
    voice_model = os.getenv("VOICE_MODEL", "Qwen/Qwen3.5-9B")
    text_base = f"http://{args.host}:{text_port}/v1"
    voice_base = f"http://{args.host}:{voice_port}/v1"
    speech_base = f"http://{args.host}:{speech_port}"

    print("[1/7] Text LLM")
    result = chat(
        text_base,
        text_model,
        [{"role": "user", "content": "운영체제에서 가상 메모리가 무엇인지 한 문장으로 설명해줘."}],
        max_tokens=96,
        chat_template_kwargs={"enable_thinking": False},
    )
    assert result["choices"][0]["message"].get("content")

    print("[2/7] Text streaming")
    check_streaming(text_base, text_model)

    print("[3/7] Text tool calling")
    check_tool_call(text_base, text_model)

    print("[4/7] Voice LLM low-latency streaming")
    check_streaming(voice_base, voice_model)

    print("[5/7] TTS Korean / Sohee")
    with httpx.Client(timeout=180, headers=headers()) as client:
        tts = client.post(
            f"{speech_base}/v1/audio/speech",
            json={
                "input": "안녕하세요. 강의 내용을 같이 공부해 볼까요?",
                "voice": "Sohee",
                "language": "Korean",
                "response_format": "pcm",
            },
        )
        raise_for_status_with_body(tts)
        assert tts.headers.get("x-audio-sample-rate") == "24000"
        assert tts.headers.get("x-audio-channels") == "1"
        assert tts.headers.get("x-audio-sample-format") == "pcm_s16le"
        assert len(tts.content) > 0 and len(tts.content) % 2 == 0

        print("[6/7] ASR Korean")
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(args.asr_wav) if args.asr_wav else Path(tmpdir) / "tts_korean.wav"
            if not args.asr_wav:
                save_pcm_response_as_wav(wav_path, tts.content)
            with wav_path.open("rb") as audio_file:
                asr = client.post(
                    f"{speech_base}/v1/audio/transcriptions",
                    files={"file": (wav_path.name, audio_file, "audio/wav")},
                    data={"language": "Korean"},
                )
            raise_for_status_with_body(asr)
            asr_json = asr.json()
            assert isinstance(asr_json.get("text"), str) and asr_json["text"].strip()
            print("    transcript:", asr_json["text"])

    print("[7/7] Speech health")
    with httpx.Client(timeout=30, headers=headers()) as client:
        health = check_json(client.get(f"{speech_base}/health"))
        assert health["ready"] is True

    print("All smoke tests passed.")


if __name__ == "__main__":
    main()
