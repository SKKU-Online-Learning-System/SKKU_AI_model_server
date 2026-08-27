from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx


def auth_headers() -> dict[str, str]:
    key = os.getenv("MODEL_SERVER_API_KEY", "")
    return {"Authorization": f"Bearer {key}"} if key else {}


def gpu_snapshot() -> list[dict[str, str]]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception as exc:
        return [{"error": str(exc)}]
    rows = []
    for line in output.strip().splitlines():
        idx, name, used, total, util = [item.strip() for item in line.split(",", 4)]
        rows.append(
            {
                "index": idx,
                "name": name,
                "memory_used_mib": used,
                "memory_total_mib": total,
                "utilization_gpu_pct": util,
            }
        )
    return rows


def benchmark_llm(base_url: str, model: str, prompt: str) -> dict[str, float | int | None]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    start = time.perf_counter()
    first_token_at = None
    usage = None
    with httpx.Client(timeout=180, headers=auth_headers()) as client:
        with client.stream("POST", f"{base_url}/chat/completions", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                if first_token_at is None:
                    choices = chunk.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        if delta.get("content") or delta.get("reasoning"):
                            first_token_at = time.perf_counter()
                if chunk.get("usage"):
                    usage = chunk["usage"]
    end = time.perf_counter()
    ttft_ms = (first_token_at - start) * 1000 if first_token_at else None
    total_s = end - start
    output_tokens = int((usage or {}).get("completion_tokens", 0))
    generation_s = max(end - (first_token_at or start), 1e-9)
    return {
        "ttft_ms": round(ttft_ms, 2) if ttft_ms is not None else None,
        "tokens_per_sec": round(output_tokens / generation_s, 2) if output_tokens else None,
        "total_latency_ms": round(total_s * 1000, 2),
        "output_tokens": output_tokens,
    }


def benchmark_asr(speech_base: str, wav_path: Path) -> dict[str, float | int]:
    start = time.perf_counter()
    with httpx.Client(timeout=180, headers=auth_headers()) as client, wav_path.open("rb") as f:
        response = client.post(
            f"{speech_base}/v1/audio/transcriptions",
            files={"file": (wav_path.name, f, "audio/wav")},
            data={"language": "Korean"},
        )
        response.raise_for_status()
    total_ms = (time.perf_counter() - start) * 1000
    result = response.json()
    duration_ms = max(int(result["audio_duration_ms"]), 1)
    inference_ms = int(result["inference_ms"])
    return {
        "http_latency_ms": round(total_ms, 2),
        "inference_ms": inference_ms,
        "audio_duration_ms": duration_ms,
        "rtf": round(inference_ms / duration_ms, 4),
    }


def benchmark_tts(speech_base: str) -> dict[str, float | int]:
    start = time.perf_counter()
    with httpx.Client(timeout=180, headers=auth_headers()) as client:
        response = client.post(
            f"{speech_base}/v1/audio/speech",
            json={
                "input": "운영체제에서 가상 메모리는 물리 메모리를 보완하는 주소 공간 추상화입니다.",
                "voice": "Sohee",
                "language": "Korean",
                "response_format": "pcm",
            },
        )
        response.raise_for_status()
    total_ms = (time.perf_counter() - start) * 1000
    generation_ms = int(response.headers["x-inference-ms"])
    audio_duration_ms = max(int(response.headers["x-audio-duration-ms"]), 1)
    return {
        "http_latency_ms": round(total_ms, 2),
        "generation_ms": generation_ms,
        "audio_duration_ms": audio_duration_ms,
        "rtf": round(generation_ms / audio_duration_ms, 4),
    }


def summarize(items: list[dict]) -> dict:
    keys = items[0].keys()
    out: dict[str, object] = {"runs": items}
    for key in keys:
        values = [item[key] for item in items if isinstance(item.get(key), (int, float)) and item[key] is not None]
        if values:
            out[f"{key}_mean"] = round(statistics.mean(values), 2)
    return out


def run_concurrent(fn, concurrency: int) -> list[dict]:
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(lambda _: fn(), range(concurrency)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--concurrency", type=int, choices=[1, 2, 4], default=1)
    parser.add_argument("--asr-wav", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    text_port = int(os.getenv("TEXT_PORT", "8001"))
    voice_port = int(os.getenv("VOICE_PORT", "8002"))
    speech_port = int(os.getenv("SPEECH_PORT", "8010"))
    text_model = os.getenv("TEXT_MODEL", "Qwen/Qwen3.8-27B")
    voice_model = os.getenv("VOICE_MODEL", "Qwen/Qwen3.5-9B")
    text_base = f"http://{args.host}:{text_port}/v1"
    voice_base = f"http://{args.host}:{voice_port}/v1"
    speech_base = f"http://{args.host}:{speech_port}"

    report: dict[str, object] = {
        "concurrency": args.concurrency,
        "gpu_before": gpu_snapshot(),
    }
    text_fn = lambda: benchmark_llm(
        text_base,
        text_model,
        "운영체제에서 가상 메모리의 목적을 두 문장으로 설명해줘.",
    )
    voice_fn = lambda: benchmark_llm(
        voice_base,
        voice_model,
        "학생에게 프로세스와 스레드의 차이를 짧고 자연스럽게 설명해줘.",
    )
    report["text_llm"] = summarize(run_concurrent(text_fn, args.concurrency))
    report["voice_llm"] = summarize(run_concurrent(voice_fn, args.concurrency))
    report["tts"] = benchmark_tts(speech_base)
    if args.asr_wav:
        report["asr"] = benchmark_asr(speech_base, args.asr_wav)
    else:
        report["asr"] = {"skipped": "Pass --asr-wav PATH to measure ASR RTF."}
    report["gpu_after"] = gpu_snapshot()

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
