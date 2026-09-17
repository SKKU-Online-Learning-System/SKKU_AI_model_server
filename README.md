# SKKU AI Model Server

Independent, stateless GPU inference server for **SKKU Course Agent**.

This repository serves models only. Application logic remains in `SKKU_AI_agent`.

## Architecture

| Service | Model | GPU | Runtime | Port |
|---|---|---:|---|---:|
| Text LLM | `Qwen/Qwen3.8-27B` (vision on) | 0,1,2,3 | vLLM 0.28.0, BF16, TP=4 | 8001 |
| Voice LLM | `Qwen/Qwen3.5-9B` | 4 | vLLM 0.28.0, BF16, TP=1 | 8002 |
| ASR | `Qwen/Qwen3-ASR-0.6B` | 5 | official `qwen-asr` | 8004 |
| TTS | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | 5 | `faster-qwen3-tts` (streaming) | 8012 |
| Embedding | `Qwen/Qwen3-VL-Embedding-2B` | 5 | vLLM 0.28.0, BF16, pooling | 8003 |

The TTS service is the **streaming Qwen3-TTS runtime on port 8012**. The earlier
in-process, non-streaming `qwen-tts` on the Speech port remains off by default
(`SPEECH_TTS_ENABLED=false`); loading it only costs GPU 5 memory.

This branch is configured for the Backend.AI session whose preopen ports are
`8000`, `8001`, `8002` and `8004`: the Speech API listens on `8004` instead of
`8010`, while TTS (`8012`) and Embedding (`8003`) are reached over localhost by
the application running in the same session. See
[Backend.AI Preopen Ports](#backendai-preopen-ports) for which ports actually
have to be opened.

Expected application URLs:

```text
TEXT_LLM_BASE_URL=http://<MODEL_SERVER>:8001/v1
VOICE_LLM_BASE_URL=http://<MODEL_SERVER>:8002/v1
SPEECH_BASE_URL=http://<MODEL_SERVER>:8004
TTS_BASE_URL=http://<MODEL_SERVER>:8012
EMBEDDING_BASE_URL=http://<MODEL_SERVER>:8003/v1
```

## Repository boundary

This repository does **not** implement RAG, databases, pgvector, course logic, tool execution, web search,
visualization, VAD, turn-taking, barge-in, WebSocket session orchestration, frontend logic, or chat history.

## Runtime: uv only

The school development session is already a GPU-enabled container. The model server therefore runs directly
with **uv**. There is no Docker Compose runtime and no Docker-in-Docker requirement.

```text
school GPU container
├── GPU 0,1,2,3  Text LLM    :8001
├── GPU 4        Voice LLM   :8002
├── GPU 5        Speech      :8004  (ASR; Backend.AI preopen port)
├── GPU 5        Qwen3-TTS   :8012  (streaming TTS, localhost only)
└── GPU 5        Embedding   :8003  (localhost only)
```

Lifecycle scripts:

```text
scripts/start_all.sh
scripts/stop_all.sh
scripts/healthcheck.sh
scripts/gpu_status.sh
scripts/start_qwen_tts.sh
scripts/make_tts_reference.sh
```

Runtime state:

```text
.run/text-llm.pid
.run/voice-llm.pid
.run/speech.pid

logs/text-llm.log
logs/voice-llm.log
logs/speech.log
```

`.run/` and `logs/` are ignored by Git.

## School GPU environment

Target:

- NVIDIA RTX A5000 24 GB x 6
- NVIDIA driver: R570 family
- Driver-reported CUDA capability: CUDA 12.8
- Python: 3.12

## CUDA compatibility split

The LLM and Speech services intentionally use separate uv environments.

### LLM runtime

A plain PyPI resolution of vLLM 0.28.0 can install `torch 2.13.0+cu130`. That CUDA 13 PyTorch build cannot
initialize CUDA on the school R570 / CUDA-12.x driver.

The repository therefore pins the official x86_64 vLLM 0.28.0 CUDA 12.9 wheel and explicitly routes the
PyTorch family to the official CUDA 12.9 index from `llm_runtime/pyproject.toml`:

```toml
[tool.uv.sources]
torch = { index = "pytorch-cu129" }
torchaudio = { index = "pytorch-cu129" }
torchvision = { index = "pytorch-cu129" }

[[tool.uv.index]]
name = "pytorch-cu129"
url = "https://download.pytorch.org/whl/cu129"
explicit = true
```

This explicit project configuration is important: `UV_TORCH_BACKEND` / `--torch-backend` is intended for
uv's pip interface and must not be relied on to select a CUDA backend during `uv sync` project resolution.

Expected LLM runtime:

```text
vllm=0.28.0
torch=2.13.0+cu129
cuda=12.9
```

### Speech runtime

Speech stays on the PyTorch 2.8 CUDA 12.8 line. `speech_server/pyproject.toml` explicitly routes `torch` and
`torchaudio` to:

```text
https://download.pytorch.org/whl/cu128
```

Expected Speech runtime:

```text
torch=2.8.0+cu128
cuda=12.8
```

Do not merge the LLM and Speech Python environments.

## Dependency management

All Python package management uses **uv**.

Projects:

- root `pyproject.toml`: tests and benchmark client
- `llm_runtime/pyproject.toml`: vLLM runtime
- `speech_server/pyproject.toml`: FastAPI + Qwen ASR/TTS runtime

Speech also has a Transformers packaging override because the official Qwen packages currently declare
different patch versions:

- `qwen-asr==0.0.6` -> `transformers==4.57.6`
- `qwen-tts==0.1.1` -> `transformers==4.57.3`

The Speech environment resolves both with `transformers==4.57.6`.

## 1. Update the repository

```bash
cd ~/workspace/SKKU_AI_model_server
git pull
```

## 2. Verify GPUs and uv

```bash
nvidia-smi
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv
uv --version
```

## 3. Configure environment variables

For a new checkout:

```bash
cp .env.example .env
```

Important defaults:

```dotenv
HF_HOME=${HOME}/.cache/huggingface
MODEL_SERVER_HOST=0.0.0.0

TEXT_MODEL=Qwen/Qwen3.8-27B
TEXT_PORT=8001
TEXT_GPU_IDS=0,1,2,3
TEXT_TENSOR_PARALLEL_SIZE=4
TEXT_MAX_MODEL_LEN=16384

VOICE_MODEL=Qwen/Qwen3.5-9B
VOICE_PORT=8002
VOICE_GPU_IDS=4
VOICE_MAX_MODEL_LEN=8192

ASR_MODEL=Qwen/Qwen3-ASR-0.6B
TTS_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
SPEECH_PORT=8004
SPEECH_GPU_ID=5
SPEECH_TTS_ENABLED=false
QWEN_TTS_ENABLED=true
QWEN_TTS_PORT=8012

EMBEDDING_ENABLED=true
EMBEDDING_PORT=8003

TTS_LANGUAGE=Auto
TTS_SPEAKER=Sohee
```

## Backend.AI Preopen Ports

이 세션의 preopen port는 `8000`, `8001`, `8002`, `8004`입니다
(`echo $BACKENDAI_PREOPEN_PORTS`). Backend.AI 앱 프록시는 컨테이너 인터페이스로
접속하므로 공개할 서비스는 `0.0.0.0`에 바인딩돼야 합니다(`MODEL_SERVER_HOST` 기본값).

| 포트 | 서비스 | 공개(**Open app to public**)가 필요한 경우 |
|---:|---|---|
| 8000 | Course Agent API (`SKKU_AI_agent`, uvicorn) | 브라우저가 API를 직접 호출하도록 `NEXT_PUBLIC_API_BASE_URL`을 이 앱 주소로 둘 때. 웹 UI의 `/api` rewrite로 같은 origin에서 호출하면 공개하지 않아도 됩니다 |
| 8001 | Text LLM | 앱이 같은 세션에 있으면 localhost로 호출합니다. 세션 밖 앱이나 외부 테스트에서만 |
| 8002 | Voice LLM (+ 페이지 이미지 입력) | 같음 |
| 8004 | Speech API (ASR) — `SPEECH_PORT=8004` | 같음 |
| 8012 | Qwen3-TTS (streaming) | preopen port가 아닙니다. 같은 세션의 앱이 localhost로 호출합니다 |
| 8003 | Embedding | preopen port가 아닙니다. 같은 세션의 앱이 localhost로 호출합니다 |

### 앱과 모델 서버가 같은 세션에 있을 때 (기본)

1. `.env`에 비어 있지 않은 `MODEL_SERVER_API_KEY`를 설정하고 `./scripts/start_all.sh`를 실행합니다.
2. `SKKU_AI_agent`의 `.env`는 `SPEECH_BASE_URL=http://localhost:8004`,
   `TTS_BASE_URL=http://localhost:8012`, `EMBEDDING_BASE_URL=http://localhost:8003/v1`을 씁니다.
   모델 서버 쪽에는 **추가 preopen port가 필요하지 않습니다.**
3. 브라우저가 닿아야 하는 것은 웹 UI(그리고 선택적으로 API `8000`)뿐입니다. 웹 UI(Next.js)의
   기본 포트 `3000`은 preopen port가 아니고 이미지가 `ipython` pty 서비스 포트로 예약하고 있으므로,
   SSH 터널이나 VS Code 포트 포워딩으로 열거나 남는 preopen port에서 띄워야 합니다.
   `SKKU_AI_agent/README.md`의 "Backend.AI preopen port"를 참고합니다.

### 앱이 세션 밖에서 돌 때

앱에 노출해야 하는 모델 서버 포트는 다섯 개입니다: `8001`, `8002`, `8004`, `8012`, `8003`.
이 토폴로지에서는 `8000`이 비므로 `EMBEDDING_PORT=8000`으로 돌려 써도 네 개 중 하나가 부족합니다.

- 세션을 다시 만들 때 preopen port를 추가합니다(`8012`, `8003`이 허용되면 그대로 등록).
- 포트를 늘릴 수 없으면 `SPEECH_TTS_ENABLED=true`, `QWEN_TTS_ENABLED=false`로 TTS를 `8004`
  Speech API에 합칩니다. 이 경로는 스트리밍이 없고 CustomVoice 화자(`Sohee` 등)만 받으므로
  앱의 `TTS_SPEAKER=ryan`(voice_clone)은 쓸 수 없고 첫 음성까지의 지연이 늘어납니다.

각 포트를 **Open app to public**으로 열고, 발급된 HTTPS 주소를 앱의 `TEXT_LLM_BASE_URL`,
`VOICE_LLM_BASE_URL`, `SPEECH_BASE_URL`, `TTS_BASE_URL`, `EMBEDDING_BASE_URL`에 넣습니다.
LLM과 Embedding 주소에만 `/v1`을 붙입니다. 모든 앱은 같은 `MODEL_SERVER_API_KEY`로 보호됩니다.

Backend.AI endpoint는 세션이 살아 있는 동안만 유지됩니다. 세션을 다시 만들거나 앱
주소가 바뀌면 애플리케이션 환경변수도 갱신해야 합니다.

### API key

Generate one locally on the server:

```bash
echo "skku-ms-$(openssl rand -hex 32)"
```

Store it only in `.env`:

```dotenv
MODEL_SERVER_API_KEY=skku-ms-...
```

Never commit `.env` or the API key.

For faster Hugging Face downloads, set your own token in `.env`:

```dotenv
HF_TOKEN=hf_...
```

## 4. Download model snapshots

```bash
./scripts/download_models.sh
```

The default persistent cache is:

```text
~/.cache/huggingface
```

Existing snapshots are reused.

## 5. Start all services

If older processes are running, stop them first:

```bash
./scripts/stop_all.sh
```

If an LLM environment was created before the explicit CUDA 12.9 index was added, remove only that Python
environment and its local lock before the first restart:

```bash
rm -rf llm_runtime/.venv
rm -f llm_runtime/uv.lock
```

If needed, do the same for Speech:

```bash
rm -rf speech_server/.venv
rm -f speech_server/uv.lock
```

Do **not** delete `~/.cache/huggingface`; model weights are independent of the Python environments.

Then start:

```bash
./scripts/start_all.sh
```

`start_all.sh` runs `uv sync` for each isolated project, then verifies the actual interpreter inside each
`.venv` before launching model processes.

Expected LLM check:

```text
llm vllm=0.28.0 torch=2.13.0+cu129 cuda=12.9 available=True
llm gpu=NVIDIA RTX A5000
```

Expected Speech check:

```text
speech torch=2.8.0+cu128 cuda=12.8 available=True
speech gpu=NVIDIA RTX A5000
```

The checks call `torch.cuda.set_device(0)`, not only `torch.cuda.is_available()`, so a driver/runtime mismatch
fails before model processes are launched.

After setup, service scripts execute the already prepared binaries directly:

```text
llm_runtime/.venv/bin/vllm
speech_server/.venv/bin/uvicorn
```

They do not invoke another dependency resolution during service startup.

## 6. Watch startup

```bash
tail -f logs/text-llm.log
```

```bash
tail -f logs/voice-llm.log
```

```bash
tail -f logs/speech.log
```

GPU status:

```bash
./scripts/gpu_status.sh
```

## 7. Health checks

```bash
./scripts/healthcheck.sh
```

Endpoints:

```text
GET http://127.0.0.1:8001/v1/models
GET http://127.0.0.1:8002/v1/models
GET http://127.0.0.1:8004/health
```

Speech returns `ready=false` while its models are still loading.

## 8. Debug the environments

LLM:

```bash
CUDA_VISIBLE_DEVICES=4 llm_runtime/.venv/bin/python - <<'PY'
import torch, vllm
print("vLLM:", vllm.__version__)
print("Torch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
torch.cuda.set_device(0)
print("GPU:", torch.cuda.get_device_name(0))
PY
```

Correct LLM result is `torch 2.13.0+cu129` / CUDA `12.9`, not `cu130` / `13.0`.

Speech:

```bash
CUDA_VISIBLE_DEVICES=5 speech_server/.venv/bin/python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
torch.cuda.set_device(0)
print("GPU:", torch.cuda.get_device_name(0))
PY
```

## 9. Text and Voice APIs

Both vLLM services expose OpenAI-compatible APIs:

```text
GET  /v1/models
POST /v1/chat/completions
```

The launch scripts enable:

```text
--reasoning-parser qwen3
--enable-auto-tool-choice
--tool-call-parser qwen3_coder
```

No application system prompt is injected by the model server.

Thinking can be disabled per request, for example:

```json
{
  "model": "Qwen/Qwen3.5-9B",
  "messages": [{"role":"user","content":"짧게 답해줘."}],
  "stream": true,
  "chat_template_kwargs": {"enable_thinking": false}
}
```

## 10. ASR API

```text
POST /v1/audio/transcriptions
Content-Type: multipart/form-data
```

Example:

```bash
curl -X POST http://localhost:8004/v1/audio/transcriptions \
  -H "Authorization: Bearer $MODEL_SERVER_API_KEY" \
  -F 'file=@sample.wav' \
  -F 'language=Korean'
```

Response contract:

```json
{
  "text": "운영체제에서 가상 메모리가 뭐야?",
  "language": "Korean",
  "audio_duration_ms": 2310,
  "inference_ms": 190
}
```

The MVP is utterance-level. Raw PCM16 mono 16 kHz and WAV are supported.

## 11. TTS API

```text
POST /v1/audio/speech
Content-Type: application/json
```

Example:

```bash
curl -X POST http://localhost:8004/v1/audio/speech \
  -H "Authorization: Bearer $MODEL_SERVER_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "input":"가상 메모리는 실제 메모리보다 큰 주소 공간을 제공하는 방식이에요.",
    "voice":"Sohee",
    "language":"Korean",
    "response_format":"wav"
  }' \
  --output output.wav
```

Output:

- `pcm`: PCM16 little-endian, mono, 24 kHz
- `wav`: PCM16 WAV, mono, 24 kHz

## Streaming Qwen3-TTS service (default TTS)

`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` served through `faster-qwen3-tts` (MIT),
which adds CUDA graph capture on top of the Apache-2.0 model. Korean is one of
the model's ten officially supported languages.

Port `8012` is not a Backend.AI preopen port. The application in the same
session reaches it over localhost, so nothing has to be opened; only an
application outside the session needs it published (see
[Backend.AI Preopen Ports](#backendai-preopen-ports)).

```bash
curl --no-buffer http://localhost:8012/v1/audio/speech \
  -H "Authorization: Bearer $MODEL_SERVER_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"input":"안녕하세요","voice":"Sohee","language":"Korean","response_format":"pcm","stream":true}' \
  --output output.pcm
```

The body is chunked PCM16 little-endian, mono, 24 kHz. `hop_len` optionally
overrides `QWEN_TTS_CHUNK_SIZE` for a single request. Set `QWEN_TTS_SPEED=1.2`
for 20% faster speech globally, or send `"speed": 1.2` per request. The accepted
range is 0.5-2.0, and FFmpeg preserves the original pitch.

### Voice consistency: why this runs in clone mode

A turn is synthesized as more than one request (an opening clause, then the rest),
and with a CustomVoice speaker id each request realises the timbre independently.
Measured across requests versus within a single utterance:

| | speaker id | reference audio |
|---|---:|---:|
| spectral-shape distance | 1.52x | **~1.0x** |
| median F0 difference | 11.5 Hz | **6-10 Hz** |
| loudness ratio | 1.47x (max 1.67x) | **1.10x (max 1.27x)** |
| spoken-duration CV | 0.09-0.27 | **0.03-0.07** |

So `QWEN_TTS_MODE=voice_clone` conditions every request on the same recording,
which needs a `*-Base` model plus `QWEN_TTS_REF_WAV` and its exact transcript in
`QWEN_TTS_REF_TEXT`. `custom_voice` keeps the nine built-in timbres and a
`*-CustomVoice` model, and is fine for one-shot synthesis.

`samples/qwen_ryan_reference.wav` is itself rendered with the CustomVoice `ryan`
timbre, which measured cleanest of the nine, then reused as the reference. That
combines the timbre with clone-mode stability. Audio is deployment data and is
not committed, so generate it once per deployment:

```bash
./scripts/make_tts_reference.sh
```

It renders several takes, keeps the one with the least narrowband energy (takes
vary audibly) and prints the transcript to copy into `QWEN_TTS_REF_TEXT`. Change
`QWEN_TTS_REFERENCE_SPEAKER` to base the voice on a different built-in timbre.

Loudness is normalised per request towards `QWEN_TTS_TARGET_RMS_DBFS`. The gain is
estimated from the voiced audio seen so far and locked after a second, so no
buffering is added; estimating from only the first block mis-scaled requests with
a soft onset and made the mismatch worse.

### Silence trimming

The model pads roughly 0.6-0.95 s of silence onto both ends of every utterance.
Left in, that is added latency at the start of a turn and an unnatural pause at
every chunk seam, so the runtime trims it. Only the silence run at the very end
of the stream is dropped; a pause is held back until the next chunk proves it was
internal to the utterance. `QWEN_TTS_SILENCE_THRESHOLD=0` disables this.

### Abandoned streams

The agent drops a TTS response mid-stream on every barge-in. Starlette cancels the
response without closing its body iterator, so the runtime never holds the GPU
lock inside that iterator: synthesis runs on its own thread and the lock is
released whether or not the response is ever finalised. Closing the response stops
that thread on the next block; if it is somehow never closed,
`QWEN_TTS_STREAM_QUEUE_BLOCKS` (default 8) of audio buffer up and the thread gives
up after `QWEN_TTS_STREAM_STALL_TIMEOUT` seconds (default 5). Without this, one
abandoned stream stranded the lock and every later request returned 200 with the
headers and then zero audio.

## 12. Tests and smoke test

Unit/API tests do not load the large models:

```bash
uv sync
uv run pytest
```

After all real services are healthy:

```bash
./scripts/smoke_test.sh
```

## 13. Benchmark

```bash
uv run python scripts/benchmark.py \
  --concurrency 1 \
  --asr-wav /path/to/korean.wav \
  --output benchmark-results/concurrency-1.json
```

Metrics include LLM TTFT/tokens-per-second, ASR/TTS latency and RTF, and GPU utilization snapshots.

## 14. Stop all services

```bash
./scripts/stop_all.sh
```

Model weights remain under `HF_HOME`, so restarting does not require downloading them again.

## Multimodal document inference

`start_all.sh` now also starts `Qwen/Qwen3-VL-Embedding-2B` (BF16, pooling)
with `scripts/start_embedding.sh`, on GPU 5 / port 8003 by default. Like TTS,
the port is localhost-only on Backend.AI and needs no preopen port. Configure
`EMBEDDING_ENABLED`, `EMBEDDING_MODEL`, `EMBEDDING_GPU_IDS`, `EMBEDDING_PORT`,
`EMBEDDING_MAX_MODEL_LEN`, and `EMBEDDING_GPU_MEMORY_UTILIZATION` in `.env`.
The application owns document conversion, page storage, retrieval and prompts.

The existing Qwen3.5-9B server accepts page images when
`VOICE_LANGUAGE_MODEL_ONLY=false` (the new default). Voice generation keeps the
same model. No separate vision-generation service is required.

For this embedding checkpoint, `/v1/embeddings` chat requests must specify
`add_generation_prompt: true` **and** `add_special_tokens: true`. The tokenizer's
terminal `<|endoftext|>` is the pooled token. Omitting it returns valid-shaped but
incorrect embeddings. Verify against official Transformers with:

```bash
uv run --project llm_runtime python scripts/check_embedding_parity.py
```

This explicit live check loads the official model on CPU, compares token IDs and
requires vector cosine > 0.99. It is not part of offline unit tests.
