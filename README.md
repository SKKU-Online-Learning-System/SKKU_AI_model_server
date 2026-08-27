# SKKU AI Model Server

Independent, stateless GPU inference server for **SKKU Course Agent**.

This repository serves models only. Application logic remains in
`SKKU-Online-Learning-System/SKKU_AI_agent` and is intentionally not implemented here.

## Architecture

| Service | Model | Physical GPU | Runtime | Port |
|---|---|---:|---|---:|
| Text LLM | `Qwen/Qwen3.8-27B` | 0,1,2,3 | vLLM, BF16, TP=4 | 8001 |
| Voice LLM | `Qwen/Qwen3.5-9B` | 4 | vLLM, BF16, TP=1 | 8002 |
| ASR | `Qwen/Qwen3-ASR-0.6B` | 5 | official `qwen-asr` | 8010 |
| TTS | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | 5 | official `qwen-tts` | 8010 |

The Speech service is one FastAPI process. ASR and TTS are loaded in the same process and share one
inference lock. This matches the current single-user development workload and avoids allocating separate
GPUs to ASR and TTS.

Expected application URLs:

```text
TEXT_LLM_BASE_URL=http://<MODEL_SERVER>:8001/v1
VOICE_LLM_BASE_URL=http://<MODEL_SERVER>:8002/v1
SPEECH_BASE_URL=http://<MODEL_SERVER>:8010
```

## Repository boundary

This repository does **not** implement RAG, PostgreSQL/pgvector, JWT/RBAC, course logic, External Brain,
tool execution, trusted web search, visualization, Silero VAD, turn-taking, barge-in orchestration,
WebSocket session orchestration, frontend logic, or chat history.

Those responsibilities belong to `SKKU_AI_agent`.

## Runtime model: uv only

The school development session is already provided inside a GPU-enabled environment. This repository runs
all model processes **directly with uv**.

There is no Docker Compose runtime and no Docker-in-Docker requirement. `/var/run/docker.sock` is not
required.

The runtime layout is:

```text
school GPU environment
├── GPU 0,1,2,3  Text LLM    :8001
├── GPU 4        Voice LLM   :8002
└── GPU 5        Speech      :8010
    ├── Qwen3-ASR-0.6B
    └── Qwen3-TTS-12Hz-0.6B-CustomVoice
```

Service lifecycle is managed directly by:

```text
scripts/start_all.sh
scripts/stop_all.sh
```

`start_all.sh` launches each service in its own process group and stores:

```text
.run/text-llm.pid
.run/voice-llm.pid
.run/speech.pid

logs/text-llm.log
logs/voice-llm.log
logs/speech.log
```

Both directories are ignored by Git.

## Target hardware

Current school development server target:

- NVIDIA RTX A5000 24 GB x 6
- Driver 570.169
- CUDA 12.8
- CPU: 32 cores recommended
- RAM: 128-160 GiB recommended

Current school GPU environment is based on CUDA 12.8.1 / PyTorch 2.8.0 / Ubuntu 24.04.

## Dependency management

All Python dependency management uses **uv**.

There are three uv projects:

- root `pyproject.toml`: tests and benchmark client
- `llm_runtime/pyproject.toml`: vLLM runtime
- `speech_server/pyproject.toml`: FastAPI + Qwen ASR/TTS runtime

Pinned primary packages:

```text
vllm==0.28.0
qwen-asr==0.0.6
qwen-tts==0.1.1
transformers==4.57.6  # speech runtime override
```

### Qwen ASR/TTS dependency conflict

The official packages currently declare different Transformers patch versions:

- `qwen-asr==0.0.6` -> `transformers==4.57.6`
- `qwen-tts==0.1.1` -> `transformers==4.57.3`

The Speech project therefore keeps a **local uv override** for `transformers==4.57.6`. This override is
isolated to `speech_server` and must not be applied to the vLLM environment.

## 1. Clone or update the repository

```bash
git clone https://github.com/SKKU-Online-Learning-System/SKKU_AI_model_server.git
cd SKKU_AI_model_server
```

If it is already cloned:

```bash
git pull
```

## 2. Verify the GPU environment

```bash
nvidia-smi
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv
```

Verify uv:

```bash
uv --version
```

The projects require Python 3.12 or newer. If an appropriate interpreter is not already available:

```bash
uv python install 3.12
```

## 3. Configure environment variables

```bash
cp .env.example .env
```

Important defaults:

```dotenv
HF_HOME=/models/huggingface
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
SPEECH_PORT=8010
SPEECH_GPU_ID=5

TTS_LANGUAGE=Korean
TTS_SPEAKER=Sohee
```

All model IDs, GPU mappings, ports, context limits, and the optional API key are configurable through
`.env`.

### API key

Generate a private model-server key, for example:

```bash
echo "skku-ms-$(openssl rand -hex 32)"
```

Then place it only in `.env`:

```dotenv
MODEL_SERVER_API_KEY=skku-ms-...
```

Never commit `.env` or the API key.

## 4. Prepare the Hugging Face cache

The configured default is:

```text
/models/huggingface
```

Make sure the directory exists and is writable by the current user. If `/models` is not writable, point
`HF_HOME` in `.env` to another persistent directory provided by the school environment.

For example:

```bash
mkdir -p "$HOME/.cache/huggingface"
```

and:

```dotenv
HF_HOME=/home/<USER>/.cache/huggingface
```

Model weights are never stored in Git.

Pre-download the configured models:

```bash
./scripts/download_models.sh
```

Existing Hugging Face snapshots are reused.

## 5. Sync the uv environments

The all-in-one startup script performs these syncs automatically, but they can also be run explicitly:

```bash
uv sync --project llm_runtime --no-dev
uv sync --project speech_server --no-dev
```

For repository tests:

```bash
uv sync
```

After dependency compatibility is verified on the school server, lock the projects for reproducibility:

```bash
uv lock
uv lock --project llm_runtime
uv lock --project speech_server
```

## 6. Start all model services

```bash
./scripts/start_all.sh
```

The script:

1. loads `.env`
2. verifies `uv`, `nvidia-smi`, and `setsid`
3. syncs the LLM and Speech uv projects
4. launches Text LLM on GPU 0-3
5. launches Voice LLM on GPU 4
6. launches Speech on GPU 5
7. writes PID files under `.run/`
8. writes logs under `logs/`

Launching a process does not mean the model is immediately ready. Large models can take several minutes to
load.

### Watch logs

```bash
tail -f logs/text-llm.log
```

```bash
tail -f logs/voice-llm.log
```

```bash
tail -f logs/speech.log
```

### Inspect GPU usage

```bash
./scripts/gpu_status.sh
```

## 7. Start a service manually

The individual scripts run in the foreground and are useful for debugging.

Text LLM:

```bash
./scripts/start_text_llm.sh
```

Voice LLM:

```bash
./scripts/start_voice_llm.sh
```

Speech:

```bash
./scripts/start_speech.sh
```

The scripts calculate the repository root dynamically; there is no fixed `/app/...` runtime path.

## 8. Health checks

After the models finish loading:

```bash
./scripts/healthcheck.sh
```

Endpoints:

```text
GET http://localhost:8001/v1/models
GET http://localhost:8002/v1/models
GET http://localhost:8010/health
```

Speech `/health` returns `ready=false` while ASR/TTS are still loading.

Example ready response:

```json
{
  "status": "ok",
  "ready": true,
  "gpu": "NVIDIA RTX A5000",
  "asr_model": "Qwen/Qwen3-ASR-0.6B",
  "tts_model": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
  "error": null
}
```

## 9. Text and Voice LLM APIs

Both vLLM services expose OpenAI-compatible endpoints:

```text
GET  /v1/models
POST /v1/chat/completions
```

The launch scripts enable the current Qwen reasoning/tool parsers:

```text
--reasoning-parser qwen3
--enable-auto-tool-choice
--tool-call-parser qwen3_coder
```

The server does not inject its own application system prompt.

### Disable thinking per request

For latency-sensitive Voice Agent requests:

```json
{
  "model": "Qwen/Qwen3.5-9B",
  "messages": [
    {"role": "user", "content": "짧게 답해줘."}
  ],
  "stream": true,
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}
```

## 10. ASR API

```text
POST /v1/audio/transcriptions
Content-Type: multipart/form-data
```

Input:

- `file=<audio>`
- optional `language=Korean`
- raw PCM: PCM16, mono, 16 kHz
- WAV is also supported and normalized before inference

Example:

```bash
curl -X POST http://localhost:8010/v1/audio/transcriptions \
  -H "Authorization: Bearer $MODEL_SERVER_API_KEY" \
  -F 'file=@sample.wav' \
  -F 'language=Korean'
```

Example response:

```json
{
  "text": "운영체제에서 가상 메모리가 뭐야?",
  "language": "Korean",
  "audio_duration_ms": 2310,
  "inference_ms": 190
}
```

The MVP API is utterance-level. `ASRService` is isolated so a future streaming backend can replace it without
changing the public HTTP contract.

## 11. TTS API

```text
POST /v1/audio/speech
Content-Type: application/json
```

Example:

```bash
curl -X POST http://localhost:8010/v1/audio/speech \
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

Supported output formats:

- `pcm`: raw PCM16 little-endian, mono, 24 kHz
- `wav`: PCM16 WAV, mono, 24 kHz

The service uses the official Qwen CustomVoice implementation. `TTSService` remains isolated so the backend
can be replaced later without changing the API contract.

## 12. Authentication

When `MODEL_SERVER_API_KEY` is non-empty, all model APIs require:

```text
Authorization: Bearer <MODEL_SERVER_API_KEY>
```

For example:

```bash
curl http://localhost:8001/v1/models \
  -H "Authorization: Bearer $MODEL_SERVER_API_KEY"
```

Leave `MODEL_SERVER_API_KEY` empty only for authentication-free development inside a trusted environment.

## 13. Unit and API-contract tests

Tests do not download the large models. They use mocked ASR/TTS backends.

```bash
uv sync
uv run pytest
```

Covered contracts include:

- config parsing
- `/health`
- optional authentication
- malformed audio
- ASR response schema
- TTS request validation
- invalid speaker
- invalid output format
- PCM16 mono 24 kHz output

## 14. GPU smoke test

After all services are healthy:

```bash
uv sync
./scripts/smoke_test.sh
```

It verifies:

1. Text LLM Korean completion
2. Text streaming
3. Text tool calling
4. Voice LLM streaming
5. Korean Sohee TTS
6. Korean ASR
7. PCM16 / mono / 24 kHz Speech contract

To use an independent Korean WAV:

```bash
./scripts/smoke_test.sh --asr-wav /path/to/korean.wav
```

## 15. Benchmark

Default single-user benchmark:

```bash
uv run python scripts/benchmark.py \
  --concurrency 1 \
  --asr-wav /path/to/korean.wav \
  --output benchmark-results/concurrency-1.json
```

Metrics:

- Text LLM: TTFT, output tokens/sec, total latency
- Voice LLM: TTFT, output tokens/sec, total latency
- ASR: inference latency, audio duration, RTF
- TTS: generation latency, generated audio duration, RTF
- GPU: VRAM usage/utilization snapshot

Do not publish benchmark values until they are measured on the actual school A5000 server.

## 16. Stop all services

```bash
./scripts/stop_all.sh
```

The script reads the `.run/*.pid` files, sends `SIGTERM` to each process group, waits up to 30 seconds, and
uses `SIGKILL` only if a process does not stop cleanly.

Model weights remain in `HF_HOME`, so restarting the services reuses the existing cache.
