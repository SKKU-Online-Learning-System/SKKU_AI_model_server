# SKKU AI Model Server

Independent, stateless GPU inference server for **SKKU Course Agent**.

This repository serves only models. Application logic remains in
`SKKU-Online-Learning-System/SKKU_AI_agent` and is intentionally not implemented here.

## Architecture

| Service | Model | Physical GPU | Runtime | Port |
|---|---|---:|---|---:|
| Text LLM | `Qwen/Qwen3.8-27B` | 0,1,2,3 | vLLM, BF16, TP=4 | 8001 |
| Voice LLM | `Qwen/Qwen3.5-9B` | 4 | vLLM, BF16, TP=1 | 8002 |
| ASR | `Qwen/Qwen3-ASR-0.6B` | 5 | official `qwen-asr` | 8010 |
| TTS | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | 5 | official `qwen-tts` | 8010 |

The speech service is one FastAPI service. ASR and TTS are loaded sequentially into the same process and
share one inference lock. This matches the current single-user development workload and avoids unnecessary
microservices or ASR/TTS GPU duplication.

The expected application URLs are:

```text
TEXT_LLM_BASE_URL=http://<MODEL_SERVER>:8001/v1
VOICE_LLM_BASE_URL=http://<MODEL_SERVER>:8002/v1
SPEECH_BASE_URL=http://<MODEL_SERVER>:8010
```

## Explicit non-goals

This repository does **not** implement RAG, PostgreSQL/pgvector, JWT/RBAC, course logic, External Brain,
tool execution, trusted web search, visualization, Silero VAD, turn-taking, barge-in orchestration,
WebSocket session orchestration, frontend logic, or chat history.

## Target hardware

Current school development server target:

- NVIDIA RTX A5000 24 GB x 6
- Driver 570.169
- CUDA 12.8
- CPU: 32 cores recommended
- RAM: 128-160 GiB recommended
- Docker shared memory: 32 GiB

Current Docker base image configured by default:

```text
runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2404-cluster
```

The `runpod/` namespace is only the image name; the deployment target is the school GPU server.
Override `BASE_IMAGE` in `.env` if the server uses another compatible CUDA 12.8 image.

## Dependency management: uv

All Python dependency management uses **uv**.

There are three projects:

- root `pyproject.toml`: developer tests and benchmark client
- `llm_runtime/pyproject.toml`: pinned vLLM runtime
- `speech_server/pyproject.toml`: FastAPI + Qwen ASR/TTS runtime

Pinned primary runtime packages:

```text
vllm==0.28.0
qwen-asr==0.0.6
qwen-tts==0.1.1
transformers==4.57.6  # speech runtime override
```

### Why Speech uses a uv override

The official packages currently have a packaging conflict:

- `qwen-asr==0.0.6` declares `transformers==4.57.6`
- `qwen-tts==0.1.1` declares `transformers==4.57.3`

The speech project therefore uses `tool.uv.override-dependencies` to resolve both official packages with
`transformers==4.57.6`. Keep this override isolated to `speech_server`; do not apply it to the vLLM project.

After dependency resolution has been verified on the school server, generate lock files with:

```bash
uv lock
uv lock --project llm_runtime
uv lock --project speech_server
```

The Dockerfiles intentionally use `uv sync` and can bootstrap from the pinned `pyproject.toml` files before
lock files exist. Once lock files are committed, change the image build to `uv sync --frozen` for stricter
reproducibility.

## 1. Verify the school server

From the repository root:

```bash
nvidia-smi
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv

docker --version
docker compose version

docker run --rm --gpus all \
  runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2404-cluster \
  nvidia-smi
```

You should see six RTX A5000 GPUs and CUDA available inside the container.

## 2. Configure environment variables

```bash
cp .env.example .env
```

Important defaults:

```dotenv
HF_HOME=/models/huggingface
TEXT_GPU_IDS=0,1,2,3
VOICE_GPU_IDS=4
SPEECH_GPU_ID=5
TEXT_MAX_MODEL_LEN=16384
VOICE_MAX_MODEL_LEN=8192
TTS_LANGUAGE=Korean
TTS_SPEAKER=Sohee
```

All model IDs, GPU mappings, ports, context limits, and the optional API key can be changed in `.env`.

## 3. Prepare the persistent Hugging Face cache

Model weights are never stored in Git. Create the host cache once:

```bash
sudo mkdir -p /models/huggingface
sudo chown -R "$USER":"$USER" /models/huggingface
```

Download configured models using uvx:

```bash
./scripts/download_models.sh
```

Hugging Face snapshots already present in `HF_HOME` are reused rather than downloaded from scratch on each
container restart.

## 4. Build and start services

Start only the Text LLM:

```bash
docker compose up -d --build text-llm
```

Start only the Voice LLM:

```bash
docker compose up -d --build voice-llm
```

Start only Speech:

```bash
docker compose up -d --build speech
```

Start everything:

```bash
./scripts/start_all.sh
```

Inspect logs without logging request bodies:

```bash
docker compose logs -f text-llm
docker compose logs -f voice-llm
docker compose logs -f speech
```

## 5. Health checks

```bash
./scripts/healthcheck.sh
./scripts/gpu_status.sh
```

Endpoints:

```text
GET http://localhost:8001/v1/models
GET http://localhost:8002/v1/models
GET http://localhost:8010/health
```

Speech `/health` returns `ready=false` while the models are loading. A successful ready response resembles:

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

## 6. Text/Voice LLM APIs

Both vLLM services expose OpenAI-compatible endpoints, including:

```text
GET  /v1/models
POST /v1/chat/completions
```

The launch scripts enable the current Qwen3 reasoning and tool parsers:

```text
--reasoning-parser qwen3
--enable-auto-tool-choice
--tool-call-parser qwen3_coder
```

The servers do not inject a system prompt. Application-provided messages pass through unchanged.

### Request-level thinking disable

For latency-sensitive Voice Agent requests, disable thinking per request rather than hard-coding a server
system prompt:

```json
{
  "model": "Qwen/Qwen3.5-9B",
  "messages": [{"role": "user", "content": "짧게 답해줘."}],
  "stream": true,
  "chat_template_kwargs": {"enable_thinking": false}
}
```

Current vLLM also accepts reasoning effort controls; the application should choose the behavior request by
request.

## 7. ASR API

```text
POST /v1/audio/transcriptions
Content-Type: multipart/form-data
```

Input:

- `file=<audio>`
- optional `language=Korean`
- raw PCM: PCM16, mono, 16 kHz
- WAV: mono/stereo and other sample rates are normalized to mono 16 kHz before inference

Example:

```bash
curl -X POST http://localhost:8010/v1/audio/transcriptions \
  -F 'file=@sample.wav' \
  -F 'language=Korean'
```

Response:

```json
{
  "text": "운영체제에서 가상 메모리가 뭐야?",
  "language": "Korean",
  "audio_duration_ms": 2310,
  "inference_ms": 190
}
```

This MVP is utterance-level. `ASRService` is isolated so a future streaming backend can replace it without
changing the public HTTP contract.

## 8. TTS API

```text
POST /v1/audio/speech
Content-Type: application/json
```

Example:

```bash
curl -X POST http://localhost:8010/v1/audio/speech \
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

The service uses official `generate_custom_voice`. It does not fake streaming when the selected backend call
is utterance-level. `TTSService` is isolated so a true streaming backend can replace it later.

## 9. Optional Bearer authentication

The server is intended for a private network. Set:

```dotenv
MODEL_SERVER_API_KEY=your-secret
```

Then all three services require:

```text
Authorization: Bearer your-secret
```

Leave the value empty for authentication-free local development. No browser CORS middleware is enabled.

## 10. Unit/API contract tests

Tests never download the large models. They use fake ASR/TTS backends.

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

## 11. GPU smoke test

After all services are healthy:

```bash
uv sync
./scripts/smoke_test.sh
```

It verifies:

1. Text LLM Korean completion
2. Text streaming chunk
3. Text tool calling
4. Voice LLM streaming
5. Korean Sohee TTS
6. Korean ASR
7. Speech health and PCM16/mono/24 kHz contract

By default, the script synthesizes its own Korean WAV with the TTS service and then transcribes it. To use an
independent Korean WAV:

```bash
./scripts/smoke_test.sh --asr-wav /path/to/korean.wav
```

## 12. Benchmark

Default single-user benchmark:

```bash
uv run python scripts/benchmark.py \
  --concurrency 1 \
  --asr-wav /path/to/korean.wav \
  --output benchmark-results/concurrency-1.json
```

Optional development probes:

```bash
uv run python scripts/benchmark.py --concurrency 2 --asr-wav /path/to/korean.wav
uv run python scripts/benchmark.py --concurrency 4 --asr-wav /path/to/korean.wav
```

Metrics:

- Text LLM: TTFT, output tokens/sec, total latency
- Voice LLM: TTFT, output tokens/sec, total latency
- ASR: inference latency, audio duration, RTF
- TTS: generation latency, generated audio duration, RTF
- GPU: VRAM usage/utilization snapshot before and after

Do not publish benchmark numbers until they are measured on the actual school A5000 server.

## 13. Shutdown

```bash
./scripts/stop_all.sh
```

The Hugging Face cache remains under `/models/huggingface`, so restarting the containers does not require
redownloading model weights.
