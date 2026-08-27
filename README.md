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

## Execution modes on the school server

There are two different ways this repository can be run. Check which environment you are currently in
before using the startup commands.

### Mode A: school host shell with Docker daemon

Use Docker Compose only when the current shell can reach the host Docker daemon.

Verify:

```bash
ls -l /var/run/docker.sock
docker info
ps -p 1 -o comm=
```

A usable Docker host normally has `/var/run/docker.sock`, and `docker info` shows both the Client and Server
sections without a connection error.

In this mode, the Docker Compose commands in this README are valid.

### Mode B: already inside the school GPU container

The current development environment may already be a Docker container created by the school server. A
common signature is:

```text
$ ls -l /var/run/docker.sock
ls: cannot access '/var/run/docker.sock': No such file or directory

$ docker info
Client: Docker Engine - Community
...
Server:
failed to connect to the docker API at unix:///var/run/docker.sock

$ ps -p 1 -o comm=
docker-init
```

This means the Docker CLI is installed, but the container does **not** have access to the host Docker daemon.
The effective layout is:

```text
school GPU server host
└── existing development GPU container
    └── SKKU_AI_model_server
```

Do **not** try to solve this by running Docker-in-Docker. In particular, the following commands will fail in
this mode because they eventually require `/var/run/docker.sock`:

```bash
docker compose up -d --build text-llm
docker compose up -d
./scripts/start_all.sh
```

Instead, run the model processes directly inside the existing GPU container with **uv**. The intended layout
is:

```text
existing school GPU container
├── GPU 0,1,2,3  Text LLM    :8001
├── GPU 4        Voice LLM   :8002
└── GPU 5        Speech      :8010
```

> Note: the current `start_all.sh` is Docker-Compose based. Until the lifecycle scripts are changed to
> support direct-process mode, use the direct uv commands below when working inside the existing school GPU
> container.

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

First check the GPUs:

```bash
nvidia-smi
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv
```

Then identify whether you are on the Docker host or already inside a container:

```bash
ls -l /var/run/docker.sock
docker info
ps -p 1 -o comm=
```

If you are on a Docker-capable host, you can additionally verify the configured base image:

```bash
docker --version
docker compose version

docker run --rm --gpus all \
  runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2404-cluster \
  nvidia-smi
```

If `/var/run/docker.sock` does not exist and PID 1 is `docker-init`, skip the Docker commands and use the
direct uv execution section below.

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

Load `.env` into the current shell before direct uv execution:

```bash
set -a
source .env
set +a
```

## 3. Prepare the persistent Hugging Face cache

Model weights are never stored in Git.

On a host where `/models/huggingface` is writable, create the cache once:

```bash
sudo mkdir -p /models/huggingface
sudo chown -R "$USER":"$USER" /models/huggingface
```

If the current school container already mounts a persistent model cache, use the mounted path instead and
set `HF_HOME` accordingly in `.env`.

Download configured models using uvx:

```bash
./scripts/download_models.sh
```

Hugging Face snapshots already present in `HF_HOME` are reused rather than downloaded from scratch on each
restart.

## 4. Start services

### 4A. Docker Compose mode

Use this section only from a shell where `docker info` successfully connects to the Docker Server.

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

### 4B. Direct uv mode inside the existing school GPU container

Use this section when `/var/run/docker.sock` is missing and the current process tree indicates that you are
already inside the GPU container.

From the repository root:

```bash
cd ~/workspace/SKKU_AI_model_server

set -a
source .env
set +a
```

#### Text LLM - GPU 0,1,2,3

Install/sync the LLM runtime:

```bash
uv sync --project llm_runtime
```

Start the Text LLM:

```bash
export CUDA_VISIBLE_DEVICES="${TEXT_GPU_IDS:-0,1,2,3}"

uv run --project llm_runtime \
  vllm serve "${TEXT_MODEL:-Qwen/Qwen3.8-27B}" \
  --host "${MODEL_SERVER_HOST:-0.0.0.0}" \
  --port "${TEXT_PORT:-8001}" \
  --tensor-parallel-size "${TEXT_TENSOR_PARALLEL_SIZE:-4}" \
  --max-model-len "${TEXT_MAX_MODEL_LEN:-16384}" \
  --gpu-memory-utilization "${TEXT_GPU_MEMORY_UTILIZATION:-0.90}" \
  --dtype bfloat16 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --disable-log-requests \
  --enable-request-id-headers \
  --language-model-only \
  ${MODEL_SERVER_API_KEY:+--api-key "$MODEL_SERVER_API_KEY"}
```

After loading finishes, verify from another shell:

```bash
curl http://localhost:8001/v1/models \
  ${MODEL_SERVER_API_KEY:+-H "Authorization: Bearer $MODEL_SERVER_API_KEY"}
```

#### Voice LLM - GPU 4

Start from another shell after loading `.env`:

```bash
export CUDA_VISIBLE_DEVICES="${VOICE_GPU_IDS:-4}"

uv run --project llm_runtime \
  vllm serve "${VOICE_MODEL:-Qwen/Qwen3.5-9B}" \
  --host "${MODEL_SERVER_HOST:-0.0.0.0}" \
  --port "${VOICE_PORT:-8002}" \
  --tensor-parallel-size 1 \
  --max-model-len "${VOICE_MAX_MODEL_LEN:-8192}" \
  --gpu-memory-utilization "${VOICE_GPU_MEMORY_UTILIZATION:-0.88}" \
  --dtype bfloat16 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --disable-log-requests \
  --enable-request-id-headers \
  --language-model-only \
  ${MODEL_SERVER_API_KEY:+--api-key "$MODEL_SERVER_API_KEY"}
```

#### Speech server - GPU 5

Sync the Speech runtime:

```bash
uv sync --project speech_server
```

Then start the FastAPI server:

```bash
export CUDA_VISIBLE_DEVICES="${SPEECH_GPU_ID:-5}"

uv run --project speech_server \
  uvicorn speech_server.main:app \
  --host "${MODEL_SERVER_HOST:-0.0.0.0}" \
  --port "${SPEECH_PORT:-8010}" \
  --no-access-log
```

The three commands above are foreground processes. During development, keep them in separate terminal/tmux/
zellij panes. A future direct-process lifecycle script can wrap them with `nohup`/PID files when persistent
background execution is needed.

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

The launch configuration enables the current Qwen3 reasoning and tool parsers:

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

A recommended 256-bit development key can be generated on the server with:

```bash
echo "skku-ms-$(openssl rand -hex 32)"
```

Then all three services require:

```text
Authorization: Bearer your-secret
```

Leave the value empty for authentication-free local development. Never commit the actual key to Git. No
browser CORS middleware is enabled.

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

### Docker Compose mode

```bash
./scripts/stop_all.sh
```

### Direct uv mode

The model servers are foreground processes. Stop each process with `Ctrl-C` in its terminal/tmux/zellij pane.
If you later run them with `nohup`, keep explicit PID files and terminate those PIDs rather than using broad
`pkill` commands.

The Hugging Face cache remains under the configured `HF_HOME`, so restarting the model processes does not
require redownloading model weights.
