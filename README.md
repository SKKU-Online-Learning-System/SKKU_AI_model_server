# SKKU AI Model Server

Independent, stateless GPU inference server for **SKKU Course Agent**.

This repository serves models only. Application logic remains in `SKKU_AI_agent`.

## Architecture

| Service | Model | GPU | Runtime | Port |
|---|---|---:|---|---:|
| Text LLM | `Qwen/Qwen3.8-27B` | 0,1,2,3 | vLLM 0.28.0, BF16, TP=4 | 8001 |
| Voice LLM | `Qwen/Qwen3.5-9B` | 4 | vLLM 0.28.0, BF16, TP=1 | 8002 |
| ASR | `Qwen/Qwen3-ASR-0.6B` | 5 | official `qwen-asr` | 8010 |
| TTS | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | 5 | official `qwen-tts` | 8010 |

Expected application URLs:

```text
TEXT_LLM_BASE_URL=http://<MODEL_SERVER>:8001/v1
VOICE_LLM_BASE_URL=http://<MODEL_SERVER>:8002/v1
SPEECH_BASE_URL=http://<MODEL_SERVER>:8010
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
└── GPU 5        Speech      :8010
    ├── ASR
    └── TTS
```

Lifecycle scripts:

```text
scripts/start_all.sh
scripts/stop_all.sh
scripts/healthcheck.sh
scripts/gpu_status.sh
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

### Important CUDA compatibility split

The LLM and Speech services intentionally use separate uv environments.

#### LLM runtime

A plain PyPI resolution of `vllm==0.28.0` can select the CUDA 13 build and install a PyTorch build such as
`torch 2.13.0+cu130`. That build cannot initialize CUDA on the school R570 / CUDA-12.x driver.

The repository therefore pins the official x86_64 **vLLM 0.28.0 CUDA 12.9 wheel** directly:

```text
vllm-0.28.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl
```

and syncs the LLM environment with:

```bash
UV_TORCH_BACKEND=cu129 uv sync --project llm_runtime --no-dev
```

The default is also exposed as:

```dotenv
VLLM_TORCH_BACKEND=cu129
```

CUDA 12.9 is in the CUDA 12.x minor-version compatibility family used by the school R570 driver.

#### Speech runtime

Speech remains pinned to the CUDA 12.8 PyTorch line:

```text
torch==2.8.0
torchaudio==2.8.0
```

and is synced with:

```bash
UV_TORCH_BACKEND=cu128 uv sync --project speech_server --no-dev
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
HF_HOME=/models/huggingface
MODEL_SERVER_HOST=0.0.0.0
VLLM_TORCH_BACKEND=cu129

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
/models/huggingface
```

Existing snapshots are reused.

## 5. Start all services

If older processes are running, stop them first:

```bash
./scripts/stop_all.sh
```

Then:

```bash
./scripts/start_all.sh
```

Before launching the services, `start_all.sh` performs two real CUDA preflight checks.

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

Large models can take several minutes to load after the processes start.

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
GET http://127.0.0.1:8010/health
```

Speech returns `ready=false` while its models are still loading.

## 8. Debug the LLM environment

Check exactly what uv installed:

```bash
uv run --project llm_runtime python - <<'PY'
import torch, vllm
print("vLLM:", vllm.__version__)
print("Torch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
torch.cuda.set_device(0)
print("GPU:", torch.cuda.get_device_name(0))
PY
```

Correct result should be CUDA 12.9, not CUDA 13.0.

If an older LLM environment was created before the cu129 pin was added, reset only that environment:

```bash
rm -rf llm_runtime/.venv
rm -f llm_runtime/uv.lock
UV_TORCH_BACKEND=cu129 uv sync --project llm_runtime --no-dev
```

Do not delete the Hugging Face model cache; the downloaded model weights can be reused.

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
curl -X POST http://localhost:8010/v1/audio/transcriptions \
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

Output:

- `pcm`: PCM16 little-endian, mono, 24 kHz
- `wav`: PCM16 WAV, mono, 24 kHz

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
